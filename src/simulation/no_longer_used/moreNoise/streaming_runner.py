"""
Streaming (O(log N)) purification runner using Qiskit circuits.

Implements two modes:

1) REGULAR streaming mode:
   - streams N noisy copies and merges them with the O(log N) stack.
   - NOTE: If you enable Clifford twirling here and generate different twirls per
     incoming copy, merges will generally violate the "identical inputs" SWAP-test
     assumption unless you do careful subtree-consistent pairing.

2) ITERATIVE mode (Option B implemented here):
   - Start from perfect |ψ⟩.
   - For each iteration t:
       (a) Apply noise ONCE to the current state to produce a single noisy density matrix.
           If dephase_z and twirling enabled: apply a *global single-qubit Clifford* C_t
           to ALL qubits (same gate on each qubit), apply Z-dephasing, then undo C_t.
           The Clifford C_t cycles deterministically over {I, H, S H} across iterations.
       (b) Create 2^ℓ IDENTICAL copies of that noisy density matrix.
       (c) Run ℓ clean SWAP-purification levels (no intermediate noise).
       (d) The purified output becomes the new current state.

This satisfies:
- effective-channel approach (uses the Kraus Z-dephasing channel directly; no “effective depol” shortcut),
- noise applied once per iteration (not per SWAP round),
- ℓ > 1 supported (uses 2^ℓ identical inputs per iteration).

IMPORTANT:
- The SWAP-test purification theory assumes the two inputs at each merge are identical.
  Iterative mode below enforces this by cloning the *same* noisy density matrix 2^ℓ times.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import itertools
from functools import lru_cache

from qiskit import transpile
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit.quantum_info.operators import Pauli

# Aer simulator import compatible across Qiskit versions
try:  # qiskit-aer >= 0.12
    from qiskit_aer import AerSimulator
except Exception:  # pragma: no cover
    from qiskit.providers.aer import AerSimulator  # type: ignore

from .configs import NoiseMode, NoiseType, RunSpec
from .state_factory import build_target
from .noise_engine import build_noisy_copy, sample_error_pattern, ErrorPattern, apply_noise_to_density_matrix
from .amplified_swap import purify_two_from_density

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# -----------------------------
# Backend helpers
# -----------------------------

def _make_backend(method: str) -> AerSimulator:
    return AerSimulator(method=method)


def _density_from_circuit(qc, backend: AerSimulator) -> DensityMatrix:
    """Execute circuit and return resulting density matrix."""
    qc2 = qc.copy()
    qc2.save_density_matrix()
    t = transpile(qc2, backend)
    res = backend.run(t, shots=1024).result()
    data0 = res.data(0)

    rho = data0.get("density_matrix")
    if rho is None:
        rho = data0.get("statevector")

    dm = DensityMatrix(rho)
    logger.debug(f"Generated density matrix: dim={dm.dim}, purity={_purity(dm):.6f}")
    return dm


# -----------------------------
# Metric helpers
# -----------------------------

def _fidelity_to_pure(rho: DensityMatrix, psi: Statevector) -> float:
    """F = ⟨ψ|ρ|ψ⟩"""
    v = psi.data.reshape((-1, 1))
    return float(np.real(np.conj(v).T @ (rho.data @ v)))


def _trace_distance_to_pure(rho: DensityMatrix, psi: Statevector) -> float:
    """ε_L = (1/2) ||ρ - |ψ⟩⟨ψ|||_1"""
    proj = np.outer(psi.data, np.conj(psi.data))
    diff = rho.data - proj
    # Hermitian; trace norm = sum of abs eigenvalues
    evals = np.linalg.eigvalsh((diff + diff.conj().T) / 2.0)
    return 0.5 * float(np.sum(np.abs(evals)))


def _purity(rho: DensityMatrix) -> float:
    """Tr(ρ²)"""
    return float(np.real(np.trace(rho.data @ rho.data)))


def _bloch_vector_magnitude(rho: DensityMatrix) -> Optional[float]:
    """For single-qubit states, compute |r⃗| where ρ = (I + r⃗·σ⃗)/2."""
    if rho.dim != 2:
        return None
    rx = np.real(np.trace(rho.data @ Pauli("X").to_matrix()))
    ry = np.real(np.trace(rho.data @ Pauli("Y").to_matrix()))
    rz = np.real(np.trace(rho.data @ Pauli("Z").to_matrix()))
    return float(np.sqrt(rx**2 + ry**2 + rz**2))


# -----------------------------
# Local deterministic Clifford twirl (exact average over {I,H,HS}^⊗M)
# -----------------------------

def _U_single_qubit(gate: str) -> np.ndarray:
    """Single-qubit unitary for gate in {'i','h','hs'} with hs := HS."""
    if gate == "i":
        return np.eye(2, dtype=complex)
    if gate == "h":
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    if gate == "hs":
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        S = np.array([[1, 0], [0, 1j]], dtype=complex)
        return H @ S  # HS
    raise ValueError(f"Unknown global Clifford gate '{gate}'")

@lru_cache(maxsize=None)
def _all_local_clifford_combos(M: int) -> Tuple[Tuple[str, ...], ...]:
    """All tuples (g0,...,g_{M-1}) with gj in {'i','h','hs'}."""
    gates = ("i", "h", "hs")
    return tuple(itertools.product(gates, repeat=M))

@lru_cache(maxsize=None)
def _all_local_unitaries(M: int) -> Tuple[np.ndarray, ...]:
    """Precompute U = ⊗_j U(gj) for all combos in {'i','h','hs'}^M."""
    combos = _all_local_clifford_combos(M)
    U_list: List[np.ndarray] = []
    for combo in combos:
        U = np.array([[1.0]], dtype=complex)
        for g in combo:
            U = np.kron(U, _U_single_qubit(g))
        U_list.append(U)
    return tuple(U_list)

def _U_global(gate: str, M: int) -> np.ndarray:
    """Global Clifford = (U_gate)^{⊗ M}."""
    U1 = _U_single_qubit(gate)
    U = np.array([[1.0]], dtype=complex)
    for _ in range(M):
        U = np.kron(U, U1)
    return U


def _cycle_gate_for_iteration(iter_idx: int) -> str:
    """Deterministic cycle over {I, H, HS} by iteration index."""
    cycle = ["i", "h", "hs"]
    return cycle[iter_idx % len(cycle)]


# def _apply_global_frame_then_noise(
#     rho: DensityMatrix,
#     *,
#     M: int,
#     iter_idx: int,
#     spec: RunSpec,
#     twirling_active: bool,
# ) -> DensityMatrix:
#     """
#     Apply noise ONCE to rho.

#     For dephase_z with twirling enabled:
#       gate = cycle(iter_idx) in {I,H,SH}
#       rho -> U rho U†
#       apply Z-dephasing channel (no internal twirling)
#       rho -> U† rho U

#     For other noises or twirling disabled:
#       directly apply the channel once (no frame rotation).
#     """
#     # Only do global cycling for Z-dephasing (your stated scope).
#     if twirling_active and spec.noise.noise_type == NoiseType.dephase_z:
#         gate = _cycle_gate_for_iteration(iter_idx)
#         U = _U_global(gate, M)
#         Udag = U.conj().T

#         rho_rot = DensityMatrix(U @ rho.data @ Udag)

#         # Apply *raw* Z-dephasing once (no internal twirl)
#         rho_noisy_rot = apply_noise_to_density_matrix(
#             rho_rot,
#             spec.noise,
#             twirling=None,
#             twirl_seed=None,
#         )

#         rho_back = DensityMatrix(Udag @ rho_noisy_rot.data @ U)
#         logger.debug(f"Iter {iter_idx}: global frame gate={gate} applied to all qubits")
#         return rho_back

#     # Default: apply the noise channel once
#     return apply_noise_to_density_matrix(
#         rho,
#         spec.noise,
#         twirling=None,   # Important: iterative mode wants deterministic/controlled behavior
#         twirl_seed=None,
#     )

def _apply_local_deterministic_twirled_noise(
    rho: DensityMatrix,
    *,
    M: int,
    spec: RunSpec,
    twirling_active: bool,
) -> DensityMatrix:
    """
    Apply noise ONCE to rho.

    If twirling_active and noise is dephase_z:
      Implements the *exact local deterministic twirl* over {I,H,SH}^⊗M:
        rho_out = (1/3^M) sum_C C† Z_p( C rho C† ) C
      where Z_p is the raw Z-dephasing channel (no internal twirl).

    Otherwise:
      Applies the raw noise channel once.
    """
    if twirling_active and spec.noise.noise_type == NoiseType.dephase_z:
        U_list = _all_local_unitaries(M)
        acc = np.zeros_like(rho.data, dtype=complex)

        for U in U_list:
            Udag = U.conj().T

            # Rotate into Clifford frame: rho -> C rho C†
            rho_rot = DensityMatrix(U @ rho.data @ Udag)

            # Apply raw Z-dephasing once (no internal twirl)
            rho_noisy_rot = apply_noise_to_density_matrix(
                rho_rot,
                spec.noise,
                twirling=None,
                twirl_seed=None,
            )

            # Rotate back: C† (...) C
            rho_back = DensityMatrix(Udag @ rho_noisy_rot.data @ U)

            acc += rho_back.data

        acc /= float(len(U_list))
        return DensityMatrix(acc)

    # Default: apply the noise channel once
    return apply_noise_to_density_matrix(
        rho,
        spec.noise,
        twirling=None,
        twirl_seed=None,
    )



# -----------------------------
# Iterative Purification Runner (Option B)
# -----------------------------

def run_iterative_purification(spec: RunSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Iterative purification with Option B: twirl-per-iteration via global Clifford cycle.

    Protocol per iteration:
      1) Apply noise ONCE to current_state to get rho_noisy
         - if dephase_z and twirling enabled: rotate frame by global C_t, apply Z-dephase, undo
      2) Create 2^ℓ IDENTICAL copies of rho_noisy
      3) Perform ℓ clean SWAP-purification levels (no intermediate noise)
      4) Output becomes current_state for next iteration
    """
    spec.validate()
    out_dir: Path = spec.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if spec.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if spec.noise.mode != NoiseMode.iid_p:
        raise ValueError("Iterative purification only supports iid_p noise mode")

    purification_level = int(spec.purification_level)  # ℓ
    if purification_level < 0:
        raise ValueError("purification_level must be >= 0")
    num_copies_needed = 2 ** purification_level  # copies needed per iteration for ℓ levels

    # number of iterations you chose previously; keep consistent with your current convention
    num_iterations = int(np.log2(spec.N)) if spec.N > 1 else 1

    logger.info(f"Starting ITERATIVE purification (Option B): {spec.synthesize_run_id()}")
    logger.info(f"  M={spec.target.M}, iterations={num_iterations}, ℓ={purification_level}, copies/iter={num_copies_needed}")
    logger.info(f"  noise={spec.noise.noise_type.value}, p={spec.noise.p}")
    logger.info("  Protocol: noise ONCE per iteration; then ℓ clean SWAP rounds on identical inputs")

    # Twirling here means "global Clifford cycling per iteration" for Z-dephasing
    twirling_active = spec._should_apply_twirling()
    if twirling_active and spec.noise.noise_type == NoiseType.dephase_z:
        logger.info("  Global Clifford cycling ENABLED for dephase_z: C_t in {I, H, SH} (deterministic by iteration)")
    elif twirling_active:
        logger.warning("  Twirling enabled by config, but Option B runner only applies global cycling for dephase_z.")
        logger.info("  Other noise types will run without twirling in iterative mode.")
    else:
        logger.info("  Twirling disabled in iterative mode")

    # Backend (only used if you later want circuit-generated copies; iterative uses density ops)
    _ = _make_backend(spec.backend_method)

    # Target state
    prep, psi = build_target(spec.target)
    M = spec.target.M
    logger.info(f"Target state prepared: M={M}, dim={psi.dim}")

    # Start from perfect target state
    current_state = DensityMatrix(psi)

    # Baseline metrics
    # In ITERATIVE mode, "init" should mean: perfect state after ONE noise application
    # (before any purification), otherwise fidelity_init=1 and eps_init=0 by definition.
    F_perfect = _fidelity_to_pure(current_state, psi)
    eps_perfect = _trace_distance_to_pure(current_state, psi)
    pur_perfect = _purity(current_state)

    # rho_init_noisy = _apply_global_frame_then_noise(
    #     current_state,
    #     M=M,
    #     iter_idx=0,
    #     spec=spec,
    #     twirling_active=twirling_active,
    # )
    rho_init_noisy = _apply_local_deterministic_twirled_noise(
        current_state,
        M=M,
        spec=spec,
        twirling_active=twirling_active,
    )

    F_init = _fidelity_to_pure(rho_init_noisy, psi)
    eps_init = _trace_distance_to_pure(rho_init_noisy, psi)
    pur_init = _purity(rho_init_noisy)

    logger.info(
        f"Initial perfect state: F={F_perfect:.6f}, ε_L={eps_perfect:.6f}, purity={pur_perfect:.6f}"
    )
    logger.info(
        f"Initial noisy baseline (one noise application): F={F_init:.6f}, ε_L={eps_init:.6f}, purity={pur_init:.6f}"
    )

    r_init = None
    if M == 1:
        r_init = _bloch_vector_magnitude(rho_init_noisy)
        logger.info(f"  Initial noisy Bloch vector magnitude: |r⃗|={r_init:.6f}")

    steps_rows: List[Dict] = []

    for iter_idx in range(num_iterations):
        logger.info(f"=== Iteration {iter_idx + 1}/{num_iterations} ===")

        # 1) Apply noise ONCE per iteration (Option B global cycling for dephase_z)
        # rho_noisy = _apply_global_frame_then_noise(
        #     current_state,
        #     M=M,
        #     iter_idx=iter_idx,
        #     spec=spec,
        #     twirling_active=twirling_active,
        # )
        
        rho_noisy = _apply_local_deterministic_twirled_noise(
                    current_state,
                    M=M,
                    spec=spec,
                    twirling_active=twirling_active,
                )

        F_before = _fidelity_to_pure(current_state, psi)
        F_after_noise = _fidelity_to_pure(rho_noisy, psi)
        logger.info(f"  Before noise: F={F_before:.6f}")
        logger.info(f"  After noise (single application): F={F_after_noise:.6f}")

        # 2) Clone identical copies (critical for SWAP theory)
        noisy_copies: List[DensityMatrix] = [
            DensityMatrix(rho_noisy.data.copy()) for _ in range(num_copies_needed)
        ]

        # 3) Run clean SWAP streaming merges on these 2^ℓ inputs
        if purification_level == 0:
            iteration_result = noisy_copies[0]
            total_success_prob = 1.0
            merge_count = 0
        else:
            slots: Dict[int, DensityMatrix] = {}
            total_success_prob = 1.0
            merge_count = 0

            for noisy_copy in noisy_copies:
                level = 0
                carry_dm = noisy_copy

                while True:
                    if level not in slots:
                        slots[level] = carry_dm
                        break

                    left = slots.pop(level)
                    merge_count += 1

                    purified_state, meta = purify_two_from_density(left, carry_dm, spec.aa)
                    p_succ = float(meta.get("P_success", 0.0))
                    total_success_prob *= p_succ

                    carry_dm = purified_state
                    level += 1

            # After 2^ℓ copies, we expect exactly one state at level ℓ
            if set(slots.keys()) != {purification_level}:
                raise ValueError(
                    f"Unexpected slot keys after merging 2^ℓ copies: got {sorted(slots.keys())}, "
                    f"expected only level {purification_level}."
                )
            iteration_result = slots[purification_level]

            expected_merges = num_copies_needed - 1
            if merge_count != expected_merges:
                logger.warning(f"Expected {expected_merges} merges but performed {merge_count}")

        # 4) Update current state
        current_state = iteration_result

        # Metrics for this iteration
        fid = _fidelity_to_pure(current_state, psi)
        eps = _trace_distance_to_pure(current_state, psi)
        pur = _purity(current_state)

        row = {
            "run_id": spec.synthesize_run_id(),
            "merge_num": iter_idx,
            "M": M,
            "depth": (iter_idx + 1) * purification_level,     # logical depth counter you were using
            "copies_used": (iter_idx + 1) * num_copies_needed,
            "N_so_far": (iter_idx + 1) * num_copies_needed,
            "noise": spec.noise.noise_type.value,
            "mode": spec.noise.mode.value,
            "p": spec.noise.p,
            "p_channel": spec.noise.kraus_p(),
            "P_success": total_success_prob,
            "grover_iters": 0,
            "twirling_applied": bool(twirling_active and spec.noise.noise_type == NoiseType.dephase_z),
            "fidelity": fid,
            "eps_L": eps,
            "purity": pur,
            "bloch_r": _bloch_vector_magnitude(current_state) if M == 1 else None,
            "iteration": iter_idx + 1,
            "purification_level": purification_level,
            "fidelity_before_noise": F_before,
            "fidelity_after_noise": F_after_noise,
            "eps_L_before_noise": _trace_distance_to_pure(current_state, psi),  # or compute once above
            "eps_L_after_noise": _trace_distance_to_pure(rho_noisy, psi),
            "purity_before_noise": _purity(current_state),
            "purity_after_noise": _purity(rho_noisy),
        }

        logger.info(
            f"  Iteration {iter_idx + 1} complete: F={fid:.6f}, ε_L={eps:.6f}, "
            f"P_succ_total={total_success_prob:.4f}"
        )
        steps_rows.append(row)

    # Final metrics
    F_final = _fidelity_to_pure(current_state, psi)
    eps_final = _trace_distance_to_pure(current_state, psi)
    pur_final = _purity(current_state)
    reduction = eps_final / eps_init if eps_init > 0 else np.nan

    logger.info(
        f"Iterative purification complete: F={F_final:.6f}, ε_L={eps_final:.6f}, reduction={reduction:.6f}"
    )
    if M == 1:
        r_final = _bloch_vector_magnitude(current_state)
        logger.info(f"  Final Bloch vector magnitude: |r⃗|={r_final:.6f}")

    finals_row = {
        "run_id": spec.synthesize_run_id(),
        "M": M,
        "N": spec.N,
        "noise": spec.noise.noise_type.value,
        "mode": spec.noise.mode.value,
        "p": spec.noise.p,
        "p_channel": spec.noise.kraus_p(),
        "fidelity_init": F_init,
        "fidelity_final": F_final,
        "eps_L_init": eps_init,
        "eps_L_final": eps_final,
        "purity_init": pur_init,
        "purity_final": pur_final,
        "error_reduction_ratio": reduction,
        "max_depth": num_iterations * purification_level,
        "twirling_enabled": bool(twirling_active and spec.noise.noise_type == NoiseType.dephase_z),
        "iterations": num_iterations,
        "purification_level": purification_level,
    }
    if M == 1:
        # finals_row["bloch_r_init"] = _bloch_vector_magnitude(DensityMatrix(psi))
        finals_row["bloch_r_init"] = r_init
        finals_row["bloch_r_final"] = _bloch_vector_magnitude(current_state)

    return pd.DataFrame(steps_rows), pd.DataFrame([finals_row])


# -----------------------------
# Regular Streaming Runner (unchanged behavior; warns about twirling)
# -----------------------------

def run_streaming(spec: RunSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Dispatch to iterative or regular streaming mode."""
    if spec.iterative_noise:
        logger.info("Using ITERATIVE noise mode (Option B)")
        return run_iterative_purification(spec)
    logger.info("Using REGULAR streaming mode")
    return run_regular_streaming(spec)


def run_regular_streaming(spec: RunSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Original streaming purification.

    WARNING about twirling:
      If twirling is enabled and you generate different twirls per incoming copy,
      merges will generally not be on identical inputs. Kept as-is for now since
      your request was specifically Option B for iterative mode.
    """
    spec.validate()
    out_dir: Path = spec.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if spec.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Starting streaming run: {spec.synthesize_run_id()}")
    logger.info(
        f"  M={spec.target.M}, N={spec.N}, noise={spec.noise.noise_type.value}, "
        f"mode={spec.noise.mode.value}, p={spec.noise.p}"
    )

    twirling_active = spec._should_apply_twirling()
    if twirling_active:
        logger.warning(
            "Clifford twirling is enabled in REGULAR streaming. Unless you enforce identical twirl "
            "for the two inputs at each merge, SWAP-test identical-input theory is violated."
        )
        logger.info(f"  Twirling enabled (mode={spec.twirling.mode})")
    else:
        logger.info("  Twirling disabled")

    backend = _make_backend(spec.backend_method)

    prep, psi = build_target(spec.target)
    M = spec.target.M
    logger.info(f"Target state prepared: M={M}, dim={psi.dim}")

    iid_cached_dm: Optional[DensityMatrix] = None
    if spec.noise.mode == NoiseMode.iid_p and not twirling_active:
        logger.info("Building cached iid_p noisy copy (no twirling)")
        qc_noisy, _ = build_noisy_copy(prep, spec.noise, seed=spec.target.seed)
        iid_cached_dm = _density_from_circuit(qc_noisy, backend)
    else:
        logger.info("Cannot cache noisy copies (either exact_k mode or twirling enabled)")

    if iid_cached_dm is not None:
        rho_init = iid_cached_dm
    else:
        logger.info("Building sample noisy copy for baseline metrics")
        if twirling_active and spec.noise.mode == NoiseMode.iid_p:
            qc_copy, _ = build_noisy_copy(prep, spec.noise, twirling=spec.twirling, twirl_seed=spec.target.seed)
        else:
            qc_copy, _ = build_noisy_copy(prep, spec.noise, seed=spec.target.seed)
        rho_init = _density_from_circuit(qc_copy, backend)

    F_init = _fidelity_to_pure(rho_init, psi)
    eps_init = _trace_distance_to_pure(rho_init, psi)
    pur_init = _purity(rho_init)

    logger.info(f"Initial state metrics: F={F_init:.6f}, ε_L={eps_init:.6f}, purity={pur_init:.6f}")
    if M == 1:
        r_init = _bloch_vector_magnitude(rho_init)
        logger.info(f"  Initial Bloch vector magnitude: |r⃗|={r_init:.6f}")

    slots: Dict[int, DensityMatrix] = {}
    counts: Dict[int, int] = {}
    steps_rows: List[Dict] = []
    merge_counter = 0
    level0_patterns: Dict[int, ErrorPattern] = {}

    def _log_step(depth: int, rho_out: DensityMatrix, meta: Dict, inputs_used: int, carry_count: int) -> None:
        nonlocal merge_counter
        merge_counter += 1

        fid = _fidelity_to_pure(rho_out, psi)
        eps = _trace_distance_to_pure(rho_out, psi)
        pur = _purity(rho_out)

        n_so_far = sum(counts.values()) + carry_count if counts else carry_count

        row = {
            "run_id": spec.synthesize_run_id(),
            "merge_num": merge_counter,
            "M": M,
            "depth": depth,
            "copies_used": inputs_used,
            "N_so_far": n_so_far,
            "noise": spec.noise.noise_type.value,
            "mode": spec.noise.mode.value,
            "p": spec.noise.p,
            "p_channel": spec.noise.kraus_p(),
            "P_success": float(meta.get("P_success", 0.0)),
            "grover_iters": int(meta.get("grover_iters", 0)),
            "twirling_applied": twirling_active,
            "fidelity": fid,
            "eps_L": eps,
            "purity": pur,
            "bloch_r": _bloch_vector_magnitude(rho_out) if M == 1 else None,
        }
        logger.info(
            f"Merge {merge_counter} (depth {depth}): F={fid:.6f}, ε_L={eps:.6f}, "
            f"P_succ={row['P_success']:.4f}, twirl={twirling_active}"
        )
        steps_rows.append(row)

    def _get_new_copy_dm(i: int) -> DensityMatrix:
        if iid_cached_dm is not None:
            return iid_cached_dm

        if spec.noise.mode == NoiseMode.iid_p:
            if twirling_active:
                twirl_seed = (spec.target.seed or 0) + i * 1000
                qc_copy, _ = build_noisy_copy(prep, spec.noise, twirling=spec.twirling, twirl_seed=twirl_seed)
            else:
                qc_copy, _ = build_noisy_copy(prep, spec.noise, seed=(spec.target.seed or 0) + i)
            return _density_from_circuit(qc_copy, backend)

        qc_copy, pattern = build_noisy_copy(prep, spec.noise, seed=(spec.target.seed or 0) + i)
        if pattern is not None:
            level0_patterns[i] = pattern
        return _density_from_circuit(qc_copy, backend)

    logger.info(f"Processing {spec.N} incoming noisy copies...")
    for i in range(spec.N):
        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"  Processing copy {i+1}/{spec.N}")

        rho_new = _get_new_copy_dm(i)
        level = 0
        carry_dm = rho_new
        carry_count = 1

        while True:
            if level not in slots:
                slots[level] = carry_dm
                counts[level] = carry_count
                break

            left = slots.pop(level)
            left_count = counts.pop(level)

            # exact_k identical-pair fix at level 0
            if spec.noise.mode == NoiseMode.exact_k and level == 0:
                shared_seed = (spec.target.seed or 0) + i * 1000 + level
                shared_pattern = sample_error_pattern(
                    M=M,
                    noise_type=spec.noise.noise_type,
                    k=spec.noise.exact_k,
                    seed=shared_seed,
                )
                qc_left, _ = build_noisy_copy(prep, spec.noise, shared_pattern=shared_pattern)
                qc_right, _ = build_noisy_copy(prep, spec.noise, shared_pattern=shared_pattern)
                left = _density_from_circuit(qc_left, backend)
                carry_dm = _density_from_circuit(qc_right, backend)

            rho_out, meta = purify_two_from_density(left, carry_dm, spec.aa)

            _log_step(
                depth=level + 1,
                rho_out=rho_out,
                meta=meta,
                inputs_used=left_count + carry_count,
                carry_count=carry_count,
            )

            carry_dm = rho_out
            carry_count = left_count + carry_count
            level += 1

    if not counts:
        raise ValueError("No data was processed; N must be >= 1")

    max_level = max(counts.keys())
    rho_final = slots[max_level]

    F_final = _fidelity_to_pure(rho_final, psi)
    eps_final = _trace_distance_to_pure(rho_final, psi)
    pur_final = _purity(rho_final)
    reduction = eps_final / eps_init if eps_init > 0 else np.nan

    logger.info(
        f"Final state (max depth {max_level}): F={F_final:.6f}, ε_L={eps_final:.6f}, reduction={reduction:.6f}"
    )
    if M == 1:
        r_final = _bloch_vector_magnitude(rho_final)
        logger.info(f"  Final Bloch vector magnitude: |r⃗|={r_final:.6f}")

    finals_row = {
        "run_id": spec.synthesize_run_id(),
        "M": M,
        "N": spec.N,
        "noise": spec.noise.noise_type.value,
        "mode": spec.noise.mode.value,
        "p": spec.noise.p,
        "p_channel": spec.noise.kraus_p(),
        "fidelity_init": F_init,
        "fidelity_final": F_final,
        "eps_L_init": eps_init,
        "eps_L_final": eps_final,
        "purity_init": pur_init,
        "purity_final": pur_final,
        "error_reduction_ratio": reduction,
        "max_depth": int(np.log2(counts[max_level])),
        "twirling_enabled": twirling_active,
    }
    if M == 1:
        finals_row["bloch_r_init"] = _bloch_vector_magnitude(rho_init)
        finals_row["bloch_r_final"] = _bloch_vector_magnitude(rho_final)

    return pd.DataFrame(steps_rows), pd.DataFrame([finals_row])


def run_and_save(spec: RunSpec) -> Tuple[Path, Path]:
    """Run a single spec and append results to CSVs under spec.out_dir."""
    steps_df, finals_df = run_streaming(spec)
    out_dir = spec.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # steps_path = out_dir / "steps_circuit_depolarizing.csv"
    # finals_path = out_dir / "finals_circuit_depolarizing.csv"
    suffix = spec.noise.noise_type.value
    steps_path = out_dir / f"steps_circuit_{suffix}_theta_phi_no_twirl.csv"
    finals_path = out_dir / f"finals_circuit_{suffix}_theta_phi_no_twirl.csv"

    if steps_path.exists():
        prev = pd.read_csv(steps_path)
        steps_df = pd.concat([prev, steps_df], ignore_index=True)
    if finals_path.exists():
        prev = pd.read_csv(finals_path)
        finals_df = pd.concat([prev, finals_df], ignore_index=True)

    steps_df.to_csv(steps_path, index=False)
    finals_df.to_csv(finals_path, index=False)

    logger.info(f"Results saved to {steps_path.parent}")
    return steps_path, finals_path


__all__ = ["run_streaming", "run_and_save", "run_iterative_purification", "run_regular_streaming"]
