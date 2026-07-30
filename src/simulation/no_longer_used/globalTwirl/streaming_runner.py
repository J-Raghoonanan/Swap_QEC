"""
GlobalTwirl streaming runner.

MINIMAL CHANGE POLICY:
- Do NOT change CSV headers or stored columns relative to the original code.
- Do NOT change any behavior in REGULAR streaming mode (unless strictly needed).
- Only introduce the GlobalTwirl behavior in ITERATIVE mode:
    * Apply noise ONCE per iteration.
    * If dephase_z and twirling enabled: apply ONE GLOBAL Clifford gate to all qubits
      (same gate for every qubit) before noise, then undo after noise.
    * The global gate cycles deterministically over {I, H, HS} by iteration index.

CRITICAL SEMANTICS:
- iteration column is t
- purification_level column is ℓ
- depth must track t in iterative mode (NOT t·ℓ), otherwise max depth becomes 30 for (t=10,ℓ=3).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from qiskit import transpile
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit.quantum_info.operators import Pauli

try:  # qiskit-aer >= 0.12
    from qiskit_aer import AerSimulator
except Exception:  # pragma: no cover
    from qiskit.providers.aer import AerSimulator  # type: ignore

from ..moreNoise.configs import NoiseMode, NoiseType, RunSpec
from ..moreNoise.state_factory import build_target
from ..moreNoise.noise_engine import build_noisy_copy, sample_error_pattern, ErrorPattern, apply_noise_to_density_matrix
from ..moreNoise.amplified_swap import purify_two_from_density

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
    qc2 = qc.copy()
    qc2.save_density_matrix()
    t = transpile(qc2, backend)
    res = backend.run(t, shots=1024).result()
    data0 = res.data(0)

    rho = data0.get("density_matrix")
    if rho is None:
        rho = data0.get("statevector")

    dm = DensityMatrix(rho)
    return dm


# -----------------------------
# Metric helpers
# -----------------------------

def _fidelity_to_pure(rho: DensityMatrix, psi: Statevector) -> float:
    v = psi.data.reshape((-1, 1))
    return float(np.real(np.conj(v).T @ (rho.data @ v)))


def _trace_distance_to_pure(rho: DensityMatrix, psi: Statevector) -> float:
    proj = np.outer(psi.data, np.conj(psi.data))
    diff = rho.data - proj
    evals = np.linalg.eigvalsh((diff + diff.conj().T) / 2.0)
    return 0.5 * float(np.sum(np.abs(evals)))


def _purity(rho: DensityMatrix) -> float:
    return float(np.real(np.trace(rho.data @ rho.data)))


def _bloch_vector_magnitude(rho: DensityMatrix) -> Optional[float]:
    if rho.dim != 2:
        return None
    rx = np.real(np.trace(rho.data @ Pauli("X").to_matrix()))
    ry = np.real(np.trace(rho.data @ Pauli("Y").to_matrix()))
    rz = np.real(np.trace(rho.data @ Pauli("Z").to_matrix()))
    return float(np.sqrt(rx**2 + ry**2 + rz**2))


# -----------------------------
# GlobalTwirl frame utilities
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


def _apply_global_frame_then_noise(
    rho: DensityMatrix,
    *,
    M: int,
    iter_idx: int,
    spec: RunSpec,
    twirling_active: bool,
) -> DensityMatrix:
    """
    Apply noise ONCE to rho.

    If twirling_active and noise is dephase_z:
      gate = cycle(iter_idx) in {I,H,HS}
      rho -> U rho U†
      apply RAW Z-dephasing once (no internal twirl)
      rho -> U† rho U

    Otherwise:
      apply RAW noise once (no internal twirl).
    """
    if twirling_active and spec.noise.noise_type == NoiseType.dephase_z:
        gate = _cycle_gate_for_iteration(iter_idx)
        U = _U_global(gate, M)
        Udag = U.conj().T

        rho_rot = DensityMatrix(U @ rho.data @ Udag)

        # Apply *raw* channel once: IMPORTANT twirling=None in iterative mode
        rho_noisy_rot = apply_noise_to_density_matrix(
            rho_rot,
            spec.noise,
            twirling=None,
            twirl_seed=None,
        )

        rho_back = DensityMatrix(Udag @ rho_noisy_rot.data @ U)
        return rho_back

    return apply_noise_to_density_matrix(
        rho,
        spec.noise,
        twirling=None,
        twirl_seed=None,
    )


# -----------------------------
# Iterative Purification Runner
# -----------------------------

def run_iterative_purification(spec: RunSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Iterative mode:
      - Start from perfect |ψ⟩.
      - For each cycle t (iteration):
          (a) Apply noise ONCE to current state.
          (b) Create 2^ℓ IDENTICAL copies of that noisy density matrix.
          (c) Run ℓ clean SWAP-purification levels (no intermediate noise).
          (d) Output becomes the new current state.

    Semantics:
      iteration = t
      purification_level = ℓ
      depth = t  (DO NOT set depth = t·ℓ)
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
    num_copies_needed = 2 ** purification_level

    # Keep your original convention: t_max = log2(N)
    num_iterations = int(np.log2(spec.N)) if spec.N > 1 else 1

    twirling_active = spec._should_apply_twirling()

    # Target state
    prep, psi = build_target(spec.target)
    M = spec.target.M

    # Start from perfect
    current_state = DensityMatrix(psi)

    # Baseline "init" = after one noise application (iteration index 0)
    rho_init_noisy = _apply_global_frame_then_noise(
        current_state,
        M=M,
        iter_idx=0,
        spec=spec,
        twirling_active=twirling_active,
    )

    F_init = _fidelity_to_pure(rho_init_noisy, psi)
    eps_init = _trace_distance_to_pure(rho_init_noisy, psi)
    pur_init = _purity(rho_init_noisy)

    r_init = _bloch_vector_magnitude(rho_init_noisy) if M == 1 else None

    steps_rows: List[Dict] = []

    for iter_idx in range(num_iterations):
        t = iter_idx + 1  # iteration (cycle) index

        # 1) noise once per cycle
        rho_noisy = _apply_global_frame_then_noise(
            current_state,
            M=M,
            iter_idx=iter_idx,
            spec=spec,
            twirling_active=twirling_active,
        )

        F_before = _fidelity_to_pure(current_state, psi)
        F_after_noise = _fidelity_to_pure(rho_noisy, psi)

        # 2) identical copies
        noisy_copies: List[DensityMatrix] = [
            DensityMatrix(rho_noisy.data.copy()) for _ in range(num_copies_needed)
        ]

        # 3) ℓ clean SWAP levels
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

            if set(slots.keys()) != {purification_level}:
                raise ValueError(
                    f"Unexpected slot keys after merging 2^ℓ copies: got {sorted(slots.keys())}, "
                    f"expected only level {purification_level}."
                )
            iteration_result = slots[purification_level]

            expected_merges = num_copies_needed - 1
            if merge_count != expected_merges:
                logger.warning(f"Expected {expected_merges} merges but performed {merge_count}")

        # 4) update
        current_state = iteration_result

        # metrics
        fid = _fidelity_to_pure(current_state, psi)
        eps = _trace_distance_to_pure(current_state, psi)
        pur = _purity(current_state)

        # NOTE: KEEP SAME HEADERS as your original steps file
        row = {
            "run_id": spec.synthesize_run_id(),
            "merge_num": iter_idx,
            "M": M,
            # CRITICAL: depth tracks t (not t·ℓ)
            "depth": t,
            "copies_used": t * num_copies_needed,
            "N_so_far": t * num_copies_needed,
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
            "iteration": t,  # = t
            "purification_level": purification_level,  # = ℓ
            "fidelity_before_noise": F_before,
            "fidelity_after_noise": F_after_noise,
            "eps_L_before_noise": _trace_distance_to_pure(current_state, psi),
            "eps_L_after_noise": _trace_distance_to_pure(rho_noisy, psi),
            "purity_before_noise": _purity(current_state),
            "purity_after_noise": _purity(rho_noisy),
        }
        steps_rows.append(row)

    # finals
    F_final = _fidelity_to_pure(current_state, psi)
    eps_final = _trace_distance_to_pure(current_state, psi)
    pur_final = _purity(current_state)
    reduction = eps_final / eps_init if eps_init > 0 else np.nan

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
        # CRITICAL: max_depth = max t (not t·ℓ)
        "max_depth": num_iterations,
        "twirling_enabled": bool(twirling_active and spec.noise.noise_type == NoiseType.dephase_z),
        "iterations": num_iterations,
        "purification_level": purification_level,
    }
    if M == 1:
        finals_row["bloch_r_init"] = r_init
        finals_row["bloch_r_final"] = _bloch_vector_magnitude(current_state)

    return pd.DataFrame(steps_rows), pd.DataFrame([finals_row])


# -----------------------------
# Regular Streaming Runner (delegated unchanged)
# -----------------------------

def run_regular_streaming(spec: RunSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Delegate to the original regular-streaming implementation so behavior is unchanged.

    We import and call it rather than copy/paste, to enforce the “don’t change what you don’t need” rule.
    """
    from ..moreNoise.streaming_runner import run_regular_streaming as _orig_run_regular_streaming
    return _orig_run_regular_streaming(spec)


def run_streaming(spec: RunSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if spec.iterative_noise:
        logger.info("Using ITERATIVE noise mode (GlobalTwirl)")
        return run_iterative_purification(spec)
    logger.info("Using REGULAR streaming mode (delegated)")
    return run_regular_streaming(spec)


def run_and_save(spec: RunSpec) -> Tuple[Path, Path]:
    """
    Run a single spec and append results to GlobalTwirl CSVs.

    IMPORTANT:
    - Same columns/headers as original outputs.
    - Only filenames differ (namespace separation).
    """
    steps_df, finals_df = run_streaming(spec)
    out_dir = spec.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = spec.noise.noise_type.value
    steps_path = out_dir / f"steps_globalTwirl_{suffix}.csv"
    finals_path = out_dir / f"finals_globalTwirl_{suffix}.csv"

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
