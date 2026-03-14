"""
Streaming purification runner using rho2.

Implements ITERATIVE mode only (regular streaming is not used):

  ITERATIVE mode:
    - Start from perfect |ψ⟩.
    - For each iteration t = 0, 1, ..., num_iterations-1:
        (a) Apply noise ONCE to the current state using the *exact local
            deterministic twirl* over {I, H, HS}^⊗M (see below).
        (b) Create 2^ℓ IDENTICAL copies of that noisy density matrix.
        (c) Run ℓ clean rho2 purification levels (binary tree, no intermediate noise).
        (d) The purified output becomes the new current state.

TWIRLING (ported exactly from pre-rho2 streaming_runner.py):
  For dephase_z with twirling enabled, we compute the *exact* average:
      rho_out = (1/3^M) Σ_C  C† · Z_p(C ρ C†) · C
  summed over all 3^M combinations C in {I, H, HS}^⊗M.
  This is a deterministic, closed-form twirl — NOT a sampled approximation.
  For depolarizing or twirling disabled, the raw channel is applied once.

rho2 differences from SWAP:
  - State update: ρ → ρ²/Tr(ρ²)   (no SWAP unitary, no ancilla)
  - P_success = 1.0 always (deterministic)
  - C_ℓ = 2^ℓ exactly (no postselection overhead)
  - G_ℓ = 2^ℓ − 1 (total rho2 operations per iteration)

num_iterations = floor(log2(N))  (consistent with pre-rho2 convention where N
is the number of incoming copies; in iterative mode this sets the number of
noise+purification cycles).
"""
from __future__ import annotations

import itertools
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from qiskit import transpile
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit.quantum_info.operators import Pauli

try:
    from qiskit_aer import AerSimulator
except Exception:
    from qiskit.providers.aer import AerSimulator  # type: ignore

from .configs import NoiseMode, NoiseType, RunSpec
from .state_factory import build_target
from .noise_engine import apply_noise_to_density_matrix
from .rho2_purification import purify_two_from_density   # rho2 replace of amplified_swap

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers  (unchanged from pre-rho2 version)
# ─────────────────────────────────────────────────────────────────────────────

def _fidelity_to_pure(rho: DensityMatrix, psi: Statevector) -> float:
    """F = ⟨ψ|ρ|ψ⟩"""
    v = psi.data.reshape((-1, 1))
    return float(np.real(np.conj(v).T @ (rho.data @ v)))


def _trace_distance_to_pure(rho: DensityMatrix, psi: Statevector) -> float:
    """ε_L = (1/2) ‖ρ − |ψ⟩⟨ψ|‖_1"""
    proj = np.outer(psi.data, np.conj(psi.data))
    diff = rho.data - proj
    evals = np.linalg.eigvalsh((diff + diff.conj().T) / 2.0)
    return 0.5 * float(np.sum(np.abs(evals)))


def _purity(rho: DensityMatrix) -> float:
    """Tr(ρ²)"""
    return float(np.real(np.trace(rho.data @ rho.data)))


def _bloch_vector_magnitude(rho: DensityMatrix) -> Optional[float]:
    """For M=1: |r⃗| where ρ = (I + r⃗·σ⃗)/2."""
    if rho.dim != 2:
        return None
    rx = np.real(np.trace(rho.data @ Pauli("X").to_matrix()))
    ry = np.real(np.trace(rho.data @ Pauli("Y").to_matrix()))
    rz = np.real(np.trace(rho.data @ Pauli("Z").to_matrix()))
    return float(np.sqrt(rx**2 + ry**2 + rz**2))


# ─────────────────────────────────────────────────────────────────────────────
# Exact local deterministic Clifford twirl  (ported unchanged from pre-rho2 code)
# ─────────────────────────────────────────────────────────────────────────────

def _U_single_qubit(gate: str) -> np.ndarray:
    """Single-qubit unitary for gate in {'i', 'h', 'hs'}, where 'hs' := H·S."""
    if gate == "i":
        return np.eye(2, dtype=complex)
    if gate == "h":
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    if gate == "hs":
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        S = np.array([[1, 0], [0, 1j]], dtype=complex)
        return H @ S   # HS: maps Z → Y
    raise ValueError(f"Unknown global Clifford gate '{gate}'")


@lru_cache(maxsize=None)
def _all_local_clifford_combos(M: int) -> Tuple[Tuple[str, ...], ...]:
    """All tuples (g₀,…,g_{M-1}) with gⱼ ∈ {'i','h','hs'} — total 3^M tuples."""
    gates = ("i", "h", "hs")
    return tuple(itertools.product(gates, repeat=M))


@lru_cache(maxsize=None)
def _all_local_unitaries(M: int) -> Tuple[np.ndarray, ...]:
    """Precompute U = ⊗_j U(gⱼ) for every combo in {'i','h','hs'}^M."""
    combos = _all_local_clifford_combos(M)
    U_list: List[np.ndarray] = []
    for combo in combos:
        U = np.array([[1.0]], dtype=complex)
        for g in combo:
            U = np.kron(U, _U_single_qubit(g))
        U_list.append(U)
    return tuple(U_list)


def _apply_local_deterministic_twirled_noise(
    rho: DensityMatrix,
    *,
    M: int,
    spec: RunSpec,
    twirling_active: bool,
) -> DensityMatrix:
    """
    Apply noise ONCE to rho using the exact deterministic local twirl.

    For dephase_z with twirling_active=True:
      Computes exactly:
          rho_out = (1/3^M) Σ_{C ∈ {I,H,HS}^⊗M}  C† · Z_p(C ρ C†) · C
      This is the *closed-form* twirl — no sampling, no approximation.
      It converts per-qubit Z-dephasing into effective per-qubit depolarizing.

    For all other cases (depolarizing or twirling disabled):
      Applies the raw channel once without any frame rotation.
    """
    if twirling_active and spec.noise.noise_type == NoiseType.dephase_z:
        U_list = _all_local_unitaries(M)
        acc = np.zeros_like(rho.data, dtype=complex)

        for U in U_list:
            Udag = U.conj().T

            # Rotate into Clifford frame: ρ → C ρ C†
            rho_rot = DensityMatrix(U @ rho.data @ Udag)

            # Apply raw Z-dephasing (twirling=None so no internal twirl)
            rho_noisy_rot = apply_noise_to_density_matrix(
                rho_rot,
                spec.noise,
                twirling=None,
                twirl_seed=None,
            )

            # Rotate back: C† (…) C
            rho_back = DensityMatrix(Udag @ rho_noisy_rot.data @ U)
            acc += rho_back.data

        acc /= float(len(U_list))
        return DensityMatrix(acc)

    # Default: apply the noise channel once, no frame rotation
    return apply_noise_to_density_matrix(
        rho,
        spec.noise,
        twirling=None,
        twirl_seed=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Iterative rho2 purification
# ─────────────────────────────────────────────────────────────────────────────

def run_iterative_purification(spec: RunSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Iterative rho2 purification.

    Protocol per iteration:
      1) Apply noise ONCE to current_state → rho_noisy
         (exact local deterministic twirl for dephase_z if twirling enabled)
      2) Create 2^ℓ IDENTICAL copies of rho_noisy
      3) Perform ℓ clean rho2 levels (binary tree, no intermediate noise)
         rho2 merge: (rho_L, rho_R) → rho_L² / Tr(rho_L²)   [P_success = 1.0]
      4) Output becomes current_state for next iteration

    num_iterations = floor(log2(N))   [consistent with pre-rho2 convention]
    """
    spec.validate()
    out_dir: Path = spec.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if spec.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if spec.noise.mode != NoiseMode.iid_p:
        raise ValueError("Iterative purification only supports iid_p noise mode")

    purification_level = int(spec.purification_level)   # ℓ
    if purification_level < 0:
        raise ValueError("purification_level must be >= 0")
    num_copies_needed = 2 ** purification_level

    # Convention from pre-rho2: N encodes number of copies → iterations = log2(N)
    num_iterations = int(np.log2(spec.N)) if spec.N > 1 else 1

    twirling_active = spec._should_apply_twirling()

    logger.info("=" * 70)
    logger.info("ITERATIVE RHO2 MODE")
    logger.info(f"  M={spec.target.M}, iterations={num_iterations}, ℓ={purification_level}, copies/iter={num_copies_needed}")
    logger.info(f"  noise={spec.noise.noise_type.value}, p={spec.noise.p}")
    logger.info(f"  rho2 is deterministic (P_success = 1.0 always)")
    logger.info(f"  Protocol: noise ONCE per iteration; then ℓ clean rho2 rounds on identical inputs")

    if twirling_active and spec.noise.noise_type == NoiseType.dephase_z:
        logger.info("  Exact local deterministic twirl ENABLED for dephase_z: average over {I,H,HS}^⊗M")
    elif twirling_active:
        logger.warning("  Twirling enabled but only implemented for dephase_z; running without twirl for this noise type.")
    else:
        logger.info("  Twirling DISABLED")
    logger.info("=" * 70)

    # Build target state (circuit not used in iterative DM mode, but needed for psi reference)
    prep, psi = build_target(spec.target)
    M = spec.target.M
    logger.info(f"Target state prepared: M={M}, dim={psi.dim}")

    # Start from perfect target state
    current_state = DensityMatrix(psi)

    # Baseline: one noise application from perfect state (for fidelity_init in finals row)
    rho_init_noisy = _apply_local_deterministic_twirled_noise(
        current_state,
        M=M,
        spec=spec,
        twirling_active=twirling_active,
    )
    F_init     = _fidelity_to_pure(rho_init_noisy, psi)
    eps_init   = _trace_distance_to_pure(rho_init_noisy, psi)
    pur_init   = _purity(rho_init_noisy)

    logger.info(f"Baseline (perfect state + one noise app): F={F_init:.6f}, ε_L={eps_init:.6f}, purity={pur_init:.6f}")
    if M == 1:
        r_init = _bloch_vector_magnitude(rho_init_noisy)
        logger.info(f"  Baseline Bloch vector magnitude: |r⃗|={r_init:.6f}")
    else:
        r_init = None

    steps_rows: List[Dict] = []

    for iter_idx in range(num_iterations):
        logger.info(f"=== Iteration {iter_idx + 1}/{num_iterations} ===")

        # ── Step 1: Apply noise once ──────────────────────────────────────────
        F_before = _fidelity_to_pure(current_state, psi)

        rho_noisy = _apply_local_deterministic_twirled_noise(
            current_state,
            M=M,
            spec=spec,
            twirling_active=twirling_active,
        )

        F_after_noise = _fidelity_to_pure(rho_noisy, psi)
        logger.info(f"  Before noise: F={F_before:.6f}")
        logger.info(f"  After  noise: F={F_after_noise:.6f}")

        # ── Step 2: Clone identical copies ────────────────────────────────────
        noisy_copies: List[DensityMatrix] = [
            DensityMatrix(rho_noisy.data.copy()) for _ in range(num_copies_needed)
        ]

        # ── Step 3: ℓ clean rho2 levels via binary tree ────────────────────────
        if purification_level == 0:
            iteration_result  = noisy_copies[0]
            total_success_prob = 1.0
            merge_count       = 0
        else:
            slots: Dict[int, DensityMatrix] = {}
            total_success_prob = 1.0
            merge_count       = 0

            for noisy_copy in noisy_copies:
                level    = 0
                carry_dm = noisy_copy

                while True:
                    if level not in slots:
                        slots[level] = carry_dm
                        break

                    left = slots.pop(level)
                    merge_count += 1

                    # rho2 merge: ρ → ρ²/Tr(ρ²), P_success = 1.0
                    purified_state, meta = purify_two_from_density(left, carry_dm, spec.aa)
                    p_succ = float(meta.get("P_success", 1.0))
                    total_success_prob *= p_succ   # stays 1.0 for rho2

                    carry_dm = purified_state
                    level   += 1

            if set(slots.keys()) != {purification_level}:
                raise ValueError(
                    f"Unexpected slot keys after merging 2^ℓ copies: got {sorted(slots.keys())}, "
                    f"expected only level {purification_level}."
                )
            iteration_result = slots[purification_level]

            expected_merges = num_copies_needed - 1
            if merge_count != expected_merges:
                logger.warning(f"Expected {expected_merges} merges but performed {merge_count}")

        # ── Step 4: Update current state ──────────────────────────────────────
        current_state = iteration_result

        fid = _fidelity_to_pure(current_state, psi)
        eps = _trace_distance_to_pure(current_state, psi)
        pur = _purity(current_state)

        row = {
            "run_id":               spec.synthesize_run_id(),
            "iteration":            iter_idx + 1,
            "merge_num":            iter_idx,
            "M":                    M,
            "depth":                (iter_idx + 1) * purification_level,
            "copies_used":          (iter_idx + 1) * num_copies_needed,
            "N_so_far":             (iter_idx + 1) * num_copies_needed,
            "noise":                spec.noise.noise_type.value,
            "mode":                 spec.noise.mode.value,
            "p":                    spec.noise.p,
            "p_channel":            spec.noise.kraus_p(),
            "P_success":            total_success_prob,   # always 1.0 for rho2
            "grover_iters":         0,
            "twirling_applied":     bool(twirling_active and spec.noise.noise_type == NoiseType.dephase_z),
            "purification_level":   purification_level,
            "fidelity":             fid,
            "eps_L":                eps,
            "purity":               pur,
            "bloch_r":              _bloch_vector_magnitude(current_state) if M == 1 else None,
            "fidelity_before_noise": F_before,
            "fidelity_after_noise":  F_after_noise,
            "eps_L_before_noise":   _trace_distance_to_pure(current_state, psi),
            "eps_L_after_noise":    _trace_distance_to_pure(rho_noisy, psi),
            "purity_before_noise":  _purity(current_state),
            "purity_after_noise":   _purity(rho_noisy),
        }
        logger.info(
            f"  Iteration {iter_idx + 1} complete: F={fid:.6f}, ε_L={eps:.6f}, "
            f"P_succ_total={total_success_prob:.4f} (rho2=deterministic)"
        )
        steps_rows.append(row)

    # ── Final metrics ─────────────────────────────────────────────────────────
    F_final   = _fidelity_to_pure(current_state, psi)
    eps_final = _trace_distance_to_pure(current_state, psi)
    pur_final = _purity(current_state)
    reduction = eps_final / eps_init if eps_init > 0 else np.nan

    logger.info(f"rho2 iterative complete: F={F_final:.6f}, ε_L={eps_final:.6f}, reduction={reduction:.6f}")
    if M == 1:
        logger.info(f"  Final Bloch vector magnitude: |r⃗|={_bloch_vector_magnitude(current_state):.6f}")

    finals_row = {
        "run_id":              spec.synthesize_run_id(),
        "M":                   M,
        "N":                   spec.N,
        "noise":               spec.noise.noise_type.value,
        "mode":                spec.noise.mode.value,
        "p":                   spec.noise.p,
        "p_channel":           spec.noise.kraus_p(),
        "fidelity_init":       F_init,
        "fidelity_final":      F_final,
        "eps_L_init":          eps_init,
        "eps_L_final":         eps_final,
        "purity_init":         pur_init,
        "purity_final":        pur_final,
        "error_reduction_ratio": reduction,
        "max_depth":           num_iterations * purification_level,
        "iterations":          num_iterations,
        "purification_level":  purification_level,
        "twirling_enabled":    bool(twirling_active and spec.noise.noise_type == NoiseType.dephase_z),
    }
    if M == 1:
        finals_row["bloch_r_init"]  = r_init
        finals_row["bloch_r_final"] = _bloch_vector_magnitude(current_state)

    return pd.DataFrame(steps_rows), pd.DataFrame([finals_row])


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher and I/O
# ─────────────────────────────────────────────────────────────────────────────

def run_streaming(spec: RunSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Always runs iterative mode (regular streaming not used for rho2)."""
    if not spec.iterative_noise:
        logger.warning(
            "iterative_noise=False but rho2 streaming_runner only supports iterative mode. "
            "Proceeding with iterative purification."
        )
    return run_iterative_purification(spec)


def run_and_save(spec: RunSpec) -> Tuple[Path, Path]:
    """Run a single spec and append results to CSVs under spec.out_dir."""
    steps_df, finals_df = run_streaming(spec)
    out_dir = spec.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = spec.noise.noise_type.value
    steps_path  = out_dir / f"steps_rho2_{suffix}.csv"
    finals_path = out_dir / f"finals_rho2_{suffix}.csv"

    # steps_path  = out_dir / f"steps_rho2_{suffix}_theta_phi_no_twirl.csv"
    # finals_path = out_dir / f"finals_rho2_{suffix}_theta_phi_no_twirl.csv"

    if steps_path.exists():
        try:
            prev = pd.read_csv(steps_path)
            if len(prev) > 0:
                steps_df = pd.concat([prev, steps_df], ignore_index=True)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            logger.warning(f"Error reading {steps_path}: {e}. Overwriting.")

    if finals_path.exists():
        try:
            prev = pd.read_csv(finals_path)
            if len(prev) > 0:
                finals_df = pd.concat([prev, finals_df], ignore_index=True)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            logger.warning(f"Error reading {finals_path}: {e}. Overwriting.")

    steps_df.to_csv(steps_path, index=False)
    finals_df.to_csv(finals_path, index=False)

    logger.info(f"Results saved to {steps_path.parent}")
    return steps_path, finals_path


__all__ = [
    "run_streaming",
    "run_and_save",
    "run_iterative_purification",
]