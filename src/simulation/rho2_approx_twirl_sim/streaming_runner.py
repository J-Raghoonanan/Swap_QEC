"""
Streaming purification runner using rho2.

Implements ITERATIVE mode only (regular streaming is not used for rho2):

  ITERATIVE mode:
    - Start from perfect |ψ⟩.
    - For each iteration t = 0, 1, ..., num_iterations-1:
        (a) Apply noise ONCE to the current state via _apply_twirled_noise,
            which delegates entirely to noise_engine.apply_noise_to_density_matrix.
        (b) Create 2^ℓ IDENTICAL copies of that noisy density matrix.
        (c) Run ℓ clean rho2 levels (binary tree, no intermediate noise).
        (d) The purified output becomes the new current state.

TWIRLING  (delegated to noise_engine):
  For dephase_z/dephase_x with twirling enabled, noise_engine computes:

      rho_out = (1/K) Σ_{k=1}^{K}  C_k†  E( C_k ρ C_k† )  C_k

  where K is determined by spec.twirling.subset_fraction:
    - subset_fraction = 1.0  →  K = 3^M, exact full twirl (no approximation)
    - subset_fraction < 1.0  →  K < 3^M, approximate subset twirl

  The subset is drawn with a fixed seed each call, so the approximate twirl
  is a deterministic, reproducible channel (not re-sampled each iteration).

rho2 differences from SWAP:
  - State update: ρ → ρ²/Tr(ρ²)   (no SWAP unitary, no ancilla)
  - P_success = Tr(ρ²) always (purity of the input at each merge)
  - C_ℓ = 2^ℓ exactly (no postselection overhead)

num_iterations = floor(log2(N))  — consistent with pre-rho2 SWAP convention.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit.quantum_info.operators import Pauli

from .configs import NoiseMode, NoiseType, RunSpec
from .state_factory import build_target
from .noise_engine import apply_noise_to_density_matrix
from .rho2_purification import purify_two_from_density   #rho2: ρ → ρ²/Tr(ρ²)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fidelity_to_pure(rho: DensityMatrix, psi: Statevector) -> float:
    """F = ⟨ψ|ρ|ψ⟩"""
    v = psi.data.reshape((-1, 1))
    return float(np.real(np.conj(v).T @ (rho.data @ v)))


def _trace_distance_to_pure(rho: DensityMatrix, psi: Statevector) -> float:
    """ε_L = (1/2) ‖ρ − |ψ⟩⟨ψ|‖_1"""
    proj  = np.outer(psi.data, np.conj(psi.data))
    diff  = rho.data - proj
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
# Noise application  (thin delegation wrapper — all logic lives in noise_engine)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_twirled_noise(
    rho: DensityMatrix,
    *,
    spec: RunSpec,
    twirling_active: bool,
) -> DensityMatrix:
    """Apply one round of noise to rho, delegating entirely to noise_engine.

    When twirling_active is True and noise is dephase_z/dephase_x,
    noise_engine handles the Clifford-subset averaging:

        rho_out = (1/K) Σ_k  C_k†  E( C_k ρ C_k† )  C_k

    with K controlled by spec.twirling.subset_fraction (1.0 = full exact twirl).

    For depolarizing noise or twirling disabled, the raw channel is applied once.
    """
    return apply_noise_to_density_matrix(
        rho,
        spec.noise,
        twirling=spec.twirling if twirling_active else None,
        twirl_seed=spec.target.seed,  # fixed per run → deterministic subset
    )


# ─────────────────────────────────────────────────────────────────────────────
# Iterative rho2 purification
# ─────────────────────────────────────────────────────────────────────────────

def run_iterative_purification(spec: RunSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Iterative rho2 purification with optional subset Clifford twirling.

    Protocol per iteration:
      1) Record pre-noise metrics on current_state
      2) Apply noise ONCE to current_state → rho_noisy
         (Clifford-subset twirl for dephase_z/x if twirling enabled)
      3) Create 2^ℓ IDENTICAL copies of rho_noisy
      4) Perform ℓ clean rho2 levels (binary tree, no intermediate noise)
         rho2 merge: (rho_L, rho_R) → rho_L² / Tr(rho_L²)
      5) Output becomes current_state for next iteration

    num_iterations = floor(log2(N))   [consistent with pre-rho2 SWAP convention]
    """
    spec.validate()
    spec.out_dir.mkdir(parents=True, exist_ok=True)

    if spec.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if spec.noise.mode != NoiseMode.iid_p:
        raise ValueError("Iterative rho2 purification only supports iid_p noise mode")

    purification_level = int(spec.purification_level)   # ℓ
    if purification_level < 0:
        raise ValueError("purification_level must be >= 0")
    num_copies_needed = 2 ** purification_level          # 2^ℓ copies per iteration

    num_iterations = int(np.log2(spec.N)) if spec.N > 1 else 1

    twirling_active = spec._should_apply_twirling()

    logger.info("=" * 70)
    logger.info("ITERATIVE RHO2 MODE")
    logger.info(
        f"  M={spec.target.M}, iterations={num_iterations}, "
        f"ℓ={purification_level}, copies/iter={num_copies_needed}"
    )
    logger.info(f"  noise={spec.noise.noise_type.value}, p={spec.noise.p}")
    logger.info("  rho2 is deterministic  (P_success = Tr(ρ²) at each merge)")
    logger.info("  Protocol: noise ONCE per iteration; then ℓ clean rho2 rounds on identical copies")

    if twirling_active and spec.noise.noise_type in (NoiseType.dephase_z, NoiseType.dephase_x):
        frac = spec.twirling.subset_fraction
        M_   = spec.target.M
        K    = max(1, int(np.ceil(frac * 3**M_)))
        if frac >= 1.0:
            logger.info(
                f"  Exact Clifford twirl ENABLED: full average over {{I,H,HS}}^⊗{M_} "
                f"({3**M_} combinations)"
            )
        else:
            logger.info(
                f"  Approximate Clifford twirl ENABLED: subset_fraction={frac:.3f}, "
                f"K≈{K}/{3**M_} combinations, mode={spec.twirling.subset_mode!r}"
            )
    elif twirling_active:
        logger.warning(
            "  Twirling enabled but only implemented for dephase_z/dephase_x; "
            "running without twirl for this noise type."
        )
    else:
        logger.info("  Twirling DISABLED")
    logger.info("=" * 70)

    # ── Build target state ────────────────────────────────────────────────────
    _, psi = build_target(spec.target)
    M = spec.target.M
    logger.info(f"Target state prepared: M={M}, dim={psi.dim}")

    # ── Start from perfect state ──────────────────────────────────────────────
    current_state = DensityMatrix(psi)

    # Baseline: one noise application from perfect state (for fidelity_init)
    rho_init_noisy = _apply_twirled_noise(current_state, spec=spec, twirling_active=twirling_active)
    F_init   = _fidelity_to_pure(rho_init_noisy, psi)
    eps_init = _trace_distance_to_pure(rho_init_noisy, psi)
    pur_init = _purity(rho_init_noisy)
    r_init   = _bloch_vector_magnitude(rho_init_noisy) if M == 1 else None

    logger.info(
        f"Baseline (perfect + one noise): "
        f"F={F_init:.6f}, ε_L={eps_init:.6f}, purity={pur_init:.6f}"
    )
    if M == 1 and r_init is not None:
        logger.info(f"  Baseline Bloch vector magnitude: |r⃗|={r_init:.6f}")

    steps_rows: List[Dict] = []

    # ── Main iteration loop ───────────────────────────────────────────────────
    for iter_idx in range(num_iterations):
        logger.info(f"=== Iteration {iter_idx + 1}/{num_iterations} ===")

        # Step 1: Record pre-noise metrics BEFORE modifying current_state.
        #         These must be captured here — they cannot be recomputed later
        #         because current_state will be overwritten in Step 5.
        F_before      = _fidelity_to_pure(current_state, psi)
        eps_L_before  = _trace_distance_to_pure(current_state, psi)
        purity_before = _purity(current_state)

        # Step 2: Apply noise once
        rho_noisy = _apply_twirled_noise(
            current_state,
            spec=spec,
            twirling_active=twirling_active,
        )

        F_after_noise   = _fidelity_to_pure(rho_noisy, psi)
        eps_L_after     = _trace_distance_to_pure(rho_noisy, psi)
        purity_after    = _purity(rho_noisy)

        logger.info(f"  Before noise: F={F_before:.6f}")
        logger.info(f"  After  noise: F={F_after_noise:.6f}")

        # Step 3: Clone 2^ℓ identical copies
        noisy_copies: List[DensityMatrix] = [
            DensityMatrix(rho_noisy.data.copy()) for _ in range(num_copies_needed)
        ]

        # Step 4: ℓ clean rho2 levels via binary tree
        if purification_level == 0:
            iteration_result   = noisy_copies[0]
            total_success_prob = 1.0
            merge_count        = 0
        else:
            slots: Dict[int, DensityMatrix] = {}
            total_success_prob = 1.0
            merge_count        = 0

            for noisy_copy in noisy_copies:
                level    = 0
                carry_dm = noisy_copy

                while True:
                    if level not in slots:
                        slots[level] = carry_dm
                        break

                    left = slots.pop(level)
                    merge_count += 1

                    #rho2 merge: ρ → ρ²/Tr(ρ²)
                    purified_state, meta = purify_two_from_density(left, carry_dm, spec.aa)
                    total_success_prob  *= float(meta.get("P_success", 1.0))

                    carry_dm = purified_state
                    level   += 1

            if set(slots.keys()) != {purification_level}:
                raise ValueError(
                    f"Unexpected slot keys after merging 2^ℓ copies: "
                    f"got {sorted(slots.keys())}, expected only level {purification_level}."
                )
            iteration_result = slots[purification_level]

            expected_merges = num_copies_needed - 1
            if merge_count != expected_merges:
                logger.warning(
                    f"Expected {expected_merges} merges but performed {merge_count}"
                )

        # Step 5: Update current state for next iteration
        current_state = iteration_result

        fid = _fidelity_to_pure(current_state, psi)
        eps = _trace_distance_to_pure(current_state, psi)
        pur = _purity(current_state)

        row = {
            "run_id":              spec.synthesize_run_id(),
            "iteration":           iter_idx + 1,
            "merge_num":           iter_idx,
            "M":                   M,
            "depth":               (iter_idx + 1) * purification_level,
            "copies_used":         (iter_idx + 1) * num_copies_needed,
            "N_so_far":            (iter_idx + 1) * num_copies_needed,
            "noise":               spec.noise.noise_type.value,
            "mode":                spec.noise.mode.value,
            "p":                   spec.noise.p,
            "p_channel":           spec.noise.kraus_p(),
            "P_success":           total_success_prob,
            "grover_iters":        0,
            "twirling_applied":    bool(
                twirling_active
                and spec.noise.noise_type in (NoiseType.dephase_z, NoiseType.dephase_x)
            ),
            "subset_fraction":     spec.twirling.subset_fraction,
            "purification_level":  purification_level,
            "fidelity":            fid,
            "eps_L":               eps,
            "purity":              pur,
            "bloch_r":             _bloch_vector_magnitude(current_state) if M == 1 else None,
            # Pre-noise metrics captured in Step 1 (before current_state was overwritten)
            "fidelity_before_noise":  F_before,
            "fidelity_after_noise":   F_after_noise,
            "eps_L_before_noise":     eps_L_before,
            "eps_L_after_noise":      eps_L_after,
            "purity_before_noise":    purity_before,
            "purity_after_noise":     purity_after,
        }
        logger.info(
            f"  Iteration {iter_idx + 1} complete: F={fid:.6f}, ε_L={eps:.6f}"
        )
        steps_rows.append(row)

    # ── Final metrics ─────────────────────────────────────────────────────────
    F_final   = _fidelity_to_pure(current_state, psi)
    eps_final = _trace_distance_to_pure(current_state, psi)
    pur_final = _purity(current_state)
    reduction = eps_final / eps_init if eps_init > 0 else np.nan

    logger.info(
        f"rho2 iterative complete: F={F_final:.6f}, ε_L={eps_final:.6f}, "
        f"reduction={reduction:.6f}"
    )
    if M == 1:
        logger.info(
            f"  Final Bloch vector magnitude: "
            f"|r⃗|={_bloch_vector_magnitude(current_state):.6f}"
        )

    finals_row = {
        "run_id":                spec.synthesize_run_id(),
        "M":                     M,
        "N":                     spec.N,
        "noise":                 spec.noise.noise_type.value,
        "mode":                  spec.noise.mode.value,
        "p":                     spec.noise.p,
        "p_channel":             spec.noise.kraus_p(),
        "fidelity_init":         F_init,
        "fidelity_final":        F_final,
        "eps_L_init":            eps_init,
        "eps_L_final":           eps_final,
        "purity_init":           pur_init,
        "purity_final":          pur_final,
        "error_reduction_ratio": reduction,
        "max_depth":             num_iterations * purification_level,
        "iterations":            num_iterations,
        "purification_level":    purification_level,
        "subset_fraction":       spec.twirling.subset_fraction,
        "twirling_enabled":      bool(
            twirling_active
            and spec.noise.noise_type in (NoiseType.dephase_z, NoiseType.dephase_x)
        ),
    }
    if M == 1:
        finals_row["bloch_r_init"]  = r_init
        finals_row["bloch_r_final"] = _bloch_vector_magnitude(current_state)

    return pd.DataFrame(steps_rows), pd.DataFrame([finals_row])


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher and I/O
# ─────────────────────────────────────────────────────────────────────────────

def run_streaming(spec: RunSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Always runs iterative rho2 mode (regular streaming not supported for rho2)."""
    if not spec.iterative_noise:
        logger.warning(
            "iterative_noise=False passed to rho2 runner; "
            "proceeding with iterative purification regardless."
        )
    return run_iterative_purification(spec)


def run_and_save(spec: RunSpec) -> Tuple[Path, Path]:
    """Run a single spec and append results to CSVs under spec.out_dir."""
    steps_df, finals_df = run_streaming(spec)
    out_dir = spec.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = spec.noise.noise_type.value
    twirl_suffix = ""
    if spec._should_apply_twirling() and spec.twirling.subset_fraction < 1.0:
        twirl_suffix = f"_subset{spec.twirling.subset_fraction:.2f}"
        
    steps_path  = out_dir / f"steps_rho2_{suffix}{twirl_suffix}.csv"
    finals_path = out_dir / f"finals_rho2_{suffix}{twirl_suffix}.csv"

    for path, df in [(steps_path, steps_df), (finals_path, finals_df)]:
        if path.exists():
            try:
                prev = pd.read_csv(path)
                if len(prev) > 0:
                    df = pd.concat([prev, df], ignore_index=True)
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
                logger.warning(f"Could not read {path}: {exc}. Overwriting.")
        df.to_csv(path, index=False)

    logger.info(f"Results saved to {steps_path.parent}")
    return steps_path, finals_path


__all__ = [
    "run_streaming",
    "run_and_save",
    "run_iterative_purification",
]