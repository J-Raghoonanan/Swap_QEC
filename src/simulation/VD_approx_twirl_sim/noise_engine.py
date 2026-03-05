"""
Noise engine for the Virtual Distillation (VD) purification simulator.

This module applies noise channels directly to density matrices.  It does NOT
build Qiskit circuits — the VD iterative runner works entirely at the density-
matrix level and never needs circuit-level noise injection.

Supported noise types  (NoiseType enum):
    depolarizing  — E(ρ) = (1-p) ρ + (p/3)(XρX + YρY + ZρZ)
    dephase_z     — E(ρ) = (1-p) ρ + p ZρZ
    dephase_x     — E(ρ) = (1-p) ρ + p XρX

Clifford twirling  (dephase_z / dephase_x only):
    Converts Z-dephasing into effective depolarization by averaging the channel
    over a subset of K local Clifford combinations drawn from {I, H, HS}^⊗M:

        E_twirled(ρ) = (1/K) Σ_{k=1}^{K}  C_k†  E( C_k ρ C_k† )  C_k

    When subset_fraction = 1.0, K = 3^M and this is the exact full twirl.
    When subset_fraction < 1.0, K < 3^M and this is an approximate twirl.
    The subset is drawn once per call (fixed seed → deterministic channel).

Qubit ordering convention  (big-endian throughout):
    All tensor products are built left-to-right with np.kron, so qubit 0 is
    the MOST significant bit.  _build_full_clifford_operator and
    _single_qubit_to_full_operator use the same convention, ensuring that the
    Clifford frame rotation and the per-qubit noise application are consistent.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple
from itertools import product

import numpy as np
from qiskit.quantum_info import Kraus, DensityMatrix

from .configs import NoiseMode, NoiseSpec, NoiseType, TwirlingSpec

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Kraus channel constructors
# ─────────────────────────────────────────────────────────────────────────────

def _kraus_depolarizing(p: float) -> Kraus:
    """Depolarizing channel: E0 = sqrt(1-p) I,  Ej = sqrt(p/3) σj.

    Manuscript relation: δ = 4p/3  →  p = 3δ/4.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Depolarizing p must be in [0, 1], got {p}")
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return Kraus([
        np.sqrt(1.0 - p) * I,
        np.sqrt(p / 3.0) * X,
        np.sqrt(p / 3.0) * Y,
        np.sqrt(p / 3.0) * Z,
    ])


def _kraus_z_dephase(p: float) -> Kraus:
    """Z-dephasing channel: E0 = sqrt(1-p) I,  E1 = sqrt(p) Z."""
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Z-dephasing p must be in [0, 1], got {p}")
    I = np.eye(2, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return Kraus([np.sqrt(1.0 - p) * I, np.sqrt(p) * Z])


def _kraus_x_dephase(p: float) -> Kraus:
    """X-dephasing channel: E0 = sqrt(1-p) I,  E1 = sqrt(p) X."""
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"X-dephasing p must be in [0, 1], got {p}")
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    return Kraus([np.sqrt(1.0 - p) * I, np.sqrt(p) * X])


# ─────────────────────────────────────────────────────────────────────────────
# Clifford subset selection
# ─────────────────────────────────────────────────────────────────────────────

def _get_clifford_subset(
    M: int,
    fraction: float,
    mode: str,
    seed: Optional[int] = None,
) -> List[Tuple[str, ...]]:
    """Return a subset of the 3^M local Clifford combinations {I,H,HS}^⊗M.

    Parameters
    ----------
    M : int
        Number of qubits.
    fraction : float
        Fraction of the full 3^M set to include (0 < fraction ≤ 1).
    mode : str
        'random'  — randomly sample without replacement.
        'first_k' — take the first K in lexicographic order (i, h, hs).
    seed : Optional[int]
        RNG seed for reproducibility (only used in 'random' mode).

    Returns
    -------
    List[Tuple[str, ...]]
        Each tuple has M gate names drawn from {'i', 'h', 'hs'}.

    Notes
    -----
    A minimum subset size is enforced to prevent completely degenerate
    twirling:  at least 2 gates for M ≤ 2,  at least 3 for M > 2.
    """
    options = ["i", "h", "hs"]
    all_combinations: List[Tuple[str, ...]] = list(product(options, repeat=M))
    total = len(all_combinations)          # 3^M

    subset_size_raw = max(1, int(np.ceil(fraction * total)))

    # Enforce a practical minimum so the subset spans more than one direction.
    min_subset_size = min(total, 2 if M <= 2 else 3)
    subset_size = max(subset_size_raw, min_subset_size)

    if subset_size != subset_size_raw:
        logger.debug(
            f"Subset size increased from {subset_size_raw} to {subset_size} "
            f"(minimum for M={M}).  Consider fraction >= {min_subset_size / total:.2f}."
        )

    logger.debug(
        f"Clifford subset: M={M}, total={total}, "
        f"fraction={fraction:.3f}, subset_size={subset_size}"
    )

    if subset_size >= total:
        logger.debug("Using full Clifford set (fraction ≥ 1.0)")
        return all_combinations

    if mode == "random":
        rng = np.random.default_rng(seed)
        indices = rng.choice(total, size=subset_size, replace=False)
        subset = [all_combinations[i] for i in sorted(indices)]
        logger.debug(f"Randomly sampled {subset_size}/{total} Clifford combinations")
    elif mode == "first_k":
        subset = all_combinations[:subset_size]
        logger.debug(f"Using first {subset_size}/{total} Clifford combinations")
    else:
        raise ValueError(f"Unknown subset mode: {mode!r}  (choose 'random' or 'first_k')")

    return subset


# ─────────────────────────────────────────────────────────────────────────────
# Clifford unitary constructors
# ─────────────────────────────────────────────────────────────────────────────

def _get_clifford_unitary(gate_name: str) -> np.ndarray:
    """2×2 unitary for gate_name in {'i', 'h', 's', 'sdg', 'hs', 'sdgh'}."""
    if gate_name == "i":
        return np.eye(2, dtype=complex)
    if gate_name == "h":
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    if gate_name == "s":
        return np.array([[1, 0], [0, 1j]], dtype=complex)
    if gate_name == "sdg":
        return np.array([[1, 0], [0, -1j]], dtype=complex)
    if gate_name == "hs":
        # H·S  maps  Z → Y
        S = np.array([[1, 0], [0, 1j]], dtype=complex)
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        return H @ S
    if gate_name == "sdgh":
        Sdg = np.array([[1, 0], [0, -1j]], dtype=complex)
        H   = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        return H @ Sdg
    raise ValueError(f"Unknown Clifford gate: {gate_name!r}")


def _build_full_clifford_operator(gate_names: Tuple[str, ...], M: int) -> np.ndarray:
    """Build the 2^M × 2^M unitary U(g₀) ⊗ U(g₁) ⊗ … ⊗ U(g_{M-1}).

    Qubit 0 is the most significant bit (big-endian), consistent with
    _single_qubit_to_full_operator.
    """
    if len(gate_names) != M:
        raise ValueError(f"Expected {M} gate names, got {len(gate_names)}")
    result = _get_clifford_unitary(gate_names[0])
    for g in gate_names[1:]:
        result = np.kron(result, _get_clifford_unitary(g))
    return result


def _single_qubit_to_full_operator(
    single_qubit_op: np.ndarray,
    target_qubit: int,
    M: int,
) -> np.ndarray:
    """Embed a 2×2 operator into the 2^M space acting on target_qubit (MSB = 0).

    Builds:  I^{⊗ target_qubit}  ⊗  single_qubit_op  ⊗  I^{⊗ (M-target_qubit-1)}
    using the same big-endian np.kron convention as _build_full_clifford_operator.
    """
    if not (0 <= target_qubit < M):
        raise ValueError(f"target_qubit must be in [0, {M-1}], got {target_qubit}")
    result = np.array([[1.0]], dtype=complex)
    for q in range(M):
        result = np.kron(result, single_qubit_op if q == target_qubit else np.eye(2, dtype=complex))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Core DM noise application
# ─────────────────────────────────────────────────────────────────────────────

def _apply_noise_without_twirling(
    rho: DensityMatrix,
    noise: NoiseSpec,
    M: int,
    p: float,
) -> DensityMatrix:
    """Apply the raw iid per-qubit noise channel to rho (no frame rotation)."""
    if noise.noise_type == NoiseType.depolarizing:
        kraus_channel = _kraus_depolarizing(p)
    elif noise.noise_type == NoiseType.dephase_z:
        kraus_channel = _kraus_z_dephase(p)
    elif noise.noise_type == NoiseType.dephase_x:
        kraus_channel = _kraus_x_dephase(p)
    else:
        raise ValueError(f"Unsupported noise type: {noise.noise_type}")

    result_rho = rho
    for q in range(M):
        new_data = np.zeros_like(result_rho.data)
        for E in kraus_channel.data:
            F = _single_qubit_to_full_operator(E, q, M)
            new_data += F @ result_rho.data @ F.conj().T
        result_rho = DensityMatrix(new_data)

    logger.debug(f"Applied {noise.noise_type.value} noise (p={p:.4f}) without twirling")
    return result_rho


def _apply_noise_with_twirling(
    rho: DensityMatrix,
    noise: NoiseSpec,
    M: int,
    p: float,
    twirling: TwirlingSpec,
    twirl_seed: Optional[int],
) -> DensityMatrix:
    """Apply noise averaged over a Clifford subset  (subset or full twirl).

    Computes:
        E_twirled(ρ) = (1/K) Σ_{k=1}^{K}  C_k†  E( C_k ρ C_k† )  C_k

    where each C_k = U(g₀) ⊗ … ⊗ U(g_{M-1}) from the chosen subset and E is
    the raw per-qubit dephasing channel.

    When subset_fraction = 1.0, K = 3^M and this is the exact full twirl.
    """
    if noise.noise_type == NoiseType.dephase_z:
        kraus_channel = _kraus_z_dephase(p)
    elif noise.noise_type == NoiseType.dephase_x:
        kraus_channel = _kraus_x_dephase(p)
    else:
        raise ValueError(f"Twirling not applicable to {noise.noise_type}")

    # Fixed seed: same subset on every call within a run → deterministic channel.
    effective_seed = (
        twirling.subset_seed if twirling.subset_seed is not None else twirl_seed
    )
    clifford_subset = _get_clifford_subset(
        M=M,
        fraction=twirling.subset_fraction,
        mode=twirling.subset_mode,
        seed=effective_seed,
    )

    acc = np.zeros_like(rho.data, dtype=complex)

    for combo in clifford_subset:
        C     = _build_full_clifford_operator(combo, M)
        Cdag  = C.conj().T

        # Rotate into Clifford frame: ρ' = C ρ C†
        rho_rot_data = C @ rho.data @ Cdag

        # Apply raw noise to each qubit in the rotated frame
        noisy_data = rho_rot_data.copy()
        for q in range(M):
            temp = np.zeros_like(noisy_data)
            for E in kraus_channel.data:
                F = _single_qubit_to_full_operator(E, q, M)
                temp += F @ noisy_data @ F.conj().T
            noisy_data = temp

        # Rotate back: C† (…) C
        acc += Cdag @ noisy_data @ C

    acc /= float(len(clifford_subset))

    logger.debug(
        f"Applied {noise.noise_type.value} with Clifford twirling "
        f"(M={M}, p={p:.4f}, K={len(clifford_subset)}/{3**M})"
    )
    return DensityMatrix(acc)


def apply_noise_to_density_matrix(
    rho: DensityMatrix,
    noise: NoiseSpec,
    twirling: Optional[TwirlingSpec] = None,
    twirl_seed: Optional[int] = None,
) -> DensityMatrix:
    """Apply one round of iid per-qubit noise to a density matrix.

    Parameters
    ----------
    rho : DensityMatrix
        Input state.
    noise : NoiseSpec
        Noise type and probability.  Must use NoiseMode.iid_p.
    twirling : Optional[TwirlingSpec]
        If provided and noise is dephase_z / dephase_x, the channel is
        averaged over a subset of local Clifford combinations.
        Pass None to apply the raw channel (no frame rotation).
    twirl_seed : Optional[int]
        Fallback RNG seed for subset selection when twirling.subset_seed is None.

    Returns
    -------
    DensityMatrix
        Noisy output state.
    """
    if noise.mode != NoiseMode.iid_p:
        raise ValueError("apply_noise_to_density_matrix only supports iid_p mode")

    M = int(round(np.log2(rho.dim)))
    p = noise.kraus_p()

    should_twirl = (
        twirling is not None
        and twirling.enabled
        and noise.noise_type in (NoiseType.dephase_z, NoiseType.dephase_x)
    )

    if not should_twirl:
        return _apply_noise_without_twirling(rho, noise, M, p)

    return _apply_noise_with_twirling(rho, noise, M, p, twirling, twirl_seed)


__all__ = [
    "apply_noise_to_density_matrix",
]