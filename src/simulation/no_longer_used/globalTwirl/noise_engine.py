"""
Noise engine for the SWAP-based purification simulator (Qiskit).

This module builds *noisy input copies* ρ from a given target preparation
circuit U_psi. Two modes are supported:

(A) iid_p      – Apply a CPTP channel independently to each qubit with
                 probability p.
(B) exact_k    – Deterministically inject exactly k single-qubit Pauli faults
                 (Z/X for dephasing, uniform {X,Y,Z} for depolarizing).

CRITICAL FIX: Clifford twirling is now implemented correctly as CHANNEL twirling
per manuscript Eq. (54). For dephasing noise, we:
  1. Apply random Clifford C to |ψ⟩ → |ψ'⟩ = C|ψ⟩
  2. Apply dephasing channel in rotated frame
  3. Apply C† to get back: C† E_deph(C|ψ⟩⟨ψ|C†) C
This averages Z → (X+Y+Z)/3 over multiple copies with independent Cliffords.

NEW in subsetTwirling: Added subset_fraction parameter to TwirlingSpec to allow
using only a fraction of the full 3^M Clifford combinations for resource efficiency.

Returned objects are *circuits on M data qubits* that prepare the noisy state
from |0...0>.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Kraus

from .configs import NoiseMode, NoiseSpec, NoiseType, TwirlingSpec

logger = logging.getLogger(__name__)

# -----------------------------
# Explicit Kraus operators
# -----------------------------

def _kraus_depolarizing(p: float) -> Kraus:
    """Depolarizing channel with Kraus form:
    E0 = sqrt(1-p) I, Ej = sqrt(p/3) σj for j ∈ {X,Y,Z}
    
    This gives: ρ → (1-p)ρ + (p/3)(XρX + YρY + ZρZ)
    
    Manuscript relation: δ = 4p/3, so p = 3δ/4.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Depolarizing p must be in [0,1], got {p}")
    
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    E0 = np.sqrt(1.0 - p) * I
    Ex = np.sqrt(p / 3.0) * X
    Ey = np.sqrt(p / 3.0) * Y
    Ez = np.sqrt(p / 3.0) * Z
    
    logger.debug(f"Depolarizing channel: p={p:.4f}, δ_equiv={4*p/3:.4f}")
    return Kraus([E0, Ex, Ey, Ez])


def _kraus_z_dephase(p: float) -> Kraus:
    """Pure Z-dephasing channel:
    E0 = sqrt(1-p) I, E1 = sqrt(p) Z
    
    This gives: ρ → (1-p)ρ + p ZρZ
    
    For manuscript: δ = p (we take δ directly as the dephasing probability).
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Z-dephasing p must be in [0,1], got {p}")
    
    I = np.eye(2, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    E0 = np.sqrt(1.0 - p) * I
    E1 = np.sqrt(p) * Z
    
    logger.debug(f"Z-dephasing channel: p={p:.4f}")
    return Kraus([E0, E1])


def _kraus_x_dephase(p: float) -> Kraus:
    """Pure X-dephasing channel:
    E0 = sqrt(1-p) I, E1 = sqrt(p) X
    
    This gives: ρ → (1-p)ρ + p XρX
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"X-dephasing p must be in [0,1], got {p}")
    
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    
    E0 = np.sqrt(1.0 - p) * I
    E1 = np.sqrt(p) * X
    
    logger.debug(f"X-dephasing channel: p={p:.4f}")
    return Kraus([E0, E1])


# -----------------------------
# Clifford twirling support
# -----------------------------

def _sample_clifford_gate(mode: str, index: int, seed: Optional[int] = None) -> str:
    """Sample a single-qubit Clifford gate name.
    
    mode='random': uniformly random from {I, H, SH}
    mode='cyclic': deterministically cycle through set
    
    These three gates map Z to {Z, X, Y} respectively:
    - I: Z → Z  
    - H: Z → X
    - SH: Z → Y
    
    Returns gate name as string.
    """
    options = ['i', 'h', 'hs']  # Simplified set for X,Y,Z mapping
    
    if mode == "random":
        rng = np.random.default_rng(seed)
        return str(rng.choice(options))
    else:  # cyclic
        return options[index % len(options)]


def _apply_clifford_gate(qc: QuantumCircuit, qubit: int, gate_name: str) -> None:
    """Apply a single-qubit Clifford gate to the circuit."""
    if gate_name == 'i':
        pass  # identity, do nothing
    elif gate_name == 'h':
        qc.h(qubit)
    elif gate_name == 's':
        qc.s(qubit)
    elif gate_name == 'sdg':
        qc.sdg(qubit)
    elif gate_name == 'hs':
        qc.s(qubit)
        qc.h(qubit)
    elif gate_name == 'sdgh':
        qc.sdg(qubit)
        qc.h(qubit)
    else:
        raise ValueError(f"Unknown Clifford gate: {gate_name}")


def _apply_inverse_clifford_gate(qc: QuantumCircuit, qubit: int, gate_name: str) -> None:
    """Apply the inverse of a Clifford gate."""
    # For single-qubit Cliffords, inverse is straightforward
    if gate_name == 'i':
        pass
    elif gate_name == 'h':
        qc.h(qubit)  # H† = H
    elif gate_name == 's':
        qc.sdg(qubit)  # S† = Sdg
    elif gate_name == 'sdg':
        qc.s(qubit)  # Sdg† = S
    elif gate_name == 'hs':
        qc.h(qubit)  # Reverse order
        qc.sdg(qubit)
    elif gate_name == 'sdgh':
        qc.h(qubit)
        qc.s(qubit)
    else:
        raise ValueError(f"Unknown Clifford gate: {gate_name}")


def _get_clifford_subset(M: int, fraction: float, mode: str, seed: Optional[int] = None) -> List[Tuple[str, ...]]:
    """Generate a subset of Clifford gate combinations for twirling.
    
    Parameters
    ----------
    M : int
        Number of qubits
    fraction : float
        Fraction of full 3^M set to include (0.0 to 1.0)
    mode : str
        'random': randomly sample subset
        'first_k': take first k combinations in lexicographic order
    seed : Optional[int]
        RNG seed for reproducibility
        
    Returns
    -------
    List of tuples, where each tuple contains M gate names (one per qubit).
    Each gate name is from {'i', 'h', 'hs'}.
    
    Example for M=2, fraction=0.5:
        Full set has 3^2 = 9 combinations
        Subset will have 5 combinations (rounded up from 4.5)
    """
    options = ['i', 'h', 'hs']
    
    # Generate all possible combinations
    all_combinations = list(product(options, repeat=M))
    total = len(all_combinations)  # This is 3^M
    
    # Determine subset size
    subset_size_raw = max(1, int(np.ceil(fraction * total)))
    
    # CRITICAL: Enforce minimum subset size to ensure effective twirling
    # For very small systems (M=1,2), using only 1 Clifford can make twirling
    # completely ineffective if it picks an unfortunate gate. 
    # Require at least 2 gates for M≤2, or at least 3 for M>2.
    min_subset_size = min(total, 2 if M <= 2 else 3)
    subset_size = max(subset_size_raw, min_subset_size)
    
    if subset_size != subset_size_raw:
        logger.warning(
            f"Subset size increased from {subset_size_raw} to {subset_size} "
            f"(minimum required for M={M} to ensure effective twirling). "
            f"Consider using fraction >= {min_subset_size/total:.2f}"
        )
    
    logger.debug(f"Clifford subset: M={M}, total={total}, fraction={fraction:.2f}, subset_size={subset_size}")
    
    if subset_size >= total:
        logger.debug("Using full Clifford set (fraction >= 1.0)")
        return all_combinations
    
    if mode == "random":
        rng = np.random.default_rng(seed)
        indices = rng.choice(total, size=subset_size, replace=False)
        subset = [all_combinations[i] for i in indices]
        logger.debug(f"Randomly sampled {subset_size}/{total} Clifford combinations")
    elif mode == "first_k":
        subset = all_combinations[:subset_size]
        logger.debug(f"Using first {subset_size}/{total} Clifford combinations")
    else:
        raise ValueError(f"Unknown subset mode: {mode}")
    
    return subset


# -----------------------------
# Error pattern (for exact_k)
# -----------------------------

@dataclass(frozen=True)
class ErrorOp:
    qubit: int
    pauli: str  # one of {"X","Y","Z"}


ErrorPattern = Tuple[ErrorOp, ...]


def sample_error_pattern(
    M: int,
    noise_type: NoiseType,
    k: int,
    seed: Optional[int] = None,
) -> ErrorPattern:
    """Sample a deterministic pattern of exactly k single-qubit faults.

    For dephase_z: only Z faults
    For dephase_x: only X faults
    For depolarizing: uniform over {X,Y,Z}
    
    NOTE: This generates a SINGLE pattern that should be shared between
    both copies entering the SWAP test to ensure they are identical.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    if k == 0:
        return tuple()
    if k > M:
        raise ValueError(f"k={k} cannot exceed M={M} for single-qubit faults")

    rng = np.random.default_rng(seed)
    qubits = rng.choice(M, size=k, replace=False)
    ops: List[ErrorOp] = []
    
    for q in qubits:
        if noise_type == NoiseType.dephase_z:
            ops.append(ErrorOp(int(q), "Z"))
        elif noise_type == NoiseType.dephase_x:
            ops.append(ErrorOp(int(q), "X"))
        else:  # depolarizing
            pauli = rng.choice(["X", "Y", "Z"])  # uniform
            ops.append(ErrorOp(int(q), str(pauli)))
    
    # sort by qubit index for determinism
    ops.sort(key=lambda e: e.qubit)
    
    logger.debug(f"Sampled error pattern (k={k}): {[(e.qubit, e.pauli) for e in ops]}")
    return tuple(ops)


def apply_error_pattern(qc: QuantumCircuit, pattern: ErrorPattern) -> None:
    """Append the specified single-qubit Pauli gates to the circuit."""
    for op in pattern:
        if op.pauli == "X":
            qc.x(op.qubit)
        elif op.pauli == "Y":
            qc.y(op.qubit)
        elif op.pauli == "Z":
            qc.z(op.qubit)
        else:
            raise ValueError(f"Unknown Pauli '{op.pauli}' in pattern")


# -----------------------------
# Build noisy copies (circuit-based)
# -----------------------------

def build_copy_iid_p(
    prep: QuantumCircuit,
    noise: NoiseSpec,
    twirling: Optional[TwirlingSpec] = None,
    twirl_seed: Optional[int] = None,
) -> QuantumCircuit:
    """Build a circuit that prepares |ψ⟩ and then applies an iid_p noise channel.

    Clifford twirling (if enabled and applicable) conjugates the noise application
    with random single-qubit Cliffords per qubit, converting dephasing channels
    into effective depolarization at the ensemble level.

    NOTE: This function creates a SINGLE noisy copy. For twirling to take effect,
    multiple copies with independent Clifford choices must be averaged.
    """
    M = prep.num_qubits
    p = noise.kraus_p()

    # Determine if we should apply Clifford frame randomization
    should_twirl = (
        twirling is not None
        and twirling.enabled
        and noise.noise_type in [NoiseType.dephase_z, NoiseType.dephase_x]
    )

    # Step 1: Prepare |ψ⟩
    qc = prep.copy(name="noisy_iid_p")

    # Step 2: Apply random Clifford frame (if twirling)
    clifford_gates: List[str] = []
    if should_twirl:
        logger.debug(f"Applying Clifford twirling frame (mode={twirling.mode})")
        for q in range(M):
            qubit_seed = (twirl_seed + q) if twirl_seed is not None else None
            gate_name = _sample_clifford_gate(twirling.mode, index=q, seed=qubit_seed)
            clifford_gates.append(gate_name)
            _apply_clifford_gate(qc, q, gate_name)

    # Step 3: Apply noise channel in (possibly rotated) frame
    logger.debug(f"Building iid_p copy: M={M}, noise={noise.noise_type.value}, p={p:.4f}")

    if noise.noise_type == NoiseType.depolarizing:
        chan_instr = _kraus_depolarizing(p).to_instruction()
    elif noise.noise_type == NoiseType.dephase_z:
        chan_instr = _kraus_z_dephase(p).to_instruction()
    elif noise.noise_type == NoiseType.dephase_x:
        chan_instr = _kraus_x_dephase(p).to_instruction()
    else:
        raise ValueError(f"Unsupported noise type: {noise.noise_type}")

    # Apply channel to each qubit independently
    for q in range(M):
        qc.append(chan_instr, [q])

    # Step 4: Undo Clifford frame if twirling was applied
    if should_twirl:
        logger.debug("Undoing Clifford frame")
        for q in range(M):
            _apply_inverse_clifford_gate(qc, q, clifford_gates[q])

    return qc


def build_copy_exact_k(prep: QuantumCircuit, pattern: ErrorPattern) -> QuantumCircuit:
    """Return a circuit that prepares |ψ⟩ and then injects the given pattern.

    The pattern specifies *deterministic* Pauli faults to apply after preparation.
    """
    qc = prep.copy(name="noisy_exact_k")
    apply_error_pattern(qc, pattern)
    return qc


def build_noisy_copy(
    prep: QuantumCircuit,
    noise: NoiseSpec,
    seed: Optional[int] = None,
    shared_pattern: Optional[ErrorPattern] = None,
    twirling: Optional[TwirlingSpec] = None,
    twirl_seed: Optional[int] = None,
) -> Tuple[QuantumCircuit, Optional[ErrorPattern]]:
    """Factory that returns a noisy-copy circuit and the pattern used (if any).

    Parameters
    ----------
    prep : QuantumCircuit
        Preparation circuit for |ψ⟩ on M qubits.
    noise : NoiseSpec
        Noise configuration (type, mode, delta, exact_k, etc.).
    seed : Optional[int]
        RNG seed for sampling patterns (exact_k) when not provided.
    shared_pattern : Optional[ErrorPattern]
        If provided (and mode == exact_k), this pattern is used instead of
        sampling – CRITICAL for ensuring identical copies in SWAP test.
    twirling : Optional[TwirlingSpec]
        Clifford twirling configuration for dephasing noise mitigation.
    twirl_seed : Optional[int]
        Seed for Clifford sampling (reproducibility).

    Returns
    -------
    (qc, pattern)
        qc: noisy-copy circuit on M qubits starting from |0...0⟩.
        pattern: the ErrorPattern used (None for iid_p mode).
    """
    if noise.mode == NoiseMode.iid_p:
        logger.debug("Building iid_p noisy copy with channel twirling support")
        return build_copy_iid_p(prep, noise, twirling, twirl_seed), None

    # exact_k mode (twirling not implemented for exact_k)
    if shared_pattern is not None:
        logger.debug(f"Using shared pattern with {len(shared_pattern)} errors")
        pattern = shared_pattern
    else:
        logger.debug(f"Sampling new error pattern (k={noise.exact_k})")
        pattern = sample_error_pattern(
            M=prep.num_qubits,
            noise_type=noise.noise_type,
            k=noise.exact_k,
            seed=seed,
        )
    return build_copy_exact_k(prep, pattern), pattern


def apply_noise_to_density_matrix(
    rho: "DensityMatrix",
    noise: NoiseSpec,
    twirling: Optional[TwirlingSpec] = None,
    twirl_seed: Optional[int] = None,
) -> "DensityMatrix":
    """Apply noise channel directly to a density matrix.
    
    This enables iterative noise application where we need to apply
    fresh noise to an already-processed state.
    
    NEW in subsetTwirling: When twirling is enabled with fraction < 1.0,
    this function averages over a subset of Clifford combinations rather
    than the full 3^M set, reducing computational cost.
    
    Parameters
    ----------
    rho : DensityMatrix
        Input density matrix to apply noise to.
    noise : NoiseSpec
        Noise specification.
    twirling : Optional[TwirlingSpec]
        Clifford twirling configuration for dephasing noise.
    twirl_seed : Optional[int]
        Seed for Clifford sampling.
        
    Returns
    -------
    DensityMatrix
        Noisy output density matrix.
    """
    # Import here to avoid circular dependency
    from qiskit.quantum_info import DensityMatrix
    
    # Only implement for iid_p mode (exact_k doesn't make sense for iterative)
    if noise.mode != NoiseMode.iid_p:
        raise ValueError("apply_noise_to_density_matrix only supports iid_p mode")
    
    M = int(np.log2(rho.dim))
    p = noise.kraus_p()
    
    # Determine if we should apply twirling
    should_twirl = (
        twirling is not None 
        and twirling.enabled 
        and noise.noise_type in [NoiseType.dephase_z, NoiseType.dephase_x]
    )
    
    if not should_twirl:
        # No twirling: apply noise directly to each qubit
        return _apply_noise_without_twirling(rho, noise, M, p)
    
    # Twirling enabled: average over Clifford combinations
    return _apply_noise_with_twirling(rho, noise, M, p, twirling, twirl_seed)


def _apply_noise_without_twirling(
    rho: "DensityMatrix",
    noise: NoiseSpec,
    M: int,
    p: float,
) -> "DensityMatrix":
    """Apply noise to density matrix without Clifford twirling."""
    from qiskit.quantum_info import DensityMatrix
    
    # Get the appropriate Kraus operators
    if noise.noise_type == NoiseType.depolarizing:
        kraus_channel = _kraus_depolarizing(p)
    elif noise.noise_type == NoiseType.dephase_z:
        kraus_channel = _kraus_z_dephase(p)
    elif noise.noise_type == NoiseType.dephase_x:
        kraus_channel = _kraus_x_dephase(p)
    else:
        raise ValueError(f"Unsupported noise type: {noise.noise_type}")
    
    # Apply noise to each qubit independently
    result_rho = rho
    
    for q in range(M):
        new_rho_data = np.zeros_like(result_rho.data)
        
        for kraus_op in kraus_channel.data:
            # Extend single-qubit Kraus operator to full system
            full_kraus = _single_qubit_to_full_operator(kraus_op, q, M)
            new_rho_data += full_kraus @ result_rho.data @ full_kraus.conj().T
        
        result_rho = DensityMatrix(new_rho_data)
    
    logger.debug(f"Applied {noise.noise_type.value} noise (p={p:.4f}) to {M}-qubit density matrix")
    return result_rho


def _apply_noise_with_twirling(
    rho: "DensityMatrix",
    noise: NoiseSpec,
    M: int,
    p: float,
    twirling: TwirlingSpec,
    twirl_seed: Optional[int],
) -> "DensityMatrix":
    """Apply noise to density matrix WITH Clifford twirling.
    
    This averages the channel over a subset of Clifford combinations:
    
        E_twirled(ρ) = (1/K) Σ_k C_k† E(C_k ρ C_k†) C_k
    
    where K is the subset size and each C_k is a product of single-qubit Cliffords.
    """
    from qiskit.quantum_info import DensityMatrix
    
    # Get Kraus operators for the base channel
    if noise.noise_type == NoiseType.dephase_z:
        kraus_channel = _kraus_z_dephase(p)
    elif noise.noise_type == NoiseType.dephase_x:
        kraus_channel = _kraus_x_dephase(p)
    else:
        raise ValueError(f"Twirling not applicable to {noise.noise_type}")
    
    # Get subset of Clifford combinations
    clifford_subset = _get_clifford_subset(
        M=M,
        fraction=twirling.subset_fraction,
        mode=twirling.subset_mode,
        seed=twirling.subset_seed if twirling.subset_seed is not None else twirl_seed,
    )
    
    # Accumulator for averaged result
    averaged_rho_data = np.zeros_like(rho.data)
    
    # Average over Clifford combinations
    for clifford_combo in clifford_subset:
        # clifford_combo is a tuple of M gate names, one per qubit
        # Build full M-qubit Clifford operator
        C_full = _build_full_clifford_operator(clifford_combo, M)
        C_full_inv = C_full.conj().T
        
        # Rotate: ρ' = C ρ C†
        rho_rotated_data = C_full @ rho.data @ C_full_inv
        
        # Apply noise to each qubit in rotated frame
        noisy_rho_data = rho_rotated_data.copy()
        
        for q in range(M):
            temp_data = np.zeros_like(noisy_rho_data)
            
            for kraus_op in kraus_channel.data:
                full_kraus = _single_qubit_to_full_operator(kraus_op, q, M)
                temp_data += full_kraus @ noisy_rho_data @ full_kraus.conj().T
            
            noisy_rho_data = temp_data
        
        # Rotate back: C† ρ' C
        rotated_back_data = C_full_inv @ noisy_rho_data @ C_full
        
        # Add to average
        averaged_rho_data += rotated_back_data
    
    # Normalize by number of Clifford combinations
    averaged_rho_data /= len(clifford_subset)
    
    logger.debug(
        f"Applied {noise.noise_type.value} with Clifford twirling: "
        f"M={M}, p={p:.4f}, subset={len(clifford_subset)}/{3**M}"
    )
    
    return DensityMatrix(averaged_rho_data)


def _build_full_clifford_operator(gate_names: Tuple[str, ...], M: int) -> np.ndarray:
    """Build full M-qubit Clifford operator from gate names.
    
    Parameters
    ----------
    gate_names : Tuple[str, ...]
        Tuple of M gate names, one per qubit
    M : int
        Number of qubits
        
    Returns
    -------
    np.ndarray
        2^M × 2^M unitary matrix representing the full Clifford operator
    """
    if len(gate_names) != M:
        raise ValueError(f"Expected {M} gate names, got {len(gate_names)}")
    
    # Start with first qubit's gate
    result = _get_clifford_unitary(gate_names[0])
    
    # Tensor product with remaining qubits
    for gate_name in gate_names[1:]:
        single_qubit_op = _get_clifford_unitary(gate_name)
        result = np.kron(result, single_qubit_op)
    
    return result


def _get_clifford_unitary(gate_name: str) -> np.ndarray:
    """Get unitary matrix for single-qubit Clifford gate."""
    if gate_name == 'i':
        return np.eye(2, dtype=complex)
    elif gate_name == 'h':
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    elif gate_name == 's':
        return np.array([[1, 0], [0, 1j]], dtype=complex)
    elif gate_name == 'sdg':
        return np.array([[1, 0], [0, -1j]], dtype=complex)
    elif gate_name == 'hs':
        S = np.array([[1, 0], [0, 1j]], dtype=complex)
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        return H @ S
    elif gate_name == 'sdgh':
        Sdg = np.array([[1, 0], [0, -1j]], dtype=complex)
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        return H @ Sdg
    else:
        raise ValueError(f"Unknown Clifford gate: {gate_name}")


def _single_qubit_to_full_operator(single_qubit_op: np.ndarray, target_qubit: int, M: int) -> np.ndarray:
    """Extend single-qubit operator to full M-qubit system.
    
    Returns I^{⊗target_qubit} ⊗ single_qubit_op ⊗ I^{⊗(M-target_qubit-1)}
    """
    if target_qubit < 0 or target_qubit >= M:
        raise ValueError(f"target_qubit must be in [0, {M-1}], got {target_qubit}")
    
    # Build the full operator as a tensor product
    result = np.array([[1.0]], dtype=complex)  # Start with scalar 1
    
    for q in range(M):
        if q == target_qubit:
            result = np.kron(result, single_qubit_op)
        else:
            result = np.kron(result, np.eye(2, dtype=complex))
    
    return result


__all__ = [
    "ErrorOp",
    "ErrorPattern",
    "sample_error_pattern",
    "apply_error_pattern",
    "build_copy_iid_p",
    "build_copy_exact_k",
    "build_noisy_copy",
    # New functions for iterative noise
    "apply_noise_to_density_matrix",
]