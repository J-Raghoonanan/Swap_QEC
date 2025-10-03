"""
SWAP test + (emulated) amplitude amplification for purification with Clifford twirling.

This module constructs and applies the SWAP-test unitary to two identical
M-qubit input states and (optionally) performs:
1. Clifford twirling for dephasing noise mitigation (Section II.D.2)
2. Amplitude amplification (emulated via logging Grover iteration count)

The **conditional output state** given ancilla=0 is independent of whether
amplitude amplification was used; therefore we project onto ancilla |0⟩ and
extract the purified single-register state.

CLIFFORD TWIRLING PROTOCOL:
  1. Sample random single-qubit Clifford C (applied identically to both registers)
  2. Purify under symmetrized noise: Execute SWAP test on C†ρ₁C ⊗ C†ρ₂C
  3. Undo the frame: Apply C to output register

We use Qiskit's DensityMatrix evolutions with circuits (no hand multiplications),
then perform projection and partial trace with quantum_info utilities.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, partial_trace, Operator

from src.simulation.configs import AASpec, TwirlingSpec

logger = logging.getLogger(__name__)

# -----------------------------
# Clifford twirling
# -----------------------------

def _sample_single_qubit_clifford(mode: Literal["random", "cyclic"], index: int = 0, seed: Optional[int] = None) -> str:
    """Sample a single-qubit Clifford from {I, H, SH, HS, S†H, HS†}.
    
    mode='random': uniformly random from the set
    mode='cyclic': deterministically cycle through orthogonal Pauli axes
    
    Returns a string identifier for the Clifford.
    """
    options = ['I', 'H', 'SH', 'HS', 'SdH', 'HSd']
    
    if mode == "random":
        rng = np.random.default_rng(seed)
        return str(rng.choice(options))
    else:  # cyclic
        return options[index % len(options)]


def build_clifford_circuit(M: int, mode: Literal["random", "cyclic"], seed: Optional[int] = None) -> QuantumCircuit:
    """Build a circuit applying single-qubit Cliffords to M qubits.
    
    This circuit is applied identically to both registers entering the SWAP test,
    effectively conjugating the noise channel to make it isotropic.
    """
    qc = QuantumCircuit(M, name="clifford_twirl")
    
    for q in range(M):
        # Different seed per qubit for randomness, or cyclic based on qubit index
        qubit_seed = (seed + q) if seed is not None else None
        cliff = _sample_single_qubit_clifford(mode, index=q, seed=qubit_seed)
        
        if cliff == 'I':
            pass  # identity
        elif cliff == 'H':
            qc.h(q)
        elif cliff == 'SH':
            qc.s(q)
            qc.h(q)
        elif cliff == 'HS':
            qc.h(q)
            qc.s(q)
        elif cliff == 'SdH':
            qc.sdg(q)
            qc.h(q)
        elif cliff == 'HSd':
            qc.h(q)
            qc.sdg(q)
    
    logger.debug(f"Built Clifford circuit (mode={mode}) for M={M}")
    return qc


# -----------------------------
# Building the SWAP-test unitary
# -----------------------------

def build_swap_test_unitary(M: int) -> QuantumCircuit:
    """Return a circuit on (1 + 2M) qubits implementing the SWAP test:
       - H on ancilla
       - M controlled-SWAPs between regA[i] and regB[i] with control ancilla
       - H on ancilla
       
    Qubit order: [anc] + [A0..A{M-1}] + [B0..B{M-1}].
    """
    n = 1 + 2 * M
    qc = QuantumCircuit(n, name="swap_test")
    anc = 0
    A = list(range(1, 1 + M))
    B = list(range(1 + M, 1 + 2 * M))

    qc.h(anc)
    for i in range(M):
        qc.cswap(anc, A[i], B[i])
    qc.h(anc)
    
    logger.debug(f"Built SWAP test unitary for M={M} (total {n} qubits)")
    return qc


# -----------------------------
# Amplitude amplification helpers (emulated)
# -----------------------------

def ancilla_success_probability(rho_after_A: DensityMatrix, M: int) -> float:
    """Compute Pr[ancilla=0] from the (1+2M)-qubit density matrix after SWAP test.
    
    This computes Tr(Π₊ ρ) where Π₊ = |0⟩⟨0|_anc ⊗ I_A ⊗ I_B.
    
    NOTE: This is NOT the same as tracing out A and B first! For controlled gates,
    we must project first, then trace.
    
    Assumes qubit ordering [anc, A..., B...].
    """
    # Build projector Π₊ = |0⟩⟨0| on ancilla, I on registers A and B
    P0_anc = Operator(np.array([[1, 0], [0, 0]], dtype=complex))
    I_A = Operator(np.eye(2**M, dtype=complex))
    I_B = Operator(np.eye(2**M, dtype=complex))
    
    # Full projector: |0⟩⟨0|_anc ⊗ I_A ⊗ I_B
    Pi = P0_anc.tensor(I_A).tensor(I_B)
    
    # Compute Tr(Π ρ)
    projected = Pi @ rho_after_A @ Pi.adjoint()
    p0 = float(np.real(np.trace(projected.data)))
    
    # Clip for numerical safety
    p0 = max(0.0, min(1.0, p0))
    logger.debug(f"Ancilla success probability (Tr(Π₊ ρ)): {p0:.6f}")
    return p0


def choose_grover_iters(P0: float, target_success: float, max_iters: int) -> int:
    """Choose k so that sin²((k+½)θ) ≥ target_success, θ = 2 arcsin √P₀.
    
    If P0 in {0,1}, handle edge cases gracefully.
    """
    P0 = float(P0)
    
    if P0 >= target_success:
        return 0  # Already above target
    if P0 <= 0.0:
        logger.warning("P0 ≤ 0; cannot amplify, returning k=0")
        return 0
    if P0 >= 1.0:
        return 0
    
    # θ = 2 arcsin √P₀
    theta = 2.0 * np.arcsin(np.sqrt(P0))
    
    # Optimal k from geometry: (k + ½)θ ≈ π/2
    k = int(np.floor(np.pi / (2.0 * theta) - 0.5))
    k = max(0, min(k, max_iters))
    
    logger.debug(f"Grover iterations chosen: k={k} (P0={P0:.4f}, target={target_success:.4f})")
    return k


# -----------------------------
# Projection and extraction of purified state
# -----------------------------

def _project_ancilla_zero(rho: DensityMatrix, M: int) -> DensityMatrix:
    """
    Project the ancilla subsystem (qubit 0) onto |0⟩⟨0| and renormalize.

    System order is [anc] ⊗ [A (M qubits)] ⊗ [B (M qubits)].
    """
    # Projector |0⟩⟨0| on ancilla
    P0 = Operator(np.array([[1, 0], [0, 0]], dtype=complex))
    
    # Identities on A and B registers
    IA = Operator(np.eye(2**M, dtype=complex))
    IB = Operator(np.eye(2**M, dtype=complex))

    # Full projector: Π = |0⟩⟨0|_anc ⊗ I_A ⊗ I_B
    Pi = P0.tensor(IA).tensor(IB)

    # Π ρ Π†
    proj = Pi @ rho @ Pi.adjoint()

    # Renormalize by p0 = Tr(Π ρ)
    p0 = float(np.real(np.trace(proj.data)))
    
    if p0 <= 1e-12:
        logger.warning(f"Ancilla projection probability ≈ 0 (p0={p0:.2e}); returning unnormalized state")
        return DensityMatrix(np.zeros_like(rho.data))
    
    logger.debug(f"Ancilla projection: p0={p0:.6f}")
    return DensityMatrix(proj.data / p0)


def extract_purified_register(rho_after_proj: DensityMatrix, M: int) -> DensityMatrix:
    """Partial trace out ancilla (0) and regB (last M) to get ρ_out on regA (middle M).
    
    Qubit ordering: [anc=0] [A:1..M] [B:M+1..2M]
    """
    # Trace out ancilla and register B
    traced = [0] + list(range(1 + M, 1 + 2 * M))
    rho_A = partial_trace(rho_after_proj, qargs=traced)
    
    # Result is a 2^M × 2^M DensityMatrix
    logger.debug(f"Extracted purified register: dimension {rho_A.dim}")
    return rho_A


# -----------------------------
# Public API
# -----------------------------

def purify_two_from_density(
    rho_A: DensityMatrix,
    rho_B: DensityMatrix,
    aa: AASpec,
    twirling: Optional[TwirlingSpec] = None,
    twirling_seed: Optional[int] = None,
) -> Tuple[DensityMatrix, Dict]:
    """
    Purify two M-qubit inputs via the SWAP test with optional Clifford twirling.

    Inputs
    ------
    rho_A, rho_B : DensityMatrix
        Single-register density matrices on the same number of qubits (M).
        CRITICAL: For the theory to hold, these must be IDENTICAL copies.
    aa : AASpec
        Amplitude amplification config. We *emulate* AA by computing and logging
        the required Grover iteration count; we do not apply Q^k explicitly.
    twirling : Optional[TwirlingSpec]
        If provided and enabled, apply Clifford twirling to symmetrize dephasing noise.
    twirling_seed : Optional[int]
        Seed for random Clifford sampling (reproducibility).

    Returns
    -------
    rho_out : DensityMatrix
        The purified single-register state on register A (middle M qubits),
        obtained by projecting ancilla to |0⟩ and tracing out ancilla + register B.
    metrics : dict
        {"P_success": float, "grover_iters": int, "twirling_applied": bool}
    """
    if rho_A.dim != rho_B.dim:
        raise ValueError("rho_A and rho_B must have the same dimension")
    M = int(np.log2(rho_A.dim))

    logger.debug(f"Purifying two M={M} copies (dim={rho_A.dim})")

    # Step 1: Optional Clifford twirling
    twirling_applied = False
    C_circuit = None
    
    if twirling is not None and twirling.enabled:
        logger.info("Applying Clifford twirling for dephasing mitigation")
        C_circuit = build_clifford_circuit(M, mode=twirling.mode, seed=twirling_seed)
        
        # Apply C to both copies
        rho_A = rho_A.evolve(C_circuit)
        rho_B = rho_B.evolve(C_circuit)
        twirling_applied = True

    # Step 2: Joint state: |0⟩⟨0|_anc ⊗ rho_A ⊗ rho_B
    rho_joint = DensityMatrix.from_label('0').tensor(rho_A).tensor(rho_B)
    logger.debug(f"Joint state dimension: {rho_joint.dim} (should be {2**(1+2*M)})")

    # Step 3: SWAP-test unitary
    A = build_swap_test_unitary(M)
    rho_after_A = rho_joint.evolve(A)

    # Step 4: Pre-AA success probability and emulated Grover iteration count
    P0 = ancilla_success_probability(rho_after_A, M)
    k = choose_grover_iters(P0, aa.target_success, aa.max_iters)

    # Step 5: Post-select ancilla=|0⟩, then trace out ancilla + register B → register A
    rho_proj = _project_ancilla_zero(rho_after_A, M)
    rho_out = extract_purified_register(rho_proj, M)

    # Step 6: Undo Clifford frame if twirling was applied
    if twirling_applied and C_circuit is not None:
        logger.debug("Undoing Clifford frame")
        rho_out = rho_out.evolve(C_circuit.inverse())

    metrics = {
        "P_success": P0,
        "grover_iters": k,
        "twirling_applied": twirling_applied,
    }
    
    logger.info(f"Purification complete: P_success={P0:.4f}, k={k}, twirl={twirling_applied}")
    return rho_out, metrics


__all__ = [
    "build_swap_test_unitary",
    "build_clifford_circuit",
    "purify_two_from_density",
    "ancilla_success_probability",
]