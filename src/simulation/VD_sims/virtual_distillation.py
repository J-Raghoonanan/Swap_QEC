"""
Virtual Distillation for quantum error mitigation.

This module implements virtual distillation (VD) as described in Huggins et al. 
(Phys. Rev. X 11, 041036, 2021). Unlike SWAP-based purification, VD is 
**deterministic** and requires no postselection.

KEY DIFFERENCES FROM SWAP PURIFICATION:
- State update: ρ → ρ²/Tr(ρ²)  (instead of (ρ+ρ²)/(2P))
- Success probability: P_success = 1.0 always (deterministic)
- No ancilla qubit needed
- No SWAP test unitary
- Same fidelity evolution for single qubits: F_out = F²/(F² + (1-F)²)

RESOURCE COSTS:
- C_ℓ = 2^ℓ exactly (no postselection overhead)
- G_ℓ = 2^ℓ - 1 (total VD operations through level ℓ)
- ~50% fewer copies needed vs SWAP purification
- ~25% fewer operations needed vs SWAP purification

The implementation is much simpler than SWAP purification since we just need
to compute ρ², normalize it, and return P_success=1.0.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
from qiskit.quantum_info import DensityMatrix

from .configs import AASpec

logger = logging.getLogger(__name__)


def apply_virtual_distillation(
    rho: DensityMatrix,
    aa: AASpec,  # Keep for API compatibility, but unused
) -> Tuple[DensityMatrix, Dict]:
    """
    Apply virtual distillation to purify a single noisy state.
    
    Virtual distillation computes the expectation value with respect to 
    ρ²/Tr(ρ²) WITHOUT preparing multiple copies. This is the M=2 case
    of the general virtual distillation protocol.
    
    State update: ρ → ρ²/Tr(ρ²)
    
    This is deterministic - no postselection needed, so P_success = 1.0 always.
    
    Parameters
    ----------
    rho : DensityMatrix
        Input noisy density matrix on M qubits.
    aa : AASpec
        Amplitude amplification config (unused for VD, kept for API compatibility).
        
    Returns
    -------
    rho_out : DensityMatrix
        The purified density matrix ρ²/Tr(ρ²).
    metrics : dict
        {"P_success": 1.0, "grover_iters": 0}
        P_success is always 1.0 for VD (deterministic operation).
    """
    M = int(np.log2(rho.dim))
    logger.debug(f"Applying virtual distillation to M={M} qubit state (dim={rho.dim})")
    
    # Step 1: Compute ρ²
    rho_squared = rho.data @ rho.data
    
    # Step 2: Compute normalization Tr(ρ²)
    trace_rho_squared = np.real(np.trace(rho_squared))
    
    # Handle edge case (shouldn't happen for physical states)
    if trace_rho_squared <= 1e-12:
        logger.warning(f"Tr(ρ²) ≈ 0 (value={trace_rho_squared:.2e}); returning zero state")
        return DensityMatrix(np.zeros_like(rho.data)), {"P_success": 0.0, "grover_iters": 0}
    
    # Step 3: Normalize: ρ_out = ρ²/Tr(ρ²)
    rho_out_data = rho_squared / trace_rho_squared
    rho_out = DensityMatrix(rho_out_data)
    
    logger.debug(f"Virtual distillation complete: Tr(ρ²)={trace_rho_squared:.6f}")
    
    # Metrics: VD is deterministic, so P_success = 1.0 always
    metrics = {
        "P_success": 1.0,  # Always deterministic
        "grover_iters": 0,  # No amplitude amplification needed
    }
    
    logger.info(f"VD purification complete: P_success={metrics['P_success']:.4f} (deterministic)")
    return rho_out, metrics


def purify_two_from_density(
    rho_A: DensityMatrix,
    rho_B: DensityMatrix,
    aa: AASpec,
) -> Tuple[DensityMatrix, Dict]:
    """
    Virtual distillation wrapper that accepts two density matrices for API compatibility.
    
    CRITICAL: For VD, we only use rho_A and ignore rho_B. The two-copy requirement
    from SWAP purification doesn't apply here - VD operates on a single copy.
    
    In practice, when called from the streaming runner, rho_A and rho_B should be
    identical copies anyway (from the binary tree structure).
    
    Parameters
    ----------
    rho_A : DensityMatrix
        First input density matrix (this is the one we use).
    rho_B : DensityMatrix
        Second input density matrix (ignored in VD, kept for API compatibility).
    aa : AASpec
        Amplitude amplification config (unused, kept for API compatibility).
        
    Returns
    -------
    rho_out : DensityMatrix
        The purified density matrix ρ_A²/Tr(ρ_A²).
    metrics : dict
        {"P_success": 1.0, "grover_iters": 0}
    
    Notes
    -----
    This function exists to maintain API compatibility with the SWAP purification
    code. The streaming runner calls purify_two_from_density() with two inputs,
    but VD only needs one. We use rho_A and ignore rho_B.
    """
    if rho_A.dim != rho_B.dim:
        logger.warning(f"rho_A and rho_B have different dimensions ({rho_A.dim} vs {rho_B.dim})")
    
    # Check if they're actually identical (they should be for the theory to work)
    if not np.allclose(rho_A.data, rho_B.data, rtol=1e-6, atol=1e-8):
        logger.warning(
            "rho_A and rho_B are not identical! VD expects identical inputs from streaming. "
            "Using rho_A only."
        )
    
    # Apply VD to rho_A only
    return apply_virtual_distillation(rho_A, aa)


# -----------------------------
# Utility functions for metrics
# -----------------------------

def compute_purity(rho: DensityMatrix) -> float:
    """Compute purity Tr(ρ²) for diagnostics."""
    return float(np.real(np.trace(rho.data @ rho.data)))


def compute_trace_rho_squared(rho: DensityMatrix) -> float:
    """
    Compute Tr(ρ²) which appears in VD normalization and sample complexity.
    
    For VD, the sample complexity scales as Var ∝ 1/(R·Tr(ρ²)²) where R is
    the number of measurement shots.
    """
    return float(np.real(np.trace(rho.data @ rho.data)))


__all__ = [
    "apply_virtual_distillation",
    "purify_two_from_density",  # API-compatible wrapper
    "compute_purity",
    "compute_trace_rho_squared",
]