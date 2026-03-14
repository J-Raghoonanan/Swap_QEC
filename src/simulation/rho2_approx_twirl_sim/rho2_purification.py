"""
Rho2 purification: ρ → ρ²/Tr(ρ²).

Implements the deterministic rho2 operation for density-matrix simulations.
Given a noisy M-qubit state ρ, the purified output is:

    ρ_out = ρ² / Tr(ρ²)

which can be computed directly as a matrix square — no tensor products, no
SWAP operators, and no partial traces are required.

Key differences from SWAP purification:
  - Deterministic: no postselection, no amplitude amplification.
  - Success probability P = Tr(ρ²), but ρ²/Tr(ρ²) is obtained deterministically.
  - For pure states: Tr(ρ²) = 1, so ρ_out = ρ (no change, as expected).
  - For the maximally mixed state: Tr(ρ²) = 1/2^M (minimum purity).

NOTE ON THE SYMMETRIC PROJECTION APPROACH  (why we do NOT use it here):
  The expression Tr_B[Π_sym (ρ⊗ρ) Π_sym] does NOT give ρ²/Tr(ρ²).
  It gives (ρ+ρ²)/2, which after normalization is (ρ+ρ²)/(1+Tr(ρ²)) — the
  SWAP test result, not rho2.  The direct matrix-square computation below is
  both simpler and correct.

Clifford twirling is handled upstream in noise_engine.py and must NOT be
applied inside this module.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
from qiskit.quantum_info import DensityMatrix

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core rho2 operation
# ─────────────────────────────────────────────────────────────────────────────

def rho2_purification(rho: DensityMatrix) -> Tuple[DensityMatrix, Dict]:
    """Apply one round of Rho2 Purification: ρ → ρ²/Tr(ρ²).

    Parameters
    ----------
    rho : DensityMatrix
        Input M-qubit density matrix (must be trace-1 and PSD).

    Returns
    -------
    rho_out : DensityMatrix
        Purified state ρ²/Tr(ρ²).  Trace-1 and PSD by construction.
    metrics : dict
        ``{"P_success": Tr(ρ²)}``
        P_success is the purity of the input, which equals the success
        probability of the corresponding projective protocol.
    """
    M = int(round(np.log2(rho.dim)))
    logger.debug(f"Rho2: M={M} qubit state (dim={rho.dim})")

    rho_sq = rho.data @ rho.data
    P_success = float(np.real(np.trace(rho_sq)))

    logger.debug(f"Rho2: Tr(ρ²) = P_success = {P_success:.6f}")

    if P_success < 1e-12:
        logger.warning(
            f"Rho2: Tr(ρ²) ≈ 0 (P={P_success:.2e}); returning maximally mixed state"
        )
        return (
            DensityMatrix(np.eye(2**M, dtype=complex) / 2**M),
            {"P_success": P_success},
        )

    rho_out = DensityMatrix(rho_sq / P_success)

    # Numerical sanity check
    out_trace = float(np.real(np.trace(rho_out.data)))
    if abs(out_trace - 1.0) > 1e-9:
        logger.warning(f"rho2: output trace = {out_trace:.10f} (should be 1.0)")

    logger.debug(f"Rho2 complete: P_success={P_success:.6f}")
    return rho_out, {"P_success": P_success}


# ─────────────────────────────────────────────────────────────────────────────
# Binary-tree merge interface  (called by streaming_runner)
# ─────────────────────────────────────────────────────────────────────────────

def purify_two_from_density(
    rho_A: DensityMatrix,
    rho_B: DensityMatrix,
    aa: Any,                      # AASpec — accepted for API parity, unused for rho2
) -> Tuple[DensityMatrix, Dict]:
    """rho2 merge of two density matrices: (ρ_A, ρ_B) → ρ_A²/Tr(ρ_A²).

    In the rho2 iterative protocol both inputs are always IDENTICAL clones of
    the same noisy state, so only rho_A is used.  The ``aa`` (amplitude
    amplification) argument is accepted for API compatibility with the SWAP
    runner but is silently ignored — rho2 is deterministic.

    Parameters
    ----------
    rho_A : DensityMatrix
        Left copy (used for the rho2 computation).
    rho_B : DensityMatrix
        Right copy (must be identical to rho_A; used for a consistency check
        only and otherwise ignored).
    aa : AASpec
        Amplitude-amplification spec — ignored by rho2.

    Returns
    -------
    rho_out : DensityMatrix
        ρ_A²/Tr(ρ_A²)
    metrics : dict
        {"P_success": Tr(ρ_A²)}
    """
    # Soft consistency check: warn if inputs differ (should never happen in the
    # iterative protocol, which always passes identical clones).
    if not np.allclose(rho_A.data, rho_B.data, atol=1e-10):
        logger.warning(
            "purify_two_from_density: rho_A and rho_B are not identical "
            "(max diff = %.2e).  Using rho_A only.",
            float(np.max(np.abs(rho_A.data - rho_B.data))),
        )

    return rho2_purification(rho_A)


__all__ = [
    "rho2_purification",
    "purify_two_from_density",
]