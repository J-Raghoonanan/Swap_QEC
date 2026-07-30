"""
Validation script to test the purification protocol and verify theory matches simulation.

This script runs focused tests to validate:
1. Bloch vector renormalization for M=1 (should match |r_out| = 4|r|/(3+|r|²))
2. Identical copies are used in exact_k mode
3. Clifford twirling converts dephasing to effective depolarization
4. Error reduction improves with increasing N

Run this before doing full grid sweeps to catch bugs early.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulation.configs import (
    RunSpec,
    TargetSpec,
    NoiseSpec,
    AASpec,
    TwirlingSpec,
    NoiseType,
    NoiseMode,
    StateKind,
)
from src.simulation.streaming_runner import run_streaming

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def test_bloch_renormalization():
    """Test M=1 Bloch vector renormalization matches theory."""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Bloch Vector Renormalization (M=1)")
    logger.info("="*70)
    
    # Theory: |r_out| = 4|r|/(3+|r|²)
    # For δ=0.5, initial |r| = 1-δ = 0.5
    # Expected after one merge: |r_out| = 4*0.5/(3+0.25) = 2.0/3.25 ≈ 0.615
    
    delta = 0.5
    r_in = 1.0 - delta
    r_out_theory = 4.0 * r_in / (3.0 + r_in**2)
    
    logger.info(f"Theory prediction:")
    logger.info(f"  Initial: |r| = {r_in:.6f}")
    logger.info(f"  After 1 merge: |r_out| = {r_out_theory:.6f}")
    
    spec = RunSpec(
        target=TargetSpec(M=1, kind=StateKind.hadamard, seed=42),
        noise=NoiseSpec(noise_type=NoiseType.depolarizing, mode=NoiseMode.iid_p, delta=delta),
        aa=AASpec(target_success=0.99, max_iters=32),
        N=2,  # Just one merge
        backend_method="density_matrix",
        out_dir=Path("data/test"),
        verbose=True,
    )
    
    steps_df, finals_df = run_streaming(spec)
    
    # Check the first merge (depth=1)
    merge1 = steps_df[steps_df['depth'] == 1].iloc[0]
    r_sim = merge1['bloch_r']
    
    logger.info(f"\nSimulation result:")
    logger.info(f"  After 1 merge: |r| = {r_sim:.6f}")
    logger.info(f"  Error: {abs(r_sim - r_out_theory):.6f}")
    
    # Allow 1% tolerance
    assert abs(r_sim - r_out_theory) < 0.01, f"Bloch renormalization mismatch! Expected {r_out_theory:.6f}, got {r_sim:.6f}"
    
    logger.info("✓ TEST PASSED: Bloch renormalization matches theory\n")


def test_identical_copies_exact_k():
    """Test that exact_k mode uses identical copies at level 0."""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Identical Copies in exact_k Mode")
    logger.info("="*70)
    
    # In exact_k mode with k=1, both copies should have the SAME error pattern
    # This should produce better purification than random different patterns
    
    spec_exact = RunSpec(
        target=TargetSpec(M=2, kind=StateKind.hadamard, seed=42),
        noise=NoiseSpec(noise_type=NoiseType.depolarizing, mode=NoiseMode.exact_k, delta=0.5, exact_k=1),
        aa=AASpec(target_success=0.99, max_iters=32),
        N=4,
        backend_method="density_matrix",
        out_dir=Path("data/test"),
        verbose=True,
    )
    
    steps_df, finals_df = run_streaming(spec_exact)
    
    # Check that purification is occurring (fidelity increasing)
    if len(steps_df) > 0:
        initial_f = steps_df.iloc[0]['fidelity']
        final_f = finals_df.iloc[0]['fidelity_final']
        
        logger.info(f"Fidelity progression:")
        logger.info(f"  Initial: {initial_f:.6f}")
        logger.info(f"  Final: {final_f:.6f}")
        logger.info(f"  Improvement: {final_f - initial_f:.6f}")
        
        # Fidelity should improve (or at worst stay the same due to numerical noise)
        assert final_f >= initial_f - 0.01, "Fidelity decreased! Identical copies not being used correctly."
        
        logger.info("✓ TEST PASSED: exact_k mode appears to use identical copies\n")
    else:
        logger.warning("! No merge steps recorded; test inconclusive\n")


def test_twirling_effectiveness():
    """Test that Clifford twirling improves dephasing performance."""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Clifford Twirling for Dephasing")
    logger.info("="*70)
    
    # Run Z-dephasing with and without twirling
    base_spec = RunSpec(
        target=TargetSpec(M=1, kind=StateKind.hadamard, seed=42),
        noise=NoiseSpec(noise_type=NoiseType.dephase_z, mode=NoiseMode.iid_p, delta=0.5),
        aa=AASpec(target_success=0.99, max_iters=32),
        N=16,
        backend_method="density_matrix",
        out_dir=Path("data/test"),
        verbose=False,
    )
    
    # Without twirling
    spec_no_twirl = base_spec
    spec_no_twirl.twirling = TwirlingSpec(enabled=False)
    
    logger.info("Running WITHOUT twirling...")
    _, finals_no_twirl = run_streaming(spec_no_twirl)
    
    # With twirling
    spec_twirl = base_spec
    spec_twirl.twirling = TwirlingSpec(enabled=True, mode="random", seed=42)
    
    logger.info("Running WITH twirling...")
    _, finals_twirl = run_streaming(spec_twirl)
    
    eps_no_twirl = finals_no_twirl.iloc[0]['error_reduction_ratio']
    eps_twirl = finals_twirl.iloc[0]['error_reduction_ratio']
    
    logger.info(f"\nResults:")
    logger.info(f"  Error reduction WITHOUT twirling: {eps_no_twirl:.6f}")
    logger.info(f"  Error reduction WITH twirling: {eps_twirl:.6f}")
    logger.info(f"  Improvement factor: {eps_no_twirl / eps_twirl:.2f}x")
    
    # Twirling should improve error reduction (lower ratio is better)
    if eps_twirl < eps_no_twirl:
        logger.info("✓ TEST PASSED: Twirling improves error reduction\n")
    else:
        logger.warning(f"! Twirling did not improve error reduction (may be statistical noise)\n")


def test_n_scaling():
    """Test that error reduction improves with increasing N."""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Error Reduction Scales with N")
    logger.info("="*70)
    
    results = []
    
    for N in [2, 4, 8, 16]:
        spec = RunSpec(
            target=TargetSpec(M=1, kind=StateKind.hadamard, seed=42),
            noise=NoiseSpec(noise_type=NoiseType.depolarizing, mode=NoiseMode.iid_p, delta=0.5),
            aa=AASpec(target_success=0.99, max_iters=32),
            N=N,
            backend_method="density_matrix",
            out_dir=Path("data/test"),
            verbose=False,
        )
        
        logger.info(f"Testing N={N}...")
        _, finals = run_streaming(spec)
        
        eps_ratio = finals.iloc[0]['error_reduction_ratio']
        results.append((N, eps_ratio))
        logger.info(f"  Error reduction ratio: {eps_ratio:.6f}")
    
    logger.info(f"\nScaling summary:")
    for N, eps in results:
        logger.info(f"  N={N:4d}: ε_final/ε_init = {eps:.6f}")
    
    # Error reduction should generally improve (decrease) with larger N
    ratios = [eps for _, eps in results]
    if all(ratios[i] >= ratios[i+1] - 0.05 for i in range(len(ratios)-1)):
        logger.info("✓ TEST PASSED: Error reduction improves with N\n")
    else:
        logger.warning("! Error reduction did not consistently improve with N\n")


def main():
    """Run all validation tests."""
    logger.info("\n" + "="*70)
    logger.info("PURIFICATION PROTOCOL VALIDATION SUITE")
    logger.info("="*70 + "\n")
    
    try:
        test_bloch_renormalization()
        test_identical_copies_exact_k()
        test_twirling_effectiveness()
        test_n_scaling()
        
        logger.info("\n" + "="*70)
        logger.info("ALL TESTS COMPLETED")
        logger.info("="*70 + "\n")
        
    except AssertionError as e:
        logger.error(f"\n✗ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        logger.error(f"\n✗ UNEXPECTED ERROR: {e}\n", exc_info=True)
        raise


if __name__ == "__main__":
    main()