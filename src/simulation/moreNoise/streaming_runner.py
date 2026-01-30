"""
Streaming (O(log N)) purification runner using Qiskit circuits.

This module ties together:
  - target state preparation (state_factory.build_target),
  - noisy copy generation (noise_engine.build_noisy_copy), and
  - two-copy purification (amplified_swap.purify_two_from_density),
then logs per-merge metrics and a final summary compatible with your figure
scripts.

CRITICAL FIXES:
  1. In exact_k mode, pairs of copies at level 0 MUST share the same error
     pattern to ensure they are identical (required by SWAP test theory).
  2. All noise channels use explicit Kraus operators (no Qiskit DepolarizingChannel).
  3. Clifford twirling is applied automatically for dephasing noise types.
  4. For M=1, we log Bloch vector magnitude to verify renormalization theory.

Notes on scalability
--------------------
The purification merge operates on 1+2M qubits with a *density matrix* backend
(needed for CPTP channels). Density-matrix size grows as 4^{(1+2M)}; in
practice, M ≲ 5-6 is comfortable on a laptop. For larger M (e.g., 8-10), a
Monte Carlo trajectory mode is advisable (TODO: add in follow-up).
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from qiskit import transpile
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit.quantum_info.operators import Pauli

# Aer simulator import compatible across Qiskit versions
try:  # qiskit-aer >= 0.12
    from qiskit_aer import AerSimulator
except Exception:  # fallback for older installs
    from qiskit.providers.aer import AerSimulator  # type: ignore

from .configs import RunSpec, NoiseMode
from .state_factory import build_target
from .noise_engine import build_noisy_copy, sample_error_pattern, ErrorPattern, apply_noise_to_density_matrix
from .amplified_swap import purify_two_from_density

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# -----------------------------
# Backend helpers
# -----------------------------

def _make_backend(method: str):
    return AerSimulator(method=method)


def _density_from_circuit(qc, backend) -> DensityMatrix:
    """Execute circuit and return resulting density matrix."""
    qc2 = qc.copy()
    qc2.save_density_matrix()
    t = transpile(qc2, backend)
    res = backend.run(t, shots=1024).result()
    data0 = res.data(0)
    
    # 'density_matrix' key holds the complex array in Aer
    rho = data0.get("density_matrix")
    if rho is None:
        # older versions might use 'statevector' if method not honored
        rho = data0.get("statevector")
    
    dm = DensityMatrix(rho)
    logger.debug(f"Generated density matrix: dim={dm.dim}, purity={_purity(dm):.4f}")
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
    # Hermitian; trace norm is sum of absolute eigenvalues
    evals = np.linalg.eigvalsh((diff + diff.conj().T) / 2.0)
    return 0.5 * float(np.sum(np.abs(evals)))


def _purity(rho: DensityMatrix) -> float:
    """Tr(ρ²)"""
    return float(np.real(np.trace(rho.data @ rho.data)))


def _bloch_vector_magnitude(rho: DensityMatrix) -> Optional[float]:
    """For single-qubit states, compute |r⃗| where ρ = (I + r⃗·σ⃗)/2.
    
    Returns None if not a single-qubit state.
    """
    if rho.dim != 2:
        return None
    
    # Compute expectation values ⟨X⟩, ⟨Y⟩, ⟨Z⟩
    rx = np.real(np.trace(rho.data @ Pauli('X').to_matrix()))
    ry = np.real(np.trace(rho.data @ Pauli('Y').to_matrix()))
    rz = np.real(np.trace(rho.data @ Pauli('Z').to_matrix()))
    
    r_mag = float(np.sqrt(rx**2 + ry**2 + rz**2))
    return r_mag


# -----------------------------
# Iterative Purification Runner
# -----------------------------

def run_iterative_purification(spec: RunSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run iterative purification with proper iteration × purification level structure.
    
    This implements the correct protocol:
    1. Start from perfect target state  
    2. FOR each iteration:
       a. Apply noise to current state to get two noisy copies
       b. Apply ℓ levels of SWAP purification (using streaming logic)
       c. Result becomes new current state
    3. Repeat for specified number of iterations
    
    Key distinction:
    - ℓ (spec.purification_level): SWAP rounds per purification cycle
    - iterations: number of noise→purification cycles to repeat
    
    This models whether a fixed purification budget can keep up with 
    repeated environmental noise application.
    
    Parameters
    ----------
    spec : RunSpec
        Configuration with iterative_noise=True and purification_level=ℓ
        
    Returns
    -------
    (steps, finals) : Tuple[pd.DataFrame, pd.DataFrame]
        Steps include metrics from each iteration
    """
    spec.validate()
    out_dir: Path = spec.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if spec.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Calculate number of iterations and purification level
    purification_level = spec.purification_level  # ℓ SWAP rounds per cycle
    num_iterations = int(np.log2(spec.N)) if spec.N > 1 else 1  # Number of cycles
    
    logger.info(f"Starting ITERATIVE purification (ℓ×iterations): {spec.synthesize_run_id()}")
    logger.info(f"  M={spec.target.M}, purification_level=ℓ={purification_level}")
    logger.info(f"  iterations={num_iterations}, noise={spec.noise.noise_type.value}, p={spec.noise.p}")
    logger.info("  Using symmetric degradation within each iteration")
    
    # Only support iid_p mode for iterative
    if spec.noise.mode != NoiseMode.iid_p:
        raise ValueError("Iterative purification only supports iid_p noise mode")
    
    # Log twirling status
    twirling_active = spec._should_apply_twirling()
    if twirling_active:
        logger.info(f"  Clifford twirling ENABLED (mode={spec.twirling.mode})")
    else:
        logger.info("  Clifford twirling disabled")

    # Backend
    backend = _make_backend(spec.backend_method)

    # Target |ψ⟩ and its prep circuit
    prep, psi = build_target(spec.target)
    M = spec.target.M
    logger.info(f"Target state prepared: M={M}, dim={psi.dim}")

    # Start with perfect target state as current state
    current_state = DensityMatrix(psi)
    
    # Baseline metrics
    F_init = _fidelity_to_pure(current_state, psi)
    eps_init = _trace_distance_to_pure(current_state, psi)
    pur_init = _purity(current_state)
    
    logger.info(f"Initial perfect state: F={F_init:.6f}, ε_L={eps_init:.6f}, purity={pur_init:.6f}")
    
    if M == 1:
        r_init = _bloch_vector_magnitude(current_state)
        logger.info(f"  Initial Bloch vector magnitude: |r⃗|={r_init:.6f}")

    steps_rows: List[Dict] = []

    # OUTER LOOP: Iterate the noise→purification cycles
    for iter_idx in range(num_iterations):
        logger.info(f"=== Iteration {iter_idx + 1}/{num_iterations} ===")
        
        # Step 1: Apply noise to current state (symmetric degradation)
        twirl_seed_1 = (spec.target.seed or 0) + iter_idx * 1000  
        twirl_seed_2 = (spec.target.seed or 0) + iter_idx * 1000 + 500
        
        current_noisy_1 = apply_noise_to_density_matrix(
            current_state, 
            spec.noise, 
            twirling=spec.twirling if twirling_active else None,
            twirl_seed=twirl_seed_1
        )
        
        current_noisy_2 = apply_noise_to_density_matrix(
            current_state, 
            spec.noise, 
            twirling=spec.twirling if twirling_active else None,
            twirl_seed=twirl_seed_2
        )
        
        F_before = _fidelity_to_pure(current_state, psi)
        F_noisy_1 = _fidelity_to_pure(current_noisy_1, psi) 
        F_noisy_2 = _fidelity_to_pure(current_noisy_2, psi)
        logger.info(f"  Before noise: F={F_before:.6f}")
        logger.info(f"  After noise: F₁={F_noisy_1:.6f}, F₂={F_noisy_2:.6f}")
        
        # Step 2: Apply ℓ levels of SWAP purification using streaming logic
        # We need to simulate the streaming purification on these two noisy copies
        
        if purification_level == 0:
            # No purification - just take one of the noisy copies
            iteration_result = current_noisy_1
            total_success_prob = 1.0
            
        else:
            # Apply ℓ levels of purification
            # Start with the two noisy copies as our initial "pool"
            left_state = current_noisy_1
            right_state = current_noisy_2
            total_success_prob = 1.0
            
            # Apply successive SWAP purification rounds
            for level in range(purification_level):
                logger.debug(f"    SWAP level {level + 1}/{purification_level}")
                
                # Purify the current pair
                purified_state, meta = purify_two_from_density(
                    left_state, 
                    right_state, 
                    spec.aa
                )
                
                success_prob = meta.get("P_success", 0)
                total_success_prob *= success_prob
                grover_iters = meta.get("grover_iters", 0)
                
                F_after_level = _fidelity_to_pure(purified_state, psi)
                logger.debug(f"      After level {level + 1}: F={F_after_level:.6f}, P_succ={success_prob:.4f}")
                
                if level < purification_level - 1:
                    # For next level, we need two copies of the current purified state
                    # Apply fresh noise to create the pair for the next level
                    next_seed_1 = twirl_seed_1 + (level + 1) * 100
                    next_seed_2 = twirl_seed_2 + (level + 1) * 100
                    
                    left_state = apply_noise_to_density_matrix(
                        purified_state,
                        spec.noise,
                        twirling=spec.twirling if twirling_active else None,
                        twirl_seed=next_seed_1
                    )
                    
                    right_state = apply_noise_to_density_matrix(
                        purified_state,
                        spec.noise,
                        twirling=spec.twirling if twirling_active else None,
                        twirl_seed=next_seed_2
                    )
                else:
                    # Final level - this is our result
                    iteration_result = purified_state
        
        # Update current state for next iteration
        current_state = iteration_result
        
        # Calculate metrics for this iteration
        fid = _fidelity_to_pure(current_state, psi)
        eps = _trace_distance_to_pure(current_state, psi)
        pur = _purity(current_state)
        
        # Record step metrics  
        row = {
            "run_id": spec.synthesize_run_id(),
            "merge_num": iter_idx,
            "M": M,
            "depth": (iter_idx + 1) * purification_level,  # Total SWAP rounds so far
            "copies_used": 2 * (iter_idx + 1) * (2**purification_level - 1),  # Rough estimate
            "N_so_far": 2 * (iter_idx + 1),
            "noise": spec.noise.noise_type.value,
            "mode": spec.noise.mode.value,
            "p": spec.noise.p,
            "p_channel": spec.noise.kraus_p(),
            "P_success": total_success_prob,
            "grover_iters": 0,  # Aggregate measure not meaningful
            "twirling_applied": twirling_active,
            "fidelity": fid,
            "eps_L": eps,
            "purity": pur,
            "bloch_r": _bloch_vector_magnitude(current_state) if M == 1 else None,
            "iteration": iter_idx + 1,
            "purification_level": purification_level,
        }
        
        logger.info(f"  Iteration {iter_idx + 1} complete: F={fid:.6f}, ε_L={eps:.6f}, "
                   f"P_succ_total={total_success_prob:.4f}")
        
        steps_rows.append(row)

    # Final metrics
    F_final = _fidelity_to_pure(current_state, psi)
    eps_final = _trace_distance_to_pure(current_state, psi)
    pur_final = _purity(current_state)
    reduction = eps_final / eps_init if eps_init > 0 else np.nan

    logger.info(f"Iterative purification complete: F={F_final:.6f}, ε_L={eps_final:.6f}, "
                f"reduction={reduction:.6f}")

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
        "twirling_enabled": twirling_active,
        "iterations": num_iterations,
        "purification_level": purification_level,
    }
    
    if M == 1:
        finals_row["bloch_r_init"] = _bloch_vector_magnitude(DensityMatrix(psi))
        finals_row["bloch_r_final"] = _bloch_vector_magnitude(current_state)

    steps_df = pd.DataFrame(steps_rows)
    finals_df = pd.DataFrame([finals_row])
    
    logger.info(f"Iterative run complete: {num_iterations} iterations × {purification_level} purification levels")
    
    return steps_df, finals_df


# -----------------------------
# Regular Streaming Runner
# -----------------------------

def run_streaming(spec: RunSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run a streaming purification experiment per 'spec'.
    
    Supports both regular streaming mode and iterative noise mode based on
    spec.iterative_noise flag.

    Returns
    -------
    (steps, finals):
        steps  — one row per merge (purification step), with depth, metrics, etc.
        finals — one row summary for the run (final eps_L, reduction ratio, ...).
    """
    # Check if iterative mode is requested
    if spec.iterative_noise:
        logger.info("Using ITERATIVE noise mode")
        return run_iterative_purification(spec)
    else:
        logger.info("Using REGULAR streaming mode")
        return run_regular_streaming(spec)


def run_regular_streaming(spec: RunSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the original streaming purification experiment per 'spec'.
    
    This is the original logic moved to a separate function to support
    the new iterative mode.

    Returns
    -------
    (steps, finals):
        steps  — one row per merge (purification step), with depth, metrics, etc.
        finals — one row summary for the run (final eps_L, reduction ratio, ...).
    """
    spec.validate()
    out_dir: Path = spec.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if spec.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Starting streaming run: {spec.synthesize_run_id()}")
    logger.info(f"  M={spec.target.M}, N={spec.N}, noise={spec.noise.noise_type.value}, "
                f"mode={spec.noise.mode.value}, p={spec.noise.p}")
    
    # Log twirling status
    twirling_active = spec._should_apply_twirling()
    if twirling_active:
        logger.info(f"  Clifford twirling ENABLED (mode={spec.twirling.mode})")
    else:
        logger.info("  Clifford twirling disabled")

    # Backend
    backend = _make_backend(spec.backend_method)

    # Target |ψ⟩ and its prep circuit
    prep, psi = build_target(spec.target)
    M = spec.target.M
    logger.info(f"Target state prepared: M={M}, dim={psi.dim}")

    # Build a prototype noisy copy for iid_p mode (reusable when no twirling)
    iid_cached_dm: Optional[DensityMatrix] = None
    
    if spec.noise.mode == NoiseMode.iid_p and not twirling_active:
        # Can cache when no twirling - all copies are identical
        # No twirling: every iid_p copy is identical → safe to cache one density matrix
        logger.info("Building cached iid_p noisy copy (no twirling)")
        qc_noisy, _ = build_noisy_copy(prep, spec.noise, seed=spec.target.seed)
        iid_cached_dm = _density_from_circuit(qc_noisy, backend)
    else:
        # Either exact_k mode, or iid_p with twirling → must build copies individually
        logger.info("Cannot cache noisy copies (either exact_k mode or twirling enabled)")

    # Initial (noisy) single-register density for baseline metrics 
    if iid_cached_dm is not None:
        rho_init = iid_cached_dm
    else:
        # Build one sample copy for baseline
        logger.info("Building sample noisy copy for baseline metrics")
        if twirling_active and spec.noise.mode == NoiseMode.iid_p:
            # Pass twirling config for baseline
            # Baseline with twirling: use a fixed twirl_seed just for the first copy
            qc_copy, _ = build_noisy_copy(
                prep, 
                spec.noise, 
                twirling=spec.twirling,
                twirl_seed=spec.target.seed,
            )
        else:
            # exact_k or iid_p without twirling
            qc_copy, _ = build_noisy_copy(prep, spec.noise, seed=spec.target.seed)
        rho_init = _density_from_circuit(qc_copy, backend)
                

    # Baseline metrics before any purification
    F_init = _fidelity_to_pure(rho_init, psi)
    eps_init = _trace_distance_to_pure(rho_init, psi)
    pur_init = _purity(rho_init)
    
    logger.info(f"Initial state metrics: F={F_init:.6f}, ε_L={eps_init:.6f}, purity={pur_init:.6f}")
    
    if M == 1:
        r_init = _bloch_vector_magnitude(rho_init)
        logger.info(f"  Initial Bloch vector magnitude: |r⃗|={r_init:.6f}")

    # Memory stack for streaming purification
    slots: Dict[int, DensityMatrix] = {}  # level -> density matrix
    counts: Dict[int, int] = {}  # level -> number of input copies represented
    steps_rows: List[Dict] = []
    level0_patterns: Dict[int, ErrorPattern] = {}  # For exact_k mode tracking
    merge_counter = 0
        
        
    def _log_step(depth: int, rho_out: DensityMatrix, meta: Dict, inputs_used: int) -> None:
        nonlocal merge_counter, carry_count
        merge_counter += 1
        
        fid = _fidelity_to_pure(rho_out, psi)
        eps = _trace_distance_to_pure(rho_out, psi)
        pur = _purity(rho_out)
        
        # Calculate N_so_far (total copies processed so far)
        # Total copies represented in all slots + current carry
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
            "p": spec.noise.p,  # Using p instead of delta
            "p_channel": spec.noise.kraus_p(),
            "P_success": meta.get("P_success", 0.0),
            "grover_iters": meta.get("grover_iters", 0),
            "twirling_applied": twirling_active,
            "fidelity": fid,
            "eps_L": eps,
            "purity": pur,
        }
        
        # For M=1, add Bloch vector magnitude for theory verification
        if M == 1:
            r_mag = _bloch_vector_magnitude(rho_out)
            row["bloch_r"] = r_mag
            logger.debug(f"  Depth {depth}: |r⃗|={r_mag:.6f}, F={fid:.6f}, ε_L={eps:.6f}")
        else:
            row["bloch_r"] = None  # Ensure column exists for all rows
        
        logger.info(f"Merge {merge_counter} (depth {depth}): F={fid:.6f}, ε_L={eps:.6f}, "
                   f"P_succ={meta.get('P_success', 0):.4f}, twirl={twirling_active}")
        
        steps_rows.append(row)
       
        
    def _get_new_copy_dm(i: int) -> DensityMatrix:
        """Generate the i-th noisy copy."""
        
        if iid_cached_dm is not None:
            # Use cached copy (no twirling case)
            # iid_p, no twirl → reuse cached density matrix
            logger.debug(f"Copy {i}: Using cached iid_p density matrix")
            return iid_cached_dm
        
        if spec.noise.mode == NoiseMode.iid_p:
            # iid_p mode with twirling: generate each copy independently
            logger.debug(f"Copy {i}: Generating iid_p copy with twirling")
            twirl_seed = (spec.target.seed or 0) + i * 1000  # Different seed per copy
            
            qc_copy, _ = build_noisy_copy(
                prep, 
                spec.noise, 
                twirling=spec.twirling,
                twirl_seed=twirl_seed,
            )
            return _density_from_circuit(qc_copy, backend)
        
        #  # exact_k mode (no twirling implemented here)
        # logger.debug(f"Copy {i}: Generating new exact_k noisy copy")
        # qc_copy, pattern = build_noisy_copy(
        #     prep,
        #     spec.noise,
        #     seed=(spec.target.seed or 0) + i,
        # )

        # if pattern is not None:
        #     level0_patterns[i] = pattern

        # return _density_from_circuit(qc_copy, backend)
    
        else:
            # exact_k mode: sample new pattern per copy
            logger.debug(f"Copy {i}: Generating new exact_k noisy copy")
            qc_copy, pattern = build_noisy_copy(
                prep, 
                spec.noise, 
                seed=(spec.target.seed or 0) + i
            )
            
            if pattern is not None:
                level0_patterns[i] = pattern
                
            return _density_from_circuit(qc_copy, backend)

    # Process N incoming copies
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
                # Empty slot at this level; store and break
                slots[level] = carry_dm
                counts[level] = carry_count
                logger.debug(f"  Copy {i}: Stored at level {level}")
                break
            else:
                # Merge with existing slot at this level
                left = slots.pop(level)
                left_count = counts.pop(level)
                
                logger.debug(f"  Merging at level {level}: {left_count} + {carry_count} copies")
                
                # CRITICAL FIX: For exact_k mode at level 0, ensure identical copies
                # by using a shared error pattern
                if spec.noise.mode == NoiseMode.exact_k and level == 0:
                    # Generate a fresh shared pattern for this pair
                    shared_seed = (spec.target.seed or 0) + i * 1000 + level
                    shared_pattern = sample_error_pattern(
                        M=M,
                        noise_type=spec.noise.noise_type,
                        k=spec.noise.exact_k,
                        seed=shared_seed,
                    )
                    
                    logger.debug(f"  Level 0 merge: Creating identical pair with shared pattern")
                    
                    # Rebuild both copies with the SAME pattern
                    qc_left, _ = build_noisy_copy(prep, spec.noise, shared_pattern=shared_pattern)
                    qc_right, _ = build_noisy_copy(prep, spec.noise, shared_pattern=shared_pattern)
                    
                    left = _density_from_circuit(qc_left, backend)
                    carry_dm = _density_from_circuit(qc_right, backend)
                    
                    logger.info(f"  Rebuilt identical pair for level 0 merge (pattern: {len(shared_pattern)} errors)")
                
                # Purify two identical copies
                rho_out, meta = purify_two_from_density(
                    left, 
                    carry_dm, 
                    spec.aa
                )
                
                _log_step(
                    depth=level + 1, 
                    rho_out=rho_out, 
                    meta=meta,
                    inputs_used=left_count + carry_count
                )
                
                carry_dm = rho_out
                carry_count = left_count + carry_count
                level += 1

    # Final output = deepest slot
    if not counts:
        raise ValueError("No data was processed; N must be >= 1")
    
    max_level = max(counts.keys())
    rho_final = slots[max_level]

    F_final = _fidelity_to_pure(rho_final, psi)
    eps_final = _trace_distance_to_pure(rho_final, psi)
    pur_final = _purity(rho_final)
    reduction = eps_final / eps_init if eps_init > 0 else np.nan

    logger.info(f"Final state (max depth {max_level}): F={F_final:.6f}, ε_L={eps_final:.6f}, "
                f"reduction={reduction:.6f}")

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

    steps_df = pd.DataFrame(steps_rows)
    finals_df = pd.DataFrame([finals_row])
    
    logger.info(f"Run complete: {len(steps_rows)} merges performed")
    
    return steps_df, finals_df


def run_and_save(spec: RunSpec) -> Tuple[Path, Path]:
    """Run a single spec and append results to CSVs under spec.out_dir.

    Returns the (steps_path, finals_path).
    """
    steps_df, finals_df = run_streaming(spec)
    out_dir = spec.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use dynamic filenames based on run_id like the old version
    # steps_path = out_dir / "steps_circuit_dephasing.csv"
    # finals_path = out_dir / "finals_circuit_dephasing.csv"
    steps_path = out_dir / "steps_circuit_depolarizing.csv"
    finals_path = out_dir / "finals_circuit_depolarizing.csv"

    # Append or create
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