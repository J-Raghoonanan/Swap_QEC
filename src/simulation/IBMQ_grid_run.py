"""
IBM Quantum grid run for SWAP-based purification error correction.

This script runs a parameter sweep over system sizes M and error rates p,
executing the purification protocol on IBM quantum hardware and saving
results in a format compatible with the simulation data.

Usage:
    python -m src.simulation.IBMQ_grid_run --backend ibm_torino --shots 1024
    python -m src.simulation.IBMQ_grid_run --backend ibm_torino --quick --verbose
"""
import argparse
import logging
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any

# Local imports
from .IBMQ_components import (
    IBMQRunSpec,
    setup_ibm_backend,
    build_full_purification_experiment,
    transpile_for_backend,
    run_circuit_with_retries,
    save_ibmq_results,
    calculate_fidelity_from_counts,
    analyze_sequential_results,
)
from .configs import NoiseSpec, NoiseType, NoiseMode

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# =============================
# Experimental Parameters
# =============================

# Grid parameters
M_LIST: List[int] = [1, 2]  # System sizes  
N_LIST: List[int] = [2, 4, 8, 16]  # Number of input copies
P_LIST: List[float] = [0.01, 0.1, 0.2, 0.3]  # Error rates
NOISE_TYPES: List[NoiseType] = [NoiseType.depolarizing, NoiseType.dephase_z]
TARGET_TYPES: List[str] = ["hadamard"]  # Start with Hadamard only

# Hardware settings
DEFAULT_BACKEND = "ibm_torino"
DEFAULT_SHOTS = 8192
DEFAULT_OPTIMIZATION = 2


def run_single_experiment(run_spec: IBMQRunSpec, service, backend) -> Dict[str, Any]:
    """
    Run a single purification experiment.
    
    Args:
        run_spec: Experiment configuration
        backend: IBM quantum backend
        
    Returns:
        Dictionary of experimental results
    """
    logger.info(f"Starting experiment: M={run_spec.M}, N={run_spec.N}, p={run_spec.noise.p:.3f}, "
                f"noise={run_spec.noise.noise_type.value}, target={run_spec.target_type}")
    
    try:
        # Step 1: Build the experimental circuit
        qc = build_full_purification_experiment(
            M=run_spec.M,
            noise_spec=run_spec.noise,
            target_type=run_spec.target_type,
            N=run_spec.N
        )
        
        logger.info(f"Built circuit: {qc.num_qubits} qubits, depth {qc.depth()}")
        
        # Step 2: Transpile for backend
        transpiled_qc = transpile_for_backend(qc, backend, run_spec.transpilation_level)
        logger.info(f"Transpiled depth: {transpiled_qc.depth()}")
        
        # Step 3: Execute on hardware
        result = run_circuit_with_retries(transpiled_qc, service, backend, run_spec.shots)
        
        # Step 4: Analyze results
        counts = result.get_counts()
        logger.info(f"Total measurement outcomes: {len(counts)}")
        
        if run_spec.N == 2:
            # Single SWAP test analysis
            results = _analyze_single_swap_results(counts, run_spec)
        else:
            # Sequential purification analysis  
            results = analyze_sequential_results(counts, run_spec.N, run_spec.M, run_spec.target_type)
            
            # Add circuit info
            results.update({
                'total_shots': sum(counts.values()),
                'circuit_depth': qc.depth(),
                'transpiled_depth': transpiled_qc.depth(),
                'all_counts': dict(counts),
            })
        
        logger.info(f"Results: fidelity={results.get('fidelity', -1):.4f}, "
                   f"success_prob={results.get('success_probability', -1):.4f}")
        return results
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return error results
        return {
            'fidelity': -1.0,  # Indicate failure
            'success_probability': -1.0,
            'error': str(e),
            'total_shots': 0,
            'success_shots': 0,
            'circuit_depth': -1,
            'transpiled_depth': -1,
        }


def _analyze_single_swap_results(counts: Dict[str, int], run_spec: IBMQRunSpec) -> Dict[str, Any]:
    """
    Analyze results from single SWAP test (N=2).
    
    Args:
        counts: Measurement outcome dictionary
        run_spec: Run specification
        
    Returns:
        Analysis results dictionary
    """
    total_shots = sum(counts.values())
    if total_shots == 0:
        return {'fidelity': 0.0, 'success_probability': 0.0}
    
    # For N=2: measurement format is "fidelity_bits ancilla_bit"
    M = run_spec.M
    ancilla_counts = {}
    fidelity_counts = {}
    
    for outcome, count in counts.items():
        if len(outcome) != M + 1:  # M fidelity bits + 1 ancilla bit
            continue
            
        # ancilla_bit = outcome[0]  # First bit is ancilla
        # fidelity_bits = outcome[1:]  # Rest are fidelity measurement
        
        ############################################################### NOT SURE WHICH IS CORRECT
        ancilla_bit = outcome[-1] # Last char = ancilla register
        fidelity_bits = outcome[:-1]  # First M chars = fidelity registers
        
        # Count ancilla outcomes
        if ancilla_bit not in ancilla_counts:
            ancilla_counts[ancilla_bit] = 0
        ancilla_counts[ancilla_bit] += count
        
        # Count fidelity outcomes (conditioned on ancilla = 0 for successful SWAP)
        if ancilla_bit == '0':
            if fidelity_bits not in fidelity_counts:
                fidelity_counts[fidelity_bits] = 0
            fidelity_counts[fidelity_bits] += count
    
    # Calculate metrics
    success_count = ancilla_counts.get('0', 0)
    success_probability = success_count / total_shots
    
    # Calculate fidelity (conditioned on successful SWAP)
    if fidelity_counts:
        fidelity = calculate_fidelity_from_counts(
            fidelity_counts, M, run_spec.target_type
        )
    else:
        fidelity = 0.0
    
    return {
        'fidelity': fidelity,
        'success_probability': success_probability,
        'total_shots': total_shots,
        'success_shots': success_count,
        'ancilla_counts': ancilla_counts,
        'fidelity_counts': dict(fidelity_counts),
        'all_counts': dict(counts),
    }
    
   
    
   


def run_parameter_grid(backend_name: str, shots: int, out_dir: Path, 
                      quick_test: bool = False, target_types: List[str] = None) -> None:
    """
    Run the full parameter grid sweep.
    
    Args:
        backend_name: IBM backend name
        shots: Number of measurement shots per experiment
        out_dir: Output directory for results
        quick_test: If True, run reduced parameter space for testing
        target_types: Target state types to test
    """
    logger.info("="*70)
    logger.info("Starting IBMQ SWAP Purification Grid Run")
    logger.info("="*70)
    
    # Setup
    if target_types is None:
        target_types = TARGET_TYPES
        
    if quick_test:
        logger.info("QUICK TEST MODE: Using reduced parameter space")
        M_list = [1, 2]
        N_list = [2, 4]  # Include N in quick test
        p_list = [0.1, 0.2]
        noise_types = [NoiseType.depolarizing]
    else:
        M_list = M_LIST
        N_list = N_LIST  # Include full N range
        p_list = P_LIST  
        noise_types = NOISE_TYPES
    
    # Connect to backend
    logger.info(f"Connecting to backend: {backend_name}")
    service, backend = setup_ibm_backend(backend_name)
    logger.info(f"Backend status: {backend.status()}")
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / f"ibmq_results_{backend_name}.csv"
    
    # Calculate total experiments
    total_experiments = len(M_list) * len(N_list) * len(p_list) * len(noise_types) * len(target_types)
    logger.info(f"Total experiments planned: {total_experiments}")
    logger.info(f"Results will be saved to: {results_file}")
    
    # Run experiments
    experiment_count = 0
    start_time = time.time()
    
    for target_type in target_types:
        for noise_type in noise_types:
            for M in M_list:
                for N in N_list:
                    # Skip invalid combinations where total qubits exceed reasonable limits
                    max_ancilla = N // 2 if N > 2 else 1
                    total_qubits = N * M + max_ancilla
                    if total_qubits > 100:  # Conservative limit
                        logger.warning(f"Skipping M={M}, N={N}: too many qubits ({total_qubits})")
                        continue
                        
                    for p in p_list:
                        experiment_count += 1
                        
                        logger.info(f"\n{'='*50}")
                        logger.info(f"Experiment {experiment_count}/{total_experiments}")
                        logger.info(f"M={M}, N={N}, p={p:.3f}, noise={noise_type.value}, target={target_type}")
                        logger.info(f"{'='*50}")
                        
                        # Create run specification
                        noise_spec = NoiseSpec(
                            noise_type=noise_type,
                            mode=NoiseMode.iid_p,
                            p=p
                        )
                        
                        run_spec = IBMQRunSpec(
                            M=M,
                            N=N,
                            noise=noise_spec,
                            target_type=target_type,
                            backend_name=backend_name,
                            shots=shots,
                            out_dir=out_dir,
                            run_id=f"M{M}_N{N}_p{p:.3f}_{noise_type.value}_{target_type}"
                        )
                    
                    # Validate configuration
                    try:
                        run_spec.validate()
                    except ValueError as e:
                        logger.error(f"Invalid configuration: {e}")
                        continue
                    
                    # Run experiment
                    exp_start = time.time()
                    results = run_single_experiment(run_spec, service, backend)
                    exp_duration = time.time() - exp_start
                    
                    # Save results
                    try:
                        save_ibmq_results(results, results_file, run_spec)
                        logger.info(f"✓ Experiment completed in {exp_duration:.1f}s")
                    except Exception as e:
                        logger.error(f"✗ Failed to save results: {e}")
                    
                    # Progress update
                    elapsed = time.time() - start_time
                    avg_time = elapsed / experiment_count
                    remaining = (total_experiments - experiment_count) * avg_time
                    
                    logger.info(f"Progress: {experiment_count}/{total_experiments} "
                              f"({100*experiment_count/total_experiments:.1f}%)")
                    logger.info(f"Elapsed: {elapsed/60:.1f}min, "
                              f"Estimated remaining: {remaining/60:.1f}min")
    
    # Final summary
    total_time = time.time() - start_time
    logger.info("\n" + "="*70)
    logger.info("Grid run completed!")
    logger.info(f"Total experiments: {experiment_count}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Average per experiment: {total_time/experiment_count:.1f}s")
    logger.info(f"Results saved to: {results_file}")
    logger.info("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="IBM Quantum SWAP purification grid run")
    
    parser.add_argument(
        "--backend", 
        type=str, 
        default=DEFAULT_BACKEND,
        help=f"IBM Quantum backend name (default: {DEFAULT_BACKEND})"
    )
    
    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
        help=f"Number of measurement shots (default: {DEFAULT_SHOTS})"
    )
    
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/IBMQ"),
        help="Output directory for results (default: data/IBMQ)"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test with reduced parameter space"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable debug-level logging"
    )
    
    parser.add_argument(
        "--noise-types",
        nargs="+",
        choices=["depolarizing", "dephasing_z", "dephasing_x"],
        default=["depolarizing", "dephasing_z"],
        help="Noise types to test"
    )
    
    parser.add_argument(
        "--target-types",
        nargs="+", 
        choices=["hadamard", "ghz"],
        default=["hadamard"],
        help="Target state types to test"
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Map noise type strings to enums
    noise_type_map = {
        "depolarizing": NoiseType.depolarizing,
        "dephasing_z": NoiseType.dephase_z,
        "dephasing_x": NoiseType.dephase_x,
    }
    
    selected_noise_types = [noise_type_map[nt] for nt in args.noise_types]
    
    # Update global noise types
    global NOISE_TYPES
    NOISE_TYPES = selected_noise_types
    
    logger.info(f"Configuration:")
    logger.info(f"  Backend: {args.backend}")
    logger.info(f"  Shots: {args.shots}")
    logger.info(f"  Output: {args.out_dir}")
    logger.info(f"  Quick test: {args.quick}")
    logger.info(f"  Noise types: {[nt.value for nt in selected_noise_types]}")
    logger.info(f"  Target types: {args.target_types}")
    
    try:
        run_parameter_grid(
            backend_name=args.backend,
            shots=args.shots,
            out_dir=args.out_dir,
            quick_test=args.quick,
            target_types=args.target_types
        )
        
    except KeyboardInterrupt:
        logger.info("\nGrid run interrupted by user")
    except Exception as e:
        logger.error(f"Grid run failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())