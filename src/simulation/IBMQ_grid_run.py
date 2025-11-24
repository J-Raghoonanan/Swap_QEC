"""
Optimized grid run for SWAP purification experiments on IBM Quantum hardware.
Targeted for specific parameter ranges: M=1,2; p=0.01,0.1,0.2,0.3; N=2,4,8

This script generates exactly the data needed to validate the theoretical predictions
from the SWAP purification paper against real quantum hardware.
"""
import argparse
import csv
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Import our SWAP purification implementation (fixed version)
from .IBMQ_components import (
    PurificationConfig,
    run_complete_purification_experiment,
    setup_ibm_backend,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Target Parameter Ranges (Based on Paper Requirements)
# =============================================================================

TARGET_M_VALUES = [1, 2]  # Number of qubits
TARGET_P_VALUES = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Error probabilities  
TARGET_N_VALUES = [2, 4, 8, 16]  # Copy counts (1, 2, 3 rounds of purification)

# Quick test subset for debugging
QUICK_M_VALUES = [1]
QUICK_P_VALUES = [0.1, 0.2]
QUICK_N_VALUES = [2, 4]


def estimate_qubit_requirements(M: int, N: int) -> int:
    """
    Estimate qubits needed for SWAP purification tree.
    
    Formula: N copies of M qubits + log2(N) ancillas for SWAP tests
    """
    return N * M + int(np.log2(N))


def validate_parameter_feasibility(M: int, N: int, p: float, backend_name: str) -> tuple[bool, str]:
    """
    Check if parameter combination is feasible for the target backend.
    
    Returns:
        (is_feasible, reason) tuple
    """
    # Check qubit requirements
    required_qubits = estimate_qubit_requirements(M, N)
    
    # Conservative limits based on current NISQ hardware
    if required_qubits > 50:  # Well below IBM's ~127 limit for safety
        return False, f"Requires {required_qubits} qubits, exceeds conservative limit"
    
    # Check error probability range
    # if not (0.0 <= p <= 0.5):  # Beyond p=0.5, noise dominates
    #     return False, f"Error probability p={p} outside reasonable range [0, 0.5]"
    
    # Check N is power of 2
    if N <= 1 or (N & (N - 1)) != 0:
        return False, f"N={N} must be a power of 2 and > 1"
        
    # Check M is reasonable
    if M <= 0 or M > 5:  # Conservative limit for current hardware fidelity
        return False, f"M={M} outside practical range [1, 5]"
    
    return True, f"Feasible: {required_qubits} qubits"


def generate_experiment_configurations(M_values: List[int], N_values: List[int], 
                                     P_values: List[float], backend_name: str, 
                                     shots: int) -> List[PurificationConfig]:
    """
    Generate all valid experiment configurations.
    """
    configs = []
    
    for M in M_values:
        for N in N_values:
            for p in P_values:
                # Validate feasibility
                is_feasible, reason = validate_parameter_feasibility(M, N, p, backend_name)
                
                if not is_feasible:
                    logger.warning(f"Skipping M={M}, N={N}, p={p}: {reason}")
                    continue
                
                # Create configuration
                config = PurificationConfig(
                    M=M,
                    N=N,
                    p=p,
                    backend_name=backend_name,
                    shots=shots,
                    max_retry_attempts=3,  # Conservative for stability
                    min_success_rate=0.05,  # Lower threshold for noisy hardware
                )
                
                try:
                    config.validate()
                    configs.append(config)
                    qubits = estimate_qubit_requirements(M, N)
                    rounds = int(np.log2(N))
                    logger.info(f"✓ Valid config: M={M}, N={N} ({rounds} rounds), p={p}, ~{qubits} qubits")
                except ValueError as e:
                    logger.warning(f"✗ Invalid config M={M}, N={N}, p={p}: {e}")
    
    return configs


def save_results_to_csv(results: List[Dict[str, Any]], output_file: Path) -> None:
    """Save experimental results to CSV."""
    if not results:
        logger.warning("No results to save")
        return
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Define exact fieldnames for consistent CSV format
    fieldnames = [
        'run_id',
        'M', 'N', 'p',
        'purification_rounds',
        'estimated_qubits',
        'final_fidelity',
        'swap_success_probability', 
        'total_shots',
        'backend_name',
        'circuit_depth',
        'circuit_qubits',
        'error_message'  # Empty if successful
    ]
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            # Ensure all required fields are present
            clean_result = {field: result.get(field, '') for field in fieldnames}
            # Add computed fields
            clean_result['purification_rounds'] = int(np.log2(result.get('N', 2)))
            clean_result['estimated_qubits'] = estimate_qubit_requirements(
                result.get('M', 1), result.get('N', 2))
            writer.writerow(clean_result)
    
    logger.info(f"💾 Saved {len(results)} results to {output_file}")


def load_existing_results(output_file: Path) -> set[str]:
    """Load existing run IDs to avoid duplicate experiments."""
    if not output_file.exists():
        return set()
    
    existing_run_ids = set()
    try:
        with open(output_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                existing_run_ids.add(row.get('run_id', ''))
        logger.info(f"📂 Found {len(existing_run_ids)} existing results")
    except Exception as e:
        logger.warning(f"⚠️  Could not load existing results: {e}")
    
    return existing_run_ids


def run_swap_purification_grid_sweep(
    backend_name: str = "ibm_torino",
    shots: int = 8192,
    output_dir: Path = Path("data/IBMQ"),
    quick_mode: bool = False,
    resume: bool = True
) -> None:
    """
    Run the complete SWAP purification parameter grid sweep.
    """
    logger.info("🚀 " + "="*60)
    logger.info("🚀 STARTING SWAP PURIFICATION GRID EXPERIMENT")
    logger.info("🚀 " + "="*60)
    
    # Setup output
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"SWAP_purification_results_{backend_name}_{timestamp}.csv"
    logger.info(f"📊 Results will be saved to: {output_file}")
    
    # Choose parameter values
    if quick_mode:
        logger.info("🏃 QUICK MODE: Using reduced parameter space")
        M_values, P_values, N_values = QUICK_M_VALUES, QUICK_P_VALUES, QUICK_N_VALUES
    else:
        M_values, P_values, N_values = TARGET_M_VALUES, TARGET_P_VALUES, TARGET_N_VALUES
    
    logger.info(f"📋 Parameter space:")
    logger.info(f"   M values: {M_values}")
    logger.info(f"   P values: {P_values}")  
    logger.info(f"   N values: {N_values} (corresponding to {[int(np.log2(N)) for N in N_values]} rounds)")
    logger.info(f"   Total combinations: {len(M_values) * len(P_values) * len(N_values)}")
    
    # Generate configurations
    configs = generate_experiment_configurations(
        M_values, N_values, P_values, backend_name, shots)
    
    if not configs:
        logger.error("❌ No valid configurations generated!")
        return
    
    logger.info(f"✅ Generated {len(configs)} valid configurations")
    
    # Show resource requirements
    for config in configs[:5]:  # Show first 5 as examples
        qubits = estimate_qubit_requirements(config.M, config.N)
        rounds = int(np.log2(config.N))
        logger.info(f"   📊 M={config.M}, N={config.N} ({rounds} rounds), p={config.p}: ~{qubits} qubits")
    if len(configs) > 5:
        logger.info(f"   ... and {len(configs)-5} more configurations")
    
    # Filter existing results if resuming
    if resume:
        existing_run_ids = load_existing_results(output_file)
        configs = [c for c in configs if c.synthesize_run_id() not in existing_run_ids]
        logger.info(f"📝 After filtering existing: {len(configs)} experiments to run")
    
    if not configs:
        logger.info("✅ All configurations already completed!")
        return
    
    # Setup IBM backend
    logger.info(f"🔗 Connecting to IBM backend: {backend_name}")
    try:
        service, backend = setup_ibm_backend(backend_name)
        logger.info(f"✅ Backend status: {backend.status()}")
        logger.info(f"📊 Backend qubits: {backend.configuration().n_qubits}")
    except Exception as e:
        logger.error(f"❌ Failed to connect to backend: {e}")
        return
    
    # Run experiments
    results = []
    start_time = time.time()
    
    for i, config in enumerate(configs):
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 Experiment {i+1}/{len(configs)}: {config.synthesize_run_id()}")
        logger.info(f"📊 M={config.M}, N={config.N}, p={config.p}")
        qubits = estimate_qubit_requirements(config.M, config.N)
        rounds = int(np.log2(config.N))
        logger.info(f"🔬 {rounds} purification rounds, ~{qubits} qubits")
        logger.info(f"{'='*60}")
        
        exp_start_time = time.time()
        
        try:
            # Run the experiment
            result = run_complete_purification_experiment(config, service, backend)
            results.append(result)
            
            exp_duration = time.time() - exp_start_time
            
            logger.info(f"✅ Experiment completed in {exp_duration:.1f}s")
            logger.info(f"📈 Fidelity: {result['final_fidelity']:.4f}")
            logger.info(f"🎯 Success probability: {result['swap_success_probability']:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Experiment failed: {e}")
            
            # Save error result
            error_result = {
                'run_id': config.synthesize_run_id(),
                'M': config.M,
                'N': config.N,
                'p': config.p,
                'final_fidelity': -1.0,
                'swap_success_probability': -1.0,
                'total_shots': config.shots,
                'backend_name': config.backend_name,
                'circuit_depth': -1,
                'circuit_qubits': -1,
                'error_message': str(e),
            }
            results.append(error_result)
        
        # Save intermediate results every 3 experiments
        if (i + 1) % 3 == 0:
            save_results_to_csv(results, output_file)
            logger.info(f"💾 Saved intermediate results ({len(results)} total)")
        
        # Progress update
        elapsed_time = time.time() - start_time
        avg_time_per_exp = elapsed_time / (i + 1)
        estimated_remaining = avg_time_per_exp * (len(configs) - i - 1)
        
        logger.info(f"⏱️  Progress: {i+1}/{len(configs)} ({100*(i+1)/len(configs):.1f}%)")
        logger.info(f"⏱️  Elapsed: {elapsed_time/60:.1f} min, Est. remaining: {estimated_remaining/60:.1f} min")
    
    # Final save
    save_results_to_csv(results, output_file)
    
    # Summary
    total_time = time.time() - start_time
    successful_experiments = sum(1 for r in results if r.get('final_fidelity', -1) >= 0)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🏁 SWAP PURIFICATION GRID EXPERIMENT COMPLETED")
    logger.info(f"{'='*60}")
    logger.info(f"📊 Total experiments: {len(results)}")
    logger.info(f"✅ Successful: {successful_experiments}")
    logger.info(f"❌ Failed: {len(results) - successful_experiments}")
    logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
    logger.info(f"⏱️  Average per experiment: {total_time/len(configs):.1f} seconds")
    logger.info(f"💾 Results saved to: {output_file}")
    logger.info(f"{'='*60}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SWAP Purification Grid Experiment for IBM Quantum Hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full experiment (recommended)
  python optimized_IBMQ_grid_run.py --backend ibm_torino --shots 8192
  
  # Quick test
  python optimized_IBMQ_grid_run.py --quick --shots 2048
  
  # High-statistics run  
  python optimized_IBMQ_grid_run.py --backend ibm_torino --shots 16384
        """
    )
    
    parser.add_argument("--backend", type=str, default="ibm_torino",
                       help="IBM Quantum backend name")
    parser.add_argument("--shots", type=int, default=8192,
                       help="Number of measurement shots per experiment")
    parser.add_argument("--output-dir", type=Path, default=Path("data/IBMQ"),
                       help="Output directory for results")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test mode with reduced parameter space")
    parser.add_argument("--no-resume", action="store_true",
                       help="Don't skip existing experiments")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable debug-level logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print configuration
    logger.info(f"🔧 Configuration:")
    logger.info(f"   Backend: {args.backend}")
    logger.info(f"   Shots per experiment: {args.shots}")
    logger.info(f"   Output directory: {args.output_dir}")
    logger.info(f"   Quick mode: {args.quick}")
    logger.info(f"   Resume mode: {not args.no_resume}")
    
    try:
        run_swap_purification_grid_sweep(
            backend_name=args.backend,
            shots=args.shots,
            output_dir=args.output_dir,
            quick_mode=args.quick,
            resume=not args.no_resume
        )
        logger.info("🎉 Grid sweep completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Grid sweep interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Grid sweep failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())