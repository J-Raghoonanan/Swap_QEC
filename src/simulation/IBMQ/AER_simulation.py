# aer_grid_run.py
"""
Grid run for SWAP purification experiments using the Qiskit Aer simulator.

This script mirrors optimized_IBMQ_grid_run.py but runs everything on
an ideal (or user-specified) simulator instead of IBM Quantum hardware.

Usage examples:
  # Full experiment (recommended to start with smaller shots)
    python -m src.simulation.AER_simulation --shots 1024

  # Quick test
  python -m src.simulation.AER_simulation --quick --shots 2048

  # More Monte Carlo sampling over Pauli noise patterns
  python -m src.simulation.AER_simulation --noise-realizations 16
"""

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from qiskit_aer import AerSimulator

from .IBMQ_components import (
    PurificationConfig,
    create_batch_purification_circuit,
    add_measurements,
    analyze_results,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameter grids (same logic as optimized_IBMQ_grid_run)
# ---------------------------------------------------------------------------

TARGET_M_VALUES = [1, 2]  # Number of logical qubits in |+>^⊗M
TARGET_P_VALUES = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TARGET_N_VALUES = [2, 4, 8, 16]  # Number of copies (N = 2^ℓ)

# Quick test subset
QUICK_M_VALUES = [1]
QUICK_P_VALUES = [0.1, 0.2]
QUICK_N_VALUES = [2, 4]

# Monte Carlo defaults
DEFAULT_NOISE_REALIZATIONS = 8
QUICK_NOISE_REALIZATIONS = 3


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def estimate_qubit_requirements(M: int, N: int) -> int:
    """
    Estimate total qubits needed for the SWAP purification tree.

    N copies of M data qubits + (N - 1) ancillas (1 per SWAP test).
    """
    return N * M + (N - 1)


def validate_parameter_feasibility(M: int, N: int, p: float) -> Tuple[bool, str]:
    """
    Check if (M, N, p) is "reasonable" for simulation.

    We still keep some conservative bounds to avoid gigantic circuits,
    but we don't rely on hardware limits.
    """
    required_qubits = estimate_qubit_requirements(M, N)

    # Keep simulation size modest
    if required_qubits > 50:
        return False, f"Requires {required_qubits} qubits, which is large for Aer tests"

    if N <= 1 or (N & (N - 1)) != 0:
        return False, f"N={N} must be a power of 2 and > 1"

    if M <= 0 or M > 8:
        return False, f"M={M} outside practical range [1, 8]"

    if not (0.0 <= p <= 1.0):
        return False, f"p={p} must be in [0, 1]"

    return True, f"Feasible: {required_qubits} qubits"


def generate_experiment_configurations(
    M_values: List[int],
    N_values: List[int],
    P_values: List[float],
    noise_types: List[str],
    shots: int,
    noise_realizations: int,
) -> List[PurificationConfig]:
    """
    Generate valid PurificationConfig objects for the given parameter grid.
    """
    configs: List[PurificationConfig] = []

    for M in M_values:
        for N in N_values:
            for p in P_values:
                for noise_type in noise_types:
                    is_ok, reason = validate_parameter_feasibility(M, N, p)
                    if not is_ok:
                        logger.warning(
                            f"Skipping M={M}, N={N}, p={p}, {noise_type}: {reason}"
                        )
                        continue

                    cfg = PurificationConfig(
                        M=M,
                        N=N,
                        p=p,
                        noise_type=noise_type,
                        backend_name="aer_simulator",
                        shots=shots,
                        max_retry_attempts=1,      # not used in Aer version
                        min_success_rate=0.0,      # not used in Aer version
                        num_noise_realizations=noise_realizations,
                    )

                    try:
                        cfg.validate()
                    except ValueError as e:
                        logger.warning(
                            f"Invalid config M={M}, N={N}, p={p}, {noise_type}: {e}"
                        )
                        continue

                    qubits = estimate_qubit_requirements(M, N)
                    rounds = int(np.log2(N))
                    logger.info(
                        f"✓ Valid config: M={M}, N={N} (ℓ={rounds}), p={p}, "
                        f"{noise_type}, ~{qubits} qubits, "
                        f"noise_realizations={noise_realizations}"
                    )
                    configs.append(cfg)

    return configs


def save_results_to_csv(results: List[Dict[str, Any]], output_file: Path) -> None:
    """Save experimental results to CSV (same column layout as IBM grid run)."""
    if not results:
        logger.warning("No results to save")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_id",
        "M",
        "N",
        "p",
        "noise_type",
        "purification_rounds",
        "estimated_qubits",
        "final_fidelity",
        "swap_success_probability",
        "total_shots",
        "backend_name",
        "circuit_depth",
        "circuit_qubits",
        "num_noise_realizations",
        "error_message",
    ]

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for res in results:
            clean = {k: res.get(k, "") for k in fieldnames}
            clean["purification_rounds"] = int(np.log2(clean["N"] or 2))
            clean["estimated_qubits"] = estimate_qubit_requirements(
                clean["M"] or 1, clean["N"] or 2
            )
            writer.writerow(clean)

    logger.info(f"💾 Saved {len(results)} results to {output_file}")


# ---------------------------------------------------------------------------
# Aer-specific complete experiment
# ---------------------------------------------------------------------------

def run_complete_purification_experiment_aer(
    config: PurificationConfig,
    simulator: Optional[AerSimulator] = None,
) -> Dict[str, Any]:
    """
    Aer version of run_complete_purification_experiment.

    For each noise realization:
      * build a fresh noisy circuit with base_noise_seed = 100000 * r
      * add measurements
      * run on Aer with `config.shots`
      * aggregate all counts across realizations

    Returns a result dict compatible with the IBM grid-run output.
    """
    config.validate()

    if simulator is None:
        simulator = AerSimulator()

    logger.info(
        f"Starting Aer purification experiment: {config.synthesize_run_id()} "
        f"(M={config.M}, N={config.N}, p={config.p}, "
        f"noise={config.noise_type}, num_real={config.num_noise_realizations})"
    )

    global_counts: Dict[str, int] = {}
    measured_depth = None
    measured_qubits = None

    for r in range(config.num_noise_realizations):
        base_noise_seed = 100000 * r
        logger.info(
            f"--- Noise realization {r+1}/{config.num_noise_realizations} "
            f"(base_noise_seed={base_noise_seed}) ---"
        )

        # Build circuit for this noise pattern
        purification_circuit = create_batch_purification_circuit(
            config, base_noise_seed=base_noise_seed
        )
        measured_circuit = add_measurements(purification_circuit, config)

        # Remember some structural info (same for all realizations)
        measured_depth = measured_circuit.depth()
        measured_qubits = measured_circuit.num_qubits

        # Run on Aer
        job = simulator.run(measured_circuit, shots=config.shots)
        result = job.result()
        counts_r = result.get_counts()

        # Aggregate
        for outcome, count in counts_r.items():
            global_counts[outcome] = global_counts.get(outcome, 0) + count

        fid_r, succ_r = analyze_results(counts_r, config)
        logger.info(
            f"Realization {r+1}: fidelity={fid_r:.4f}, "
            f"success_prob={succ_r:.4f}, shots={sum(counts_r.values())}"
        )

    final_fidelity, final_success_prob = analyze_results(global_counts, config)

    logger.info("=== Aggregated Aer results over all noise realizations ===")
    logger.info(f"  Total shots: {sum(global_counts.values())}")
    logger.info(f"  Final fidelity: {final_fidelity:.4f}")
    logger.info(f"  Final success probability: {final_success_prob:.4f}")

    result_dict: Dict[str, Any] = {
        "run_id": config.synthesize_run_id(),
        "M": config.M,
        "N": config.N,
        "p": config.p,
        "noise_type": config.noise_type,
        "purification_rounds": int(np.log2(config.N)),
        "estimated_qubits": estimate_qubit_requirements(config.M, config.N),
        "final_fidelity": final_fidelity,
        "swap_success_probability": final_success_prob,
        "total_shots": sum(global_counts.values()),
        "backend_name": "aer_simulator",
        "circuit_depth": measured_depth if measured_depth is not None else -1,
        "circuit_qubits": measured_qubits if measured_qubits is not None else -1,
        "num_noise_realizations": config.num_noise_realizations,
        "error_message": "",
    }

    return result_dict


# ---------------------------------------------------------------------------
# Grid sweep driver (Aer)
# ---------------------------------------------------------------------------

def run_swap_purification_grid_sweep_aer(
    shots: int = 8192,
    output_dir: Path = Path("data/IBMQ"),
    quick_mode: bool = False,
    noise_realizations: int = DEFAULT_NOISE_REALIZATIONS,
) -> None:
    """Run full parameter grid using Aer simulator."""
    logger.info("🚀 " + "=" * 60)
    logger.info("🚀 STARTING SWAP PURIFICATION GRID EXPERIMENT (AER + MC)")
    logger.info("🚀 " + "=" * 60)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"Aer_results_all_{timestamp}.csv"
    logger.info(f"📊 Results will be saved to: {output_file}")
    logger.info(f"🎲 Monte Carlo noise realizations per config: {noise_realizations}")

    # Parameter sets
    if quick_mode:
        logger.info("🏃 QUICK MODE: reduced parameter space")
        M_values, P_values, N_values = QUICK_M_VALUES, QUICK_P_VALUES, QUICK_N_VALUES
        noise_types = ["depolarizing"]  # keep quick
    else:
        M_values, P_values, N_values = TARGET_M_VALUES, TARGET_P_VALUES, TARGET_N_VALUES
        noise_types = ["depolarizing", "dephasing"]

    logger.info("📋 Parameter space:")
    logger.info(f"   M values: {M_values}")
    logger.info(f"   P values: {P_values}")
    logger.info(
        f"   N values: {N_values} "
        f"(ℓ values: {[int(np.log2(N)) for N in N_values]})"
    )
    logger.info(f"   Noise types: {noise_types}")
    logger.info(f"   Noise realizations per config: {noise_realizations}")
    logger.info(
        f"   Total combinations: "
        f"{len(M_values) * len(P_values) * len(N_values) * len(noise_types)}"
    )

    configs = generate_experiment_configurations(
        M_values=M_values,
        N_values=N_values,
        P_values=P_values,
        noise_types=noise_types,
        shots=shots,
        noise_realizations=noise_realizations,
    )

    if not configs:
        logger.error("❌ No valid configurations generated for Aer grid run")
        return

    logger.info(f"✅ Generated {len(configs)} valid configurations")

    # Create a single simulator instance
    simulator = AerSimulator()

    results: List[Dict[str, Any]] = []
    start_time = time.time()

    for i, cfg in enumerate(configs):
        logger.info("\n" + "=" * 60)
        logger.info(f"🧪 Aer experiment {i+1}/{len(configs)}: {cfg.synthesize_run_id()}")
        logger.info(
            f"   M={cfg.M}, N={cfg.N}, ℓ={int(np.log2(cfg.N))}, p={cfg.p}, "
            f"noise={cfg.noise_type}, num_real={cfg.num_noise_realizations}"
        )
        logger.info("=" * 60)

        exp_start = time.time()
        try:
            res = run_complete_purification_experiment_aer(cfg, simulator=simulator)
            results.append(res)

            exp_dur = time.time() - exp_start
            logger.info(f"✅ Experiment completed in {exp_dur:.1f}s")
            logger.info(f"📈 Final fidelity: {res['final_fidelity']:.4f}")
            logger.info(
                f"🎯 SWAP success probability: {res['swap_success_probability']:.4f}"
            )
        except Exception as e:
            logger.error(f"❌ Aer experiment failed: {e}")
            err_res = {
                "run_id": cfg.synthesize_run_id(),
                "M": cfg.M,
                "N": cfg.N,
                "p": cfg.p,
                "noise_type": cfg.noise_type,
                "purification_rounds": int(np.log2(cfg.N)),
                "estimated_qubits": estimate_qubit_requirements(cfg.M, cfg.N),
                "final_fidelity": -1.0,
                "swap_success_probability": -1.0,
                "total_shots": cfg.shots * cfg.num_noise_realizations,
                "backend_name": "aer_simulator",
                "circuit_depth": -1,
                "circuit_qubits": -1,
                "num_noise_realizations": cfg.num_noise_realizations,
                "error_message": str(e),
            }
            results.append(err_res)

        # Save intermediate results every 5 experiments
        if (i + 1) % 5 == 0:
            save_results_to_csv(results, output_file)
            logger.info(f"💾 Saved intermediate results ({len(results)} total)")

        elapsed = time.time() - start_time
        avg_per = elapsed / (i + 1)
        remaining = avg_per * (len(configs) - i - 1)
        logger.info(
            f"⏱️  Progress: {i+1}/{len(configs)} "
            f"({100*(i+1)/len(configs):.1f}%)"
        )
        logger.info(f"⏱️  Elapsed: {elapsed/60:.1f} min, "
                    f"Est. remaining: {remaining/60:.1f} min")

    # Final save and summary
    save_results_to_csv(results, output_file)

    total_time = time.time() - start_time
    successful = sum(1 for r in results if r.get("final_fidelity", -1.0) >= 0)

    logger.info("\n" + "=" * 60)
    logger.info("🏁 AER SWAP PURIFICATION GRID EXPERIMENT COMPLETED")
    logger.info("=" * 60)
    logger.info(f"📊 Total experiments: {len(results)}")
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {len(results) - successful}")
    logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
    logger.info(
        f"⏱️  Average per experiment: {total_time/max(len(configs),1):.1f} seconds"
    )
    logger.info(f"💾 Results saved to: {output_file}")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="SWAP Purification Grid Experiment on Qiskit Aer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full experiment
  python aer_grid_run.py --shots 8192

  # Quick test
  python aer_grid_run.py --quick --shots 2048

  # Custom Monte Carlo sampling
  python aer_grid_run.py --noise-realizations 16
        """,
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=8192,
        help="Shots per experiment per noise realization",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/IBMQ"),
        help="Directory to store CSV results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with reduced parameter space",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    parser.add_argument(
        "--noise-realizations",
        type=int,
        default=None,
        help=(
            "Number of Monte Carlo noise realizations per configuration "
            f"(default {DEFAULT_NOISE_REALIZATIONS}, "
            f"or {QUICK_NOISE_REALIZATIONS} in quick mode)"
        ),
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.noise_realizations is not None:
        noise_realizations = args.noise_realizations
    else:
        noise_realizations = (
            QUICK_NOISE_REALIZATIONS if args.quick else DEFAULT_NOISE_REALIZATIONS
        )

    logger.info("🔧 Aer grid-run configuration:")
    logger.info(f"   Shots per realization: {args.shots}")
    logger.info(f"   Output directory: {args.output_dir}")
    logger.info(f"   Quick mode: {args.quick}")
    logger.info(f"   Noise realizations per config: {noise_realizations}")

    try:
        run_swap_purification_grid_sweep_aer(
            shots=args.shots,
            output_dir=args.output_dir,
            quick_mode=args.quick,
            noise_realizations=noise_realizations,
        )
        logger.info("🎉 Aer grid sweep completed successfully!")
        return 0
    except KeyboardInterrupt:
        logger.info("\n⚠️  Aer grid sweep interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Aer grid sweep failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
