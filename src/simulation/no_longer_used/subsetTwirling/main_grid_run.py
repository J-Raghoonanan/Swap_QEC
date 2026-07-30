"""
Main entry point to run a grid sweep of circuit-level purification simulations.

This script uses the *density-matrix* Aer simulator and the SWAP-test based
purification implemented in this package. It writes CSVs under
`data/subsetTwirling_simulations/` that are directly consumable by your figure-generation
scripts.

NEW in subsetTwirling:
- Added --subset-fraction CLI argument to control Clifford twirling subset size
- Added --subset-mode to choose between random sampling or deterministic selection
- Optimized for resource-constrained environments where full 3^M twirling is expensive

UPDATED FEATURES:
- Explicit Kraus operators only (no Qiskit DepolarizingChannel dependency)
- Automatic Clifford twirling for dephasing noise types
- Enhanced logging with Bloch vector tracking for M=1
- Fixed identical-copies bug in exact_k mode
- Support for both iid_p and exact_k noise modes

CHOICES (documented):
- We cap M at 6 by default because the density matrix for the merge uses
  1 + 2M qubits (ancilla + two registers). Memory/time grows quickly with M.
- We run **i.i.d. per-qubit channels** (NoiseMode.iid_p) so every input copy is
  statistically identical. This aligns with the ρ ⊗ ρ model in the manuscript
  and keeps results clean and reproducible.
- Target state defaults to **Hadamard** (H^{⊗M} |0…0⟩). You can change to Haar
  or random circuits by editing the `TARGET_KIND` constant below.
- Amplitude amplification is **emulated**: we compute and log the required
  Grover iterations but we don't explicitly apply Q^k. The postselected output
  state (ancilla = 0) is identical either way.
- Clifford twirling is **automatically enabled** for dephasing noise types
  (dephase_z, dephase_x) to convert them to effective depolarization.

You can run this file directly:
        
    python -m src.simulation.subsetTwirling.main_grid_run \
        --out data/subsetTwirling_simulations \
        --noise z \
        --m-values 1 2 3 4 5 \
        --iterative \
        --subset-fraction 0.5
        
    python -m src.simulation.subsetTwirling.main_grid_run \
        --out data/subsetTwirling_simulations \
        --noise z \
        --m-values 1 2 3 4 5 \
        --iterative \
        --subset-fraction 0.1

It will append to `steps_circuit.csv` and `finals_circuit.csv`.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import List
import numpy as np

from .configs import (
    RunSpec,
    TargetSpec,
    NoiseSpec,
    AASpec,
    TwirlingSpec,
    NoiseType,
    NoiseMode,
    StateKind,
)
from .streaming_runner import run_and_save

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------
# Defaults for the sweep
# -----------------------------
M_LIST: List[int] = [1, 2, 3, 4, 5]  # keep ≤ 6 for density-matrix practicality
N_LIST: List[int] = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
# N_LIST: List[int] = [2]
P_LIST: List[float] = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NOISES: List[NoiseType] = [NoiseType.depolarizing, NoiseType.dephase_z]
TARGET_KIND: StateKind = StateKind.hadamard  # change to StateKind.haar for random pure states
# TARGET_KIND: StateKind = StateKind.single_qubit_product
BACKEND_METHOD: str = "density_matrix"
L_LIST: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# L_LIST: List[int] = [4, 5, 6, 7, 8, 9, 10]

# AA configuration (emulated)
AA = AASpec(target_success=0.99, max_iters=32, use_postselection_only=False)


# -----------------------------
# CLI
# -----------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid sweep for SWAP-QEC circuit simulation with subset twirling")
    p.add_argument("--out", type=Path, default=Path("data/subsetTwirling_simulations"), help="Output directory for CSVs")
    p.add_argument("--max-m", type=int, default=5, help="Maximum M to include (≤ 6 recommended)")
    p.add_argument("--m-values", type=int, nargs='+', help="Specific M values to run (e.g., --m-values 1 3 5)")
    p.add_argument("--seed", type=int, default=1, help="Seed for target-state generation where applicable")
    p.add_argument(
        "--noise",
        choices=["all", "depol", "z", "x"],
        default="all",
        help="Which noise families to simulate (default: all)",
    )
    p.add_argument(
        "--mode",
        choices=[m.value for m in NoiseMode],
        default=NoiseMode.iid_p.value,
        help="Noise application mode (iid_p is manuscript-consistent)",
    )
    p.add_argument(
        "--target",
        choices=[k.value for k in StateKind],
        default=TARGET_KIND.value,
        help="Target state family for |ψ⟩",
    )
    p.add_argument(
        "--no-twirl",
        action="store_true",
        help="Disable Clifford twirling even for dephasing noise",
    )
    p.add_argument(
        "--subset-fraction",
        type=float,
        default=1.0,
        help="Fraction of 3^M Clifford combinations to use in twirling (0.0 to 1.0)",
    )
    p.add_argument(
        "--subset-mode",
        choices=["random", "first_k"],
        default="random",
        help="How to select Clifford subset: random sampling or first K combinations",
    )
    p.add_argument(
        "--subset-seed",
        type=int,
        default=None,
        help="Seed for subset selection (for reproducibility)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick test with reduced parameter space",
    )
    p.add_argument(
        "--iterative",
        action="store_true",
        help="Enable iterative noise mode: apply fresh noise before each iteration round",
    )
    p.add_argument(
        "--purification-level",
        type=int,
        default=1,
        help="Number of SWAP purification rounds per iteration (ℓ parameter)",
    )
    return p.parse_args()


def _pick_noises(flag: str) -> List[NoiseType]:
    if flag == "all":
        return NOISES
    if flag == "depol":
        return [NoiseType.depolarizing]
    if flag == "z":
        return [NoiseType.dephase_z]
    if flag == "x":
        return [NoiseType.dephase_x]
    raise ValueError(flag)


# -----------------------------
# Main sweep
# -----------------------------

def main() -> None:
    args = _parse_args()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    noises = _pick_noises(args.noise)
    mode = NoiseMode(args.mode)
    target_kind = StateKind(args.target)

    # Respect the M cap explicitly, or use specific values if provided
    if args.m_values:
        Ms = [m for m in args.m_values if m <= 6]  # Still cap at 6 for practical reasons
        if not Ms:
            raise ValueError("No valid M values provided (must be ≤ 6)")
        logger.info(f"Using specific M values: {Ms}")
    else:
        Ms = [m for m in M_LIST if m <= args.max_m]
        logger.info(f"Using M range: 1 to {args.max_m}")
    
    # Quick test mode: reduce parameter space
    if args.quick:
        logger.info("QUICK TEST MODE: Using reduced parameter space")
        if args.m_values:
            Ms = Ms[:2]  # Take first 2 from specified values
        else:
            Ms = [1, 2]
        Ns = [4, 16, 64]
        ps = [0.1, 0.5, 0.9]
        Ls = [0, 1]  # keep quick test short
    else:
        Ns = N_LIST
        ps = P_LIST
        Ls = L_LIST

    # Twirling config with subset support
    twirling = TwirlingSpec(
        enabled=not args.no_twirl,
        mode="cyclic",
        seed=args.seed,
        subset_fraction=args.subset_fraction,
        subset_mode=args.subset_mode,
        subset_seed=args.subset_seed,
    )

    logger.info(
        "="*70 + "\n"
        "Running grid sweep with SUBSET TWIRLING:\n"
        f"  Ms                = {Ms}\n"
        f"  Ns                = {Ns}\n"
        f"  ps                = {ps}\n"
        f"  noises            = {[n.value for n in noises]}\n"
        f"  mode              = {mode.value}\n"
        f"  target_kind       = {target_kind.value}\n"
        f"  backend           = {BACKEND_METHOD}\n"
        f"  twirling          = {'enabled' if twirling.enabled else 'disabled'}\n"
        f"  subset_fraction   = {twirling.subset_fraction:.2f}\n"
        f"  subset_mode       = {twirling.subset_mode}\n"
        f"  out_dir           = {out_dir}\n" +
        "="*70
    )

    started = time.time()
    total_runs = len(noises) * len(Ms) * len(Ns) * len(ps) * len(Ls)
    current_run = 0

    for noise in noises:
        for M in Ms:
            # Target |ψ⟩ spec:
            target = TargetSpec(M=M, kind=target_kind, seed=args.seed, product_theta=np.pi/3, product_phi=np.pi/4)
            for N in Ns:
                for p in ps: 
                    for ell in Ls:
                        current_run += 1
                        
                        spec = RunSpec(
                            target=target,
                            noise=NoiseSpec(noise_type=noise, mode=mode, p=p),
                            aa=AA,
                            twirling=twirling,
                            N=N,
                            backend_method=BACKEND_METHOD,
                            out_dir=out_dir,
                            verbose=args.verbose,
                            iterative_noise=args.iterative,
                            purification_level=ell,
                        )
                        
                        tag = spec.synthesize_run_id()
                        
                        logger.info(f"\n{'='*70}")
                        logger.info(f"Run {current_run}/{total_runs}: {tag} (ℓ={ell})")
                        logger.info(f"{'='*70}")
                        
                        t0 = time.time()
                        try:
                            run_and_save(spec)
                            dt = time.time() - t0
                            logger.info(f"✓ Completed in {dt:.1f}s\n")
                        except Exception as e:
                            # Keep sweeping on errors; log and continue
                            logger.error(f"✗ ERROR during {tag}: {e}\n", exc_info=True)

    total = time.time() - started
    logger.info(f"\n{'='*70}")
    logger.info(f"Grid sweep complete!")
    logger.info(f"  Total time: {total/60:.1f} min")
    logger.info(f"  Runs: {current_run}/{total_runs}")
    logger.info(f"  Data saved to: {out_dir}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
