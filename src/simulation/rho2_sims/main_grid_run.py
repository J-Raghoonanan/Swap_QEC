"""
Main entry point to run a grid sweep of purification simulations.

This script uses the density-matrix Aer simulator and 
implemented in this package. It writes CSVs under `data/rho2_sim/` that are
directly consumable by figure-generation scripts.

KEY DIFFERENCES FROM SWAP PURIFICATION:
- Uses ρ → ρ²/Tr(ρ²) instead of SWAP test
- P_success = 1.0 always (deterministic, no postselection)
- C_ℓ = 2^ℓ exactly (no overhead from failed attempts)
- G_ℓ = 2^ℓ - 1 (total rho2 operations through level ℓ)
- ~50% fewer copies needed vs SWAP
- ~25% fewer operations vs SWAP

FEATURES:
- Explicit Kraus operators only (no Qiskit DepolarizingChannel dependency)
- Automatic Clifford twirling for dephasing noise types
- Enhanced logging with Bloch vector tracking for M=1
- Support for both iid_p and exact_k noise modes
- Iterative mode: apply noise once per iteration, then ℓ rho2 purification levels

CHOICES (documented):
- We cap M at 6 by default because density matrix operations scale as 2^(2M).
- We run **i.i.d. per-qubit channels** (NoiseMode.iid_p) so every input copy is
  statistically identical. This aligns with the ρ ⊗ ρ model.
- Target state defaults to **single-qubit product** (|ψ(θ,φ)⟩^⊗M).
- Clifford twirling is **automatically enabled** for dephasing noise types
  to convert them to effective depolarization.

Usage examples:

    python -m src.simulation.rho2_sims.main_grid_run \
        --out data/rho2_sim \
        --noise depol \
        --m-values 1 2 3 4 5 \
        --iterative
    
    python -m src.simulation.rho2_sims.main_grid_run \
        --out data/rho2_sim \
        --noise z \
        --m-values 1 2 3 45 \
        --iterative \

    python -m src.simulation.rho2_sims.main_grid_run \
        --out data/rho2_sim \
        --noise z \
        --m-values 1 2 3 4 5 \
        --iterative \
        --no-twirl
        
    python -m src.simulation.rho2_sims.main_grid_run \
        --out data/rho2_sim \
        --noise z \
        --m-values 1 2 3 45 \
        --iterative \
        --no-twirl \
        --target single_qubit_product
        
        
        
        
    python -m src.simulation.rho2_sims.main_grid_run \
        --out data/rho2_sim \
        --max-m 5 \
        --noise all \
        --quick

It will append to `steps_rho2_*.csv` and `finals_rho2_*.csv`.
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
M_LIST: List[int] = [1, 2, 3, 4, 5]
# N_LIST: List[int] = [2] # For 1 PQEC cycle
N_LIST: List[int] = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
P_LIST: List[float] = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# P_LIST: List[float] = [0.75]
NOISES: List[NoiseType] = [NoiseType.depolarizing, NoiseType.dephase_z]
TARGET_KIND: StateKind = StateKind.hadamard  # change to StateKind.haar for random pure states
# TARGET_KIND: StateKind = StateKind.single_qubit_product
BACKEND_METHOD: str = "density_matrix"
L_LIST: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Purification levels
# L_LIST: List[int] = [0, 1, 2, 3] # For running M=1,5 for more PQEC cycles

# AA configuration (not needed but kept for API compatibility)
AA = AASpec(target_success=0.99, max_iters=32, use_postselection_only=False)

# Twirling configuration (auto-enabled for dephasing)
TWIRLING = TwirlingSpec(enabled=True, mode="cyclic", seed=None)


# -----------------------------
# CLI
# -----------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid sweep for rho2 purification")
    p.add_argument("--out", type=Path, default=Path("data/rho2_sim"), help="Output directory for CSVs")
    p.add_argument("--max-m", type=int, default=6, help="Maximum M to include (≤ 6 recommended)")
    p.add_argument("--m-values", type=int, nargs='+', help="Specific M values to run (e.g., --m-values 1 3 5)")
    p.add_argument("--seed", type=int, default=1, help="Seed for target-state generation")
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
        help="Noise application mode (iid_p recommended)",
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
        help="Number of rho2 purification rounds per iteration (ℓ parameter)",
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
        Ms = [m for m in args.m_values if m <= 6]
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
            Ms = Ms[:2]
        else:
            Ms = [1, 2]
        Ns = [4, 16, 64]
        ps = [0.1, 0.5, 0.9]
        Ls = [0, 1]
    else:
        Ns = N_LIST
        ps = P_LIST
        Ls = L_LIST

    # Twirling config
    twirling = TwirlingSpec(enabled=not args.no_twirl, mode="cyclic", seed=args.seed)

    logger.info(
        "="*70 + "\n"
        "Running rho2 grid sweep with:\n"
        f"  Method       = rho2\n"
        f"  P_success    = 1.0 always (deterministic)\n"
        f"  Ms           = {Ms}\n"
        f"  Ns           = {Ns}\n"
        f"  ps           = {ps}\n"
        f"  Ls           = {Ls}\n"
        f"  noises       = {[n.value for n in noises]}\n"
        f"  mode         = {mode.value}\n"
        f"  target_kind  = {target_kind.value}\n"
        f"  backend      = {BACKEND_METHOD}\n"
        f"  twirling     = {'enabled' if twirling.enabled else 'disabled'}\n"
        f"  iterative    = {args.iterative}\n"
        f"  out_dir      = {out_dir}\n" +
        "="*70
    )

    started = time.time()
    total_runs = len(noises) * len(Ms) * len(Ns) * len(ps) * len(Ls)
    current_run = 0

    for noise in noises:
        for M in Ms:
            # Target |ψ⟩ spec: single-qubit product state
            target = TargetSpec(
                M=M, 
                kind=target_kind, 
                seed=args.seed,
                product_theta=np.pi/3,
                product_phi=np.pi/4
            )
            for N in Ns:
                for p in ps: 
                    for ell in Ls:
                        current_run += 1
                        
                        spec = RunSpec(
                            target=target,
                            noise=NoiseSpec(noise_type=noise, mode=mode, p=p),
                            aa=AA,  # Not used but kept for compatibility
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
                        logger.info(f"Run {current_run}/{total_runs}: {tag} (ℓ={ell}) [rho2]")
                        logger.info(f"{'='*70}")
                        
                        t0 = time.time()
                        try:
                            run_and_save(spec)
                            dt = time.time() - t0
                            logger.info(f"✓ Completed in {dt:.1f}s\n")
                        except Exception as e:
                            logger.error(f"✗ ERROR during {tag}: {e}\n", exc_info=True)

    total = time.time() - started
    logger.info(f"\n{'='*70}")
    logger.info(f"rho2 grid sweep complete!")
    logger.info(f"  Total time: {total/60:.1f} min")
    logger.info(f"  Runs: {current_run}/{total_runs}")
    logger.info(f"  Data saved to: {out_dir}")
    logger.info(f"  Method: rho2 (deterministic, P_success=1.0)")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()