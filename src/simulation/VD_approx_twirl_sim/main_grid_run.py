"""
Main entry point for Virtual Distillation (VD) grid sweep simulations.

This script uses density-matrix simulations and implements VD purification
(ρ → ρ²/Tr(ρ²)) with optional approximate Clifford twirling for resource
efficiency when dealing with dephasing noise.

VD FEATURES:
- Deterministic purification: ρ → ρ²/Tr(ρ²)
- No amplitude amplification needed (deterministic operation)
- Approximate Clifford twirling for dephasing noise mitigation
  (subset_fraction = 1.0 → exact full twirl; < 1.0 → approximate)

CHOICES (documented):
- We cap M at 6 by default (memory grows quickly with density matrices)
- Uses i.i.d. per-qubit channels (NoiseMode.iid_p)
- Target state defaults to Hadamard product state |+⟩^⊗M
- Clifford twirling is automatically enabled for dephasing noise types

You can run this file directly:

    python -m src.simulation.VD_approx_twirl_sim.main_grid_run \
        --out data/VD_sim \
        --noise z \
        --m-values 1 5 \
        --iterative \
        --subset-fraction 0.2 \
        --subset-seed 42

It will append to `steps_vd_*.csv` and `finals_vd_*.csv`.
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
# N_LIST: List[int] = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
N_LIST: List[int] = [2]
# P_LIST: List[float] = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
P_LIST: List[float] = [0.75]
NOISES: List[NoiseType] = [NoiseType.depolarizing, NoiseType.dephase_z]
TARGET_KIND: StateKind = StateKind.hadamard
BACKEND_METHOD: str = "density_matrix"
L_LIST: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # ℓ parameter

# AA configuration (not used for VD, but kept for API compatibility)
AA = AASpec(target_success=0.99, max_iters=32, use_postselection_only=False)


# -----------------------------
# CLI
# -----------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid sweep for VD simulation with subset twirling")
    p.add_argument("--out", type=Path, default=Path("data/VD_sim"), help="Output directory for CSVs")
    p.add_argument("--max-m", type=int, default=5, help="Maximum M to include (≤ 6 recommended)")
    p.add_argument("--m-values", type=int, nargs='+', help="Specific M values (e.g., --m-values 1 5)")
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
        help="Seed for subset selection (for reproducibility; None → use --seed)",
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
        help="Enable iterative noise mode: apply fresh noise before each VD round",
    )
    p.add_argument(
        "--purification-level",
        type=int,
        default=1,
        help="Number of VD purification rounds per iteration (ℓ parameter)",
    )
    p.add_argument(
        "--theta",
        type=float,
        default=np.pi/3,
        help="Theta parameter for single_qubit_product state (default: π/3)",
    )
    p.add_argument(
        "--phi",
        type=float,
        default=np.pi/4,
        help="Phi parameter for single_qubit_product state (default: π/4)",
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

    noises      = _pick_noises(args.noise)
    mode        = NoiseMode(args.mode)
    target_kind = StateKind(args.target)

    if args.m_values:
        Ms = [m for m in args.m_values if m <= 6]
        if not Ms:
            raise ValueError("No valid M values provided (must be ≤ 6)")
        logger.info(f"Using specific M values: {Ms}")
    else:
        Ms = [m for m in M_LIST if m <= args.max_m]
        logger.info(f"Using M range: 1 to {args.max_m}")

    if args.quick:
        logger.info("QUICK TEST MODE: reduced parameter space")
        Ms = Ms[:2] if args.m_values else [1, 5]
        Ns = [4, 16, 64]
        ps = [0.1, 0.5, 0.9]
        Ls = [0, 1]
    else:
        Ns = N_LIST
        ps = P_LIST
        Ls = L_LIST

    # Use args.subset_seed from the CLI (may be None, meaning fall back to args.seed
    # inside noise_engine via twirl_seed=spec.target.seed).
    twirling = TwirlingSpec(
        enabled=not args.no_twirl,
        mode="cyclic",
        seed=args.seed,
        subset_fraction=args.subset_fraction,
        subset_mode=args.subset_mode,
        subset_seed=args.subset_seed,   # None → noise_engine falls back to target.seed
    )

    logger.info(
        "=" * 70 + "\n"
        "Running VD grid sweep with APPROXIMATE TWIRLING:\n"
        f"  Ms                = {Ms}\n"
        f"  Ns                = {Ns}\n"
        f"  ps                = {ps}\n"
        f"  Ls (ℓ parameter)  = {Ls}\n"
        f"  noises            = {[n.value for n in noises]}\n"
        f"  mode              = {mode.value}\n"
        f"  target_kind       = {target_kind.value}\n"
        f"  backend           = {BACKEND_METHOD}\n"
        f"  twirling          = {'enabled' if twirling.enabled else 'disabled'}\n"
        f"  subset_fraction   = {twirling.subset_fraction:.2f}\n"
        f"  subset_mode       = {twirling.subset_mode}\n"
        f"  subset_seed       = {twirling.subset_seed}\n"
        f"  iterative_mode    = {args.iterative}\n"
        f"  out_dir           = {out_dir}\n" +
        "=" * 70
    )

    started   = time.time()
    total_runs  = len(noises) * len(Ms) * len(Ns) * len(ps) * len(Ls)
    current_run = 0

    for noise in noises:
        for M in Ms:
            target = TargetSpec(
                M=M,
                kind=target_kind,
                seed=args.seed,
                product_theta=args.theta,
                product_phi=args.phi,
            )
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

                        logger.info(f"\n{'=' * 70}")
                        logger.info(f"Run {current_run}/{total_runs}: {tag}  (ℓ={ell})")
                        logger.info(f"{'=' * 70}")

                        t0 = time.time()
                        try:
                            run_and_save(spec)
                            logger.info(f"✓ Completed in {time.time() - t0:.1f}s\n")
                        except Exception as e:
                            logger.error(f"✗ ERROR during {tag}: {e}\n", exc_info=True)

    total = time.time() - started
    logger.info(f"\n{'=' * 70}")
    logger.info(f"VD grid sweep complete!")
    logger.info(f"  Total time : {total / 60:.1f} min")
    logger.info(f"  Runs       : {current_run}/{total_runs}")
    logger.info(f"  Data saved : {out_dir}")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()