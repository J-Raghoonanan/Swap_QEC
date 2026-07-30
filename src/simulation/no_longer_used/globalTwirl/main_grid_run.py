"""
GlobalTwirl grid sweep runner.

MINIMAL CHANGE POLICY:
- Keep same RunSpec/TargetSpec/NoiseSpec usage.
- Keep same columns/headers produced by the runner.
- Only difference vs the original grid script:
    * Default out dir is data/globalTwirl_simulations
    * Uses globalTwirl.streaming_runner.run_and_save
    * Provides same CLI flags you already used.
    
    python -m src.simulation.globalTwirl.main_grid_run \                            
  --out data/globalTwirl_simulations \
  --noise z \
  --m-values 1 5 \
  --iterative
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import List

import numpy as np

from ..moreNoise.configs import (
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------
# Defaults (match your original style)
# -----------------------------

M_LIST: List[int] = [1, 2, 3, 4, 5]
N_LIST: List[int] = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Keep your dense grid (including 0.71..0.79) if you want it; you can prune in plotting later.
P_LIST: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

NOISES: List[NoiseType] = [NoiseType.depolarizing, NoiseType.dephase_z]
TARGET_KIND: StateKind = StateKind.hadamard  # change to StateKind.haar for random pure states
# TARGET_KIND: StateKind = StateKind.single_qubit_product
BACKEND_METHOD: str = "density_matrix"

# sweep ℓ
L_LIST: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

AA = AASpec(target_success=0.99, max_iters=32, use_postselection_only=False)
TWIRLING = TwirlingSpec(enabled=True, mode="cyclic", seed=None)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid sweep for SWAP-QEC circuit simulation (GlobalTwirl)")
    p.add_argument("--out", type=Path, default=Path("data/globalTwirl_simulations"), help="Output directory for CSVs")
    p.add_argument("--max-m", type=int, default=6, help="Maximum M to include (≤ 6 recommended)")
    p.add_argument("--m-values", type=int, nargs="+", help="Specific M values to run (e.g., --m-values 1 3 5)")
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
        help="Enable iterative noise mode (GlobalTwirl behavior is implemented there)",
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


def main() -> None:
    args = _parse_args()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    noises = _pick_noises(args.noise)
    mode = NoiseMode(args.mode)
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
        logger.info("QUICK TEST MODE: Using reduced parameter space")
        Ms = Ms[:2] if len(Ms) > 2 else Ms
        Ns = [4, 16, 64]
        ps = [0.1, 0.5, 0.9]
        Ls = [0, 1]
    else:
        Ns = N_LIST
        ps = P_LIST
        Ls = L_LIST

    twirling = TwirlingSpec(enabled=not args.no_twirl, mode="cyclic", seed=args.seed)

    logger.info(
        "=" * 70
        + "\n"
        "Running GLOBAL-TWIRL grid sweep with:\n"
        f"  Ms           = {Ms}\n"
        f"  Ns           = {Ns}\n"
        f"  ps           = {ps}\n"
        f"  ℓ values      = {Ls}\n"
        f"  noises       = {[n.value for n in noises]}\n"
        f"  mode         = {mode.value}\n"
        f"  target_kind  = {target_kind.value}\n"
        f"  backend      = {BACKEND_METHOD}\n"
        f"  twirling     = {'enabled' if twirling.enabled else 'disabled'}\n"
        f"  iterative    = {bool(args.iterative)}\n"
        f"  out_dir      = {out_dir}\n"
        + "=" * 70
    )

    started = time.time()
    total_runs = len(noises) * len(Ms) * len(Ns) * len(ps) * len(Ls)
    current_run = 0

    for noise in noises:
        for M in Ms:
            # Keep target generation identical to your working setup
            target = TargetSpec(
                M=M,
                kind=target_kind,
                seed=args.seed,
                product_theta=np.pi / 3,
                product_phi=np.pi / 4,
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
                        logger.info(f"\n{'='*70}")
                        logger.info(f"Run {current_run}/{total_runs}: {tag} (ℓ={ell})")
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
    logger.info("Grid sweep complete!")
    logger.info(f"  Total time: {total/60:.1f} min")
    logger.info(f"  Runs: {current_run}/{total_runs}")
    logger.info(f"  Data saved to: {out_dir}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
