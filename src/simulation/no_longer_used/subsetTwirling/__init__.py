"""
Subset Twirling implementation for SWAP-based purification.

This package implements resource-efficient Clifford twirling that uses
only a subset of the full 3^M Clifford combinations, reducing computational
cost while maintaining effective noise mitigation.

Key differences from moreNoise implementation:
- Configurable subset_fraction parameter (0.0 to 1.0)
- Two subset selection modes: random sampling or deterministic first-k
- Explicit averaging over Clifford subset in apply_noise_to_density_matrix
"""

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
from .state_factory import build_target
from .noise_engine import build_noisy_copy, apply_noise_to_density_matrix
from .amplified_swap import purify_two_from_density
from .streaming_runner import run_streaming, run_and_save

__all__ = [
    "RunSpec",
    "TargetSpec",
    "NoiseSpec",
    "AASpec",
    "TwirlingSpec",
    "NoiseType",
    "NoiseMode",
    "StateKind",
    "build_target",
    "build_noisy_copy",
    "apply_noise_to_density_matrix",
    "purify_two_from_density",
    "run_streaming",
    "run_and_save",
]
