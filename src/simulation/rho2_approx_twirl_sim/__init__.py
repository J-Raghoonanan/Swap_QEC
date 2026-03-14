"""
Rho2 simulation package with approximate Clifford twirling.

This package implements rho2 purification (ρ → ρ^2) with resource-efficient
approximate twirling for dephasing noise mitigation.
"""

from .configs import (
    NoiseType,
    NoiseMode,
    StateKind,
    TargetSpec,
    NoiseSpec,
    AASpec,
    TwirlingSpec,
    RunSpec
)

from .rho2_purification import (
    rho2_purification,
    purify_two_from_density,
)

from .streaming_runner import (
    run_iterative_purification,
    run_and_save,
)

__all__ = [
    # Configs
    "NoiseType",
    "NoiseMode",
    "StateKind",
    "TargetSpec",
    "NoiseSpec",
    "AASpec",
    "TwirlingSpec",
    "RunSpec",
    # rho2 operations
    "rho2_purification",
    "purify_two_from_density",
    # Runners
    "run_iterative_purification",
    "run_and_save",
]
