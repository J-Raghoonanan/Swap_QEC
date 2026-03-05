"""
Virtual Distillation (VD) simulation package with approximate Clifford twirling.

This package implements VD purification (ρ → ρ^2) with resource-efficient
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

from .virtual_distillation import (
    virtual_distill,
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
    # VD operations
    "virtual_distill",
    "purify_two_from_density",
    # Runners
    "run_iterative_purification",
    "run_and_save",
]
