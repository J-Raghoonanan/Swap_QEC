"""
GlobalTwirl variant package.

This package reuses the core simulator modules (configs/state_factory/noise_engine/amplified_swap)
and swaps in a runner + grid script that implement "cheap global twirling" in ITERATIVE mode:
one single-qubit Clifford applied to ALL qubits per iteration, cycling over {I, H, HS}.
"""

from .streaming_runner import run_streaming, run_and_save

__all__ = ["run_streaming", "run_and_save"]
