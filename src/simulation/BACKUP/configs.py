"""
Configuration and typed specs for the SWAP-based purification circuit simulator.

This module centralizes all public configuration knobs so the other modules can
remain focused on logic. It also exposes a few small helpers to map between the
manuscript's physical error rate δ and the Kraus/channel probabilities p used
in common circuit noise models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Literal

# Optional type hints that may be useful for users passing manual states
try:  # keep imports lightweight; module can be imported without qiskit installed
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
except Exception:  # pragma: no cover - typing fallback only
    QuantumCircuit = object  # type: ignore
    Statevector = object  # type: ignore


class NoiseType(str, Enum):
    depolarizing = "depolarizing"
    dephase_z = "dephase_z"
    dephase_x = "dephase_x"


class NoiseMode(str, Enum):
    """How to apply noise to each input copy before purification."""

    iid_p = "iid_p"  # apply a CPTP channel with per-qubit probability p
    exact_k = "exact_k"  # deterministically inject exactly k single-qubit errors


class StateKind(str, Enum):
    manual = "manual"  # user-provided circuit or statevector
    haar = "haar"  # Haar-random pure state
    random_circuit = "random_circuit"  # shallow random circuit
    hadamard = "hadamard"  # (H|0>)^{\otimes M}
    ghz = "ghz"  # (|0...0> + |1...1>)/sqrt(2)


# ---------------------
# δ <-> p conversions
# ---------------------

def delta_to_kraus_p(noise: NoiseType, delta: float) -> float:
    """Map manuscript's physical error rate δ to the channel probability p.

    For qubit depolarizing channel with Kraus E0 = sqrt(1-p) I, Ej = sqrt(p/3) σj,
    the manuscript uses δ = 4p/3 → p = 3δ/4.

    For pure dephasing (Z) or X-flip dephasing channels we take p = δ.
    """
    if delta < 0 or delta > 1:
        raise ValueError("delta must be in [0, 1]")
    if noise == NoiseType.depolarizing:
        return 0.75 * float(delta)
    else:
        return float(delta)


def kraus_p_to_delta(noise: NoiseType, p: float) -> float:
    if p < 0 or p > 1:
        raise ValueError("p must be in [0, 1]")
    if noise == NoiseType.depolarizing:
        return (4.0 / 3.0) * float(p)
    else:
        return float(p)


# ---------------------
# Config dataclasses
# ---------------------


@dataclass
class TargetSpec:
    """Specification of the target pure state |ψ⟩ on M qubits."""

    M: int
    kind: StateKind = StateKind.haar
    # Provide either a circuit or a statevector when kind == manual
    manual_circuit: Optional[QuantumCircuit] = None
    manual_statevector: Optional[Statevector] = None
    # random circuit options
    random_layers: int = 3
    seed: Optional[int] = None


@dataclass
class NoiseSpec:
    """Specification for how to generate noisy input copies ρ from |ψ⟩."""

    noise_type: NoiseType = NoiseType.depolarizing
    mode: NoiseMode = NoiseMode.iid_p
    # Physical error rate δ (primary knob for manuscripts/figures)
    p: float = 0.1
    # exact-k mode controls
    exact_k: int = 0  # number of single-qubit errors to inject deterministically
    
    def kraus_p(self) -> float:
        """Return the channel probability p directly."""
        return float(self.p)
    
    def manuscript_delta(self) -> float:
        """Convert to manuscript's δ parameter if needed for comparison."""
        if self.noise_type == NoiseType.depolarizing:
            return (4.0 / 3.0) * self.p  # δ = (4/3)p
        else:
            return self.p  # For dephasing, δ = p
    
    # Remove the old conversion functions since we work with p directly  
    # # Optional explicit p override; if None we derive it from δ
    # p_override: Optional[float] = None

    # def kraus_p(self) -> float:
    #     if self.p_override is not None:
    #         return float(self.p_override)
    #     return delta_to_kraus_p(self.noise_type, self.delta)


@dataclass
class AASpec:
    """Amplitude amplification controls."""

    target_success: float = 0.99  # desired Pr[anc=0] after AA
    max_iters: int = 64  # hard cap to keep circuits reasonable
    use_postselection_only: bool = False  # skip AA and just postselect anc=0 in analysis


@dataclass
class TwirlingSpec:
    """Clifford twirling configuration for dephasing noise mitigation."""
    
    enabled: bool = True  # Auto-enable for dephasing noise types
    mode: Literal["random", "cyclic"] = "cyclic"  # random or deterministic cycle
    seed: Optional[int] = None  # for reproducibility in random mode


@dataclass
class RunSpec:
    """Top-level run configuration for a streaming purification experiment."""

    target: TargetSpec
    noise: NoiseSpec
    aa: AASpec
    twirling: TwirlingSpec = field(default_factory=lambda: TwirlingSpec())
    # Total number of noisy copies to stream
    N: int = 16
    # Backend configuration
    backend_method: Literal[
        "density_matrix",
        "statevector",
        "matrix_product_state",
        "automatic",
    ] = "density_matrix"
    # Output directory for CSVs
    out_dir: Path = field(default_factory=lambda: Path("data/simulations"))
    # Optional run identifier; if empty we will synthesize a descriptive one
    run_id: Optional[str] = None
    # Verbosity for logging
    verbose: bool = False
    
    # For IBMQ only
    manual_noise: bool = False
    manual_noise_mode: str = "identical"  # one of {"identical", "twirled"}
        
    def validate(self) -> None:
        if self.target.M <= 0:
            raise ValueError("M must be positive")
        if not (0.0 <= self.noise.p <= 1.0):
            raise ValueError("p must be in [0,1]")
        if self.noise.mode == NoiseMode.exact_k:
            if self.noise.exact_k < 0:
                raise ValueError("exact_k must be >= 0")
            if self.noise.exact_k > self.target.M:
                raise ValueError(f"exact_k ({self.noise.exact_k}) cannot exceed M ({self.target.M}) for single-qubit faults")
        if not (0.0 < self.aa.target_success <= 1.0):
            raise ValueError("target_success must be in (0,1]")

    def synthesize_run_id(self) -> str:
        if self.run_id:
            return self.run_id
        parts = [
            f"M{self.target.M}",
            f"N{self.N}",
            self.noise.noise_type.value,
            self.noise.mode.value,
            f"p{self.noise.p:.5f}",
        ]
        if self.noise.mode == NoiseMode.exact_k:
            parts.append(f"k{self.noise.exact_k}")
        if self._should_apply_twirling():
            parts.append("twirl")
        return "_".join(parts)
    
    def _should_apply_twirling(self) -> bool:
        """Determine if Clifford twirling should be applied for this run."""
        if not self.twirling.enabled:
            return False
        # Auto-enable for dephasing noise types
        return self.noise.noise_type in [NoiseType.dephase_z, NoiseType.dephase_x]


__all__ = [
    "NoiseType",
    "NoiseMode",
    "StateKind",
    "TargetSpec",
    "NoiseSpec",
    "AASpec",
    "TwirlingSpec",
    "RunSpec"
]