"""
Noise models for streaming QEC protocol.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union
from src.quantum_states import PurityParameterState, BlochVectorState, pure_state_to_bloch_vector


class NoiseModel(ABC):
    """Abstract base class for noise models."""
    
    @abstractmethod
    def apply_noise(self, pure_state: np.ndarray) -> Union[PurityParameterState, BlochVectorState]:
        """Apply noise to a pure state."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get descriptive name for the noise model."""
        pass


class DepolarizingNoise(NoiseModel):
    """Depolarizing noise: ρ = (1-δ)|ψ⟩⟨ψ| + δI/d"""
    
    def __init__(self, dimension: int, noise_strength: float):
        self.dimension = dimension
        self.noise_strength = noise_strength
        
        if not 0 <= noise_strength <= 1:
            raise ValueError(f"Noise strength must be in [0,1], got {noise_strength}")
    
    def apply_noise(self, pure_state: np.ndarray) -> PurityParameterState:
        """Apply depolarizing noise returning purity parameter state."""
        purity = 1 - self.noise_strength
        return PurityParameterState(purity, self.dimension, pure_state)
    
    def get_name(self) -> str:
        return f"Depolarizing_d{self.dimension}_delta{self.noise_strength:.3f}"


class PauliNoise(NoiseModel):
    """Pauli noise for qubits using proper Bloch vector evolution."""
    
    def __init__(self, px: float, py: float, pz: float):
        self.px, self.py, self.pz = px, py, pz
        
        # Validate probabilities
        total = px + py + pz
        if total > 1:
            raise ValueError(f"Total Pauli error probability {total:.3f} > 1")
        
        self.p_identity = 1 - total
    
    def apply_noise(self, pure_state: np.ndarray) -> BlochVectorState:
        """Apply Pauli noise returning Bloch vector state."""
        if len(pure_state) != 2:
            raise ValueError("Pauli noise only supports qubits")
        
        # Convert pure state to Bloch vector
        target_bloch = pure_state_to_bloch_vector(pure_state)
        
        # Apply Pauli error evolution
        noisy_bloch = self._apply_pauli_errors(target_bloch)
        
        error_rates = {'px': self.px, 'py': self.py, 'pz': self.pz}
        
        return BlochVectorState(noisy_bloch, target_bloch, pure_state, error_rates)
    
    def _apply_pauli_errors(self, bloch_vector: np.ndarray) -> np.ndarray:
        """
        Apply Pauli errors to Bloch vector.
        
        Pauli operator actions on Bloch vector:
        - I: (rx, ry, rz) → (rx, ry, rz)
        - X: (rx, ry, rz) → (rx, -ry, -rz)  
        - Y: (rx, ry, rz) → (-rx, ry, -rz)
        - Z: (rx, ry, rz) → (-rx, -ry, rz)
        """
        rx, ry, rz = bloch_vector
        
        # Correct evolution under Pauli errors
        noisy_rx = (self.p_identity * rx +      # Identity preserves
                   self.px * rx +               # X preserves rx
                   self.py * (-rx) +            # Y flips rx
                   self.pz * (-rx))             # Z flips rx
        
        noisy_ry = (self.p_identity * ry +      # Identity preserves  
                   self.px * (-ry) +            # X flips ry
                   self.py * ry +               # Y preserves ry
                   self.pz * (-ry))             # Z flips ry
        
        noisy_rz = (self.p_identity * rz +      # Identity preserves
                   self.px * (-rz) +            # X flips rz
                   self.py * (-rz) +            # Y flips rz  
                   self.pz * rz)                # Z preserves rz
        
        return np.array([noisy_rx, noisy_ry, noisy_rz])
    
    def get_name(self) -> str:
        return f"Pauli_px{self.px:.3f}_py{self.py:.3f}_pz{self.pz:.3f}"


class PureDephasing(PauliNoise):
    """Pure dephasing noise (only Z errors)."""
    
    def __init__(self, pz: float):
        super().__init__(px=0.0, py=0.0, pz=pz)
    
    def get_name(self) -> str:
        return f"PureDephasing_pz{self.pz:.3f}"


class PureBitFlip(PauliNoise):
    """Pure bit flip noise (only X errors)."""
    
    def __init__(self, px: float):
        super().__init__(px=px, py=0.0, pz=0.0)
    
    def get_name(self) -> str:
        return f"PureBitFlip_px{self.px:.3f}"


class SymmetricPauli(PauliNoise):
    """Symmetric Pauli noise with equal X, Y, Z error rates."""
    
    def __init__(self, p_total: float):
        p_each = p_total / 3
        super().__init__(px=p_each, py=p_each, pz=p_each)
    
    def get_name(self) -> str:
        return f"SymmetricPauli_p{self.px*3:.3f}"