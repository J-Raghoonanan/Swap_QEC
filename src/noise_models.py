"""
Updated noise models implementing the corrected Pauli error formulas from Section II.E
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
    """
    Pauli noise implementation using exact formulas from Section II.E.
    
    Implements the corrected Bloch vector transformations:
    - r^noisy_x = (1-2p_y-2p_z)r⁰_x  (Eq. 32)
    - r^noisy_y = (1-2p_x-2p_z)r⁰_y  (Eq. 33)  
    - r^noisy_z = (1-2p_x-2p_y)r⁰_z  (Eq. 34)
    """
    
    def __init__(self, px: float, py: float, pz: float):
        self.px, self.py, self.pz = px, py, pz
        self.p_total = px + py + pz
        
        # Validate probabilities
        if self.p_total > 1:
            raise ValueError(f"Total Pauli error probability {self.p_total:.3f} > 1")
        
        self.p_identity = 1 - self.p_total
    
    def apply_noise(self, pure_state: np.ndarray) -> BlochVectorState:
        """Apply Pauli noise using exact Section II.E formulas."""
        if len(pure_state) != 2:
            raise ValueError("Pauli noise only supports qubits")
        
        # Convert pure state to Bloch vector
        target_bloch = pure_state_to_bloch_vector(pure_state)
        
        # Apply exact Pauli error transformations from Eqs. (32)-(34)
        noisy_bloch = self._apply_exact_pauli_transformations(target_bloch)
        
        error_rates = {'px': self.px, 'py': self.py, 'pz': self.pz, 'p_total': self.p_total}
        
        return BlochVectorState(noisy_bloch, target_bloch, pure_state, error_rates)
    
    def _apply_exact_pauli_transformations(self, r0: np.ndarray) -> np.ndarray:
        """
        Apply exact Pauli transformations from manuscript Eqs. (32)-(34).
        
        Args:
            r0: Original Bloch vector [r⁰_x, r⁰_y, r⁰_z]
            
        Returns:
            Noisy Bloch vector after Pauli errors
        """
        r0_x, r0_y, r0_z = r0
        
        # Exact formulas from Section II.E
        r_noisy_x = (1 - 2*self.py - 2*self.pz) * r0_x  # Eq. (32)
        r_noisy_y = (1 - 2*self.px - 2*self.pz) * r0_y  # Eq. (33)
        r_noisy_z = (1 - 2*self.px - 2*self.py) * r0_z  # Eq. (34)
        
        return np.array([r_noisy_x, r_noisy_y, r_noisy_z])
    
    def get_success_probability_exact(self, target_bloch: np.ndarray) -> float:
        """
        Calculate exact success probability using Eq. (41) from manuscript.
        
        P_success = 1/2[2 - 2p_total + p²_total + Σp²_i]
        """
        # From Eq. (40): Tr(ρ²) = 1 - 2p_total + p²_total + Σp²_i
        sum_squared_errors = self.px**2 + self.py**2 + self.pz**2
        tr_rho_squared = 1 - 2*self.p_total + self.p_total**2 + sum_squared_errors
        
        # From Eq. (41)
        return 0.5 * (1 + tr_rho_squared)
    
    def get_name(self) -> str:
        return f"Pauli_px{self.px:.3f}_py{self.py:.3f}_pz{self.pz:.3f}"


class PureDephasingNoise(PauliNoise):
    """Pure Z-dephasing noise: px = py = 0."""
    
    def __init__(self, pz: float):
        super().__init__(px=0.0, py=0.0, pz=pz)
    
    def get_name(self) -> str:
        return f"PureDephasing_pz{self.pz:.3f}"


class PureBitFlipNoise(PauliNoise):
    """Pure X-bit flip noise: py = pz = 0."""
    
    def __init__(self, px: float):
        super().__init__(px=px, py=0.0, pz=0.0)
    
    def get_name(self) -> str:
        return f"PureBitFlip_px{self.px:.3f}"


class SymmetricPauliNoise(PauliNoise):
    """Symmetric Pauli noise: px = py = pz."""
    
    def __init__(self, p_symmetric: float):
        super().__init__(px=p_symmetric, py=p_symmetric, pz=p_symmetric)
    
    def get_name(self) -> str:
        return f"SymmetricPauli_p{self.px:.3f}"


# Factory functions for easy creation
def create_depolarizing_noise_factory(dimension: int):
    """Factory for depolarizing noise models."""
    def factory(noise_strength: float) -> DepolarizingNoise:
        return DepolarizingNoise(dimension, noise_strength)
    return factory


def create_pauli_noise_factory(px: float, py: float, pz: float):
    """Factory for Pauli noise models."""
    def factory(error_scale: float = 1.0) -> PauliNoise:
        return PauliNoise(px * error_scale, py * error_scale, pz * error_scale)
    return factory