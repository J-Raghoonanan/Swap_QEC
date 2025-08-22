"""
Core quantum state representations for streaming QEC protocol.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any
from abc import ABC, abstractmethod


class QuantumState(ABC):
    """Abstract base class for quantum state representations."""
    
    @abstractmethod
    def get_fidelity_with_target(self) -> float:
        """Return fidelity with target pure state."""
        pass
    
    @abstractmethod
    def get_logical_error(self) -> float:
        """Return logical error metric."""
        pass
    
    @abstractmethod
    def get_purity_parameter(self) -> float:
        """Return purity parameter for this state."""
        pass


@dataclass
class PurityParameterState(QuantumState):
    """State representation for depolarizing noise using purity parameter λ."""
    purity_parameter: float  # λ ∈ [0,1]
    dimension: int
    target_vector: np.ndarray
    
    def __post_init__(self):
        if not 0 <= self.purity_parameter <= 1:
            raise ValueError(f"Purity parameter must be in [0,1], got {self.purity_parameter}")
        if self.dimension < 2:
            raise ValueError(f"Dimension must be ≥ 2, got {self.dimension}")
    
    def get_fidelity_with_target(self) -> float:
        """Fidelity for depolarized state: F = λ + (1-λ)/d"""
        return self.purity_parameter + (1 - self.purity_parameter) / self.dimension
    
    def get_logical_error(self) -> float:
        """Logical error for depolarizing noise: εL = (1-λ)(d-1)/d"""
        return (1 - self.purity_parameter) * (self.dimension - 1) / self.dimension
    
    def get_purity_parameter(self) -> float:
        return self.purity_parameter
    
    def get_density_matrix(self) -> np.ndarray:
        """Full density matrix representation."""
        target_projector = np.outer(self.target_vector, self.target_vector.conj())
        mixed_state = np.eye(self.dimension) / self.dimension
        return (self.purity_parameter * target_projector + 
                (1 - self.purity_parameter) * mixed_state)


@dataclass
class BlochVectorState(QuantumState):
    """State representation for Pauli errors using Bloch vector."""
    bloch_vector: np.ndarray  # [rx, ry, rz]
    target_bloch_vector: np.ndarray  # Target Bloch vector
    target_pure_state: np.ndarray  # Track pure state
    error_rates: Dict[str, float] = None  # px, py, pz
    
    def __post_init__(self):
        if len(self.bloch_vector) != 3:
            raise ValueError("Bloch vector must be 3-dimensional")
        if len(self.target_bloch_vector) != 3:
            raise ValueError("Target Bloch vector must be 3-dimensional")
        
        # Clip to unit sphere
        if np.linalg.norm(self.bloch_vector) > 1:
            self.bloch_vector = self.bloch_vector / np.linalg.norm(self.bloch_vector)
        '''
        Need to confirm if I should do this; the whole value of the swap purification is that it renormalizes the bloch vector, 
        so maybe I shouldn't be ensuring normalization here.
        '''
    
    def get_fidelity_with_target(self) -> float:
        """Fidelity via Bloch vector dot product."""
        return (1 + np.dot(self.bloch_vector, self.target_bloch_vector)) / 2
    
    def get_logical_error(self) -> float:
        """Logical error: half Euclidean distance between Bloch vectors."""
        return 0.5 * np.linalg.norm(self.bloch_vector - self.target_bloch_vector)
    
    def get_purity_parameter(self) -> float:
        """Purity parameter is Bloch vector magnitude for qubits."""
        return np.linalg.norm(self.bloch_vector)
    
    def get_density_matrix(self) -> np.ndarray:
        """Density matrix from Bloch vector."""
        rx, ry, rz = self.bloch_vector
        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        return 0.5 * (np.eye(2, dtype=complex) + rx * pauli_x + ry * pauli_y + rz * pauli_z)


def pure_state_to_bloch_vector(pure_state: np.ndarray) -> np.ndarray:
    """Convert pure qubit state to Bloch vector."""
    if len(pure_state) != 2:
        raise ValueError("Bloch vector conversion only supports qubits")
    
    # Normalize
    pure_state = pure_state / np.linalg.norm(pure_state)
    alpha, beta = pure_state[0], pure_state[1]
    
    # Calculate Bloch vector components
    rx = 2 * np.real(alpha * np.conj(beta))
    ry = 2 * np.imag(alpha * np.conj(beta))
    rz = np.abs(alpha)**2 - np.abs(beta)**2
    
    return np.array([rx, ry, rz])


def generate_random_pure_state(dimension: int) -> np.ndarray:
    """Generate a random pure state using Haar measure."""
    # Generate complex Gaussian random numbers
    real_part = np.random.normal(0, 1, dimension)
    imag_part = np.random.normal(0, 1, dimension)
    state = real_part + 1j * imag_part
    # Normalize
    state = state / np.linalg.norm(state)
    return state