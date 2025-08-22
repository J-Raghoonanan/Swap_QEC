"""
Core swap test operations and amplitude amplification for streaming protocol.
"""

import numpy as np
from dataclasses import dataclass
from typing import Union, Tuple
from src.quantum_states import QuantumState, PurityParameterState, BlochVectorState


@dataclass
class SwapResult:
    """Result of a swap test operation."""
    output_state: QuantumState
    success_probability: float
    amplification_iterations: int
    total_gate_count: int


class SwapTestProcessor:
    """Handles swap test operations with amplitude amplification."""
    
    def __init__(self, max_amplification_iterations: int = 100):
        self.max_amplification_iterations = max_amplification_iterations
    
    def calculate_success_probability(self, state1: QuantumState, state2: QuantumState) -> float:
        """Calculate success probability for swap test on two states."""
        if isinstance(state1, PurityParameterState) and isinstance(state2, PurityParameterState):
            return self._depolarizing_success_probability(state1, state2)
        elif isinstance(state1, BlochVectorState) and isinstance(state2, BlochVectorState):
            return self._pauli_success_probability(state1, state2)
        else:
            raise ValueError("States must be of compatible types")
    
    def _depolarizing_success_probability(self, state1: PurityParameterState, 
                                        state2: PurityParameterState) -> float:
        """Success probability for depolarizing noise: P = 1/2(1 + Tr(ρ²))"""
        if state1.dimension != state2.dimension:
            raise ValueError("States must have same dimension")
        
        # For identical depolarized states ρ(δ)
        purity = state1.purity_parameter  # Assuming identical states
        d = state1.dimension
        
        # Tr(ρ²) = (1-δ)² + δ(2-δ)/d for depolarizing noise
        tr_rho_squared = (purity**2 + 
                         (1 - purity) * (2 - (1 - purity)) / d)
        
        return 0.5 * (1 + tr_rho_squared)
    
    def _pauli_success_probability(self, state1: BlochVectorState, 
                                 state2: BlochVectorState) -> float:
        """Success probability for Pauli errors (approximate)."""
        # For identical Pauli-noisy states - simplified calculation
        bloch_magnitude = np.linalg.norm(state1.bloch_vector)
        
        # Approximate success probability based on coherence
        return 0.5 * (1 + bloch_magnitude**2)
    
    def amplitude_amplified_swap(self, state1: QuantumState, state2: QuantumState) -> SwapResult:
        """Perform amplitude-amplified swap test."""
        initial_success_prob = self.calculate_success_probability(state1, state2)
        
        if initial_success_prob >= 0.99:  # Already high probability
            output_state = self.compute_output_state(state1, state2)
            return SwapResult(output_state, initial_success_prob, 0, 4)
        
        # Calculate optimal number of amplitude amplification iterations
        theta = 2 * np.arcsin(np.sqrt(initial_success_prob))
        optimal_iterations = max(0, int(np.floor(np.pi / (2 * theta) - 0.5)))
        
        # Cap iterations for practical reasons
        num_iterations = min(optimal_iterations, self.max_amplification_iterations)
        
        # Compute amplified success probability
        amplified_prob = np.sin((2 * num_iterations + 1) * theta / 2)**2
        
        # Gate count: base swap test + amplification rounds
        gate_count = 4 + 4 * num_iterations
        
        # Compute output state (same as probabilistic case but deterministic)
        output_state = self.compute_output_state(state1, state2)
        
        return SwapResult(output_state, amplified_prob, num_iterations, gate_count)
    
    def compute_output_state(self, state1: QuantumState, state2: QuantumState) -> QuantumState:
        """Compute output state after successful swap test."""
        if isinstance(state1, PurityParameterState) and isinstance(state2, PurityParameterState):
            return self._compute_depolarizing_output(state1, state2)
        elif isinstance(state1, BlochVectorState) and isinstance(state2, BlochVectorState):
            return self._compute_pauli_output(state1, state2)
        else:
            raise ValueError("States must be of compatible types")
    
    def _compute_depolarizing_output(self, state1: PurityParameterState, 
                                   state2: PurityParameterState) -> PurityParameterState:
        """Compute output for depolarizing noise using purity transformation."""
        # Assuming identical states for simplicity
        λ = state1.purity_parameter
        d = state1.dimension
        
        # Purity transformation: λ_out = λ(1 + λ + 2(1-λ)/d) / (1 + λ² + (1-λ²)/d)
        numerator = λ * (1 + λ + 2 * (1 - λ) / d)
        denominator = 1 + λ**2 + (1 - λ**2) / d
        
        output_purity = numerator / denominator
        
        return PurityParameterState(output_purity, d, state1.target_vector)
    
    def _compute_pauli_output(self, state1: BlochVectorState, 
                            state2: BlochVectorState) -> BlochVectorState:
        """Compute output for Pauli errors using Bloch vector transformation."""
        # Simple coherence-weighted purification model
        current_bloch = state1.bloch_vector.copy()
        target_bloch = state1.target_bloch_vector.copy()
        
        if state1.error_rates:
            px = state1.error_rates.get('px', 0)
            py = state1.error_rates.get('py', 0)
            pz = state1.error_rates.get('pz', 0)
            total_error_rate = px + py + pz
            
            if total_error_rate > 0:
                # Coherence-preserving purification
                # Z errors preserve more coherence than X/Y
                z_coherence_weight = 1.0
                xy_coherence_weight = 0.5
                
                coherence_factor = (
                    (1 - total_error_rate) * 1.0 +  # No error preserves all
                    pz * z_coherence_weight +         # Z error partial preservation
                    (px + py) * xy_coherence_weight   # X,Y error minimal preservation
                )
                
                # Purification strength scales with preserved coherence
                purification_strength = 0.2 + 0.6 * coherence_factor
                
                # Apply purification
                direction_to_target = target_bloch - current_bloch
                purified_bloch = current_bloch + purification_strength * direction_to_target
            else:
                purified_bloch = current_bloch
        else:
            # Default: partial restoration toward target
            purified_bloch = 0.7 * current_bloch + 0.3 * target_bloch
        
        # Ensure unit sphere constraint
        if np.linalg.norm(purified_bloch) > 1:
            purified_bloch = purified_bloch / np.linalg.norm(purified_bloch)
        
        return BlochVectorState(purified_bloch, target_bloch, state1.target_pure_state, 
                              state1.error_rates)


def theoretical_purity_evolution(initial_purity: float, dimension: int, num_levels: int) -> list:
    """Compute theoretical purity evolution for depolarizing noise."""
    purities = [initial_purity]
    
    for _ in range(num_levels):
        λ = purities[-1]
        d = dimension
        
        # Theoretical purity transformation
        numerator = λ * (1 + λ + 2 * (1 - λ) / d)
        denominator = 1 + λ**2 + (1 - λ**2) / d
        
        new_purity = numerator / denominator
        purities.append(new_purity)
    
    return purities


def theoretical_logical_errors(purities: list, dimension: int) -> list:
    """Convert purity parameters to logical errors for depolarizing noise."""
    return [(1 - λ) * (dimension - 1) / dimension for λ in purities]