"""
Main streaming purification protocol implementation.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Union, Tuple
from src.quantum_states import QuantumState, PurityParameterState, BlochVectorState, generate_random_pure_state
from src.noise_models import NoiseModel, DepolarizingNoise, PauliNoise
from src.swap_operations import SwapTestProcessor, SwapResult


@dataclass
class PurificationResult:
    """Result from streaming purification protocol."""
    initial_state: QuantumState
    final_state: QuantumState
    logical_error_evolution: List[float]
    fidelity_evolution: List[float]
    purity_evolution: List[float]
    total_swap_operations: int
    total_amplification_iterations: int
    memory_levels_used: int


class StreamingPurificationProtocol:
    """Main streaming purification protocol using recursive swap tests."""
    
    def __init__(self, max_amplification_iterations: int = 100):
        self.swap_processor = SwapTestProcessor(max_amplification_iterations)
        self.total_operations = 0
        self.total_amplification_iterations = 0
    
    def purify_stream(self, initial_error_rate: float, 
                     noise_model: NoiseModel,
                     num_input_states: int = 16,
                     target_state: np.ndarray = None) -> PurificationResult:
        """
        Main streaming purification using recursive swap tests.
        
        Args:
            initial_error_rate: Physical error rate (δ for depolarizing, p_total for Pauli)
            noise_model: Noise model to apply
            num_input_states: Number of initial noisy states (should be power of 2)
            target_state: Target pure state (random if None)
        """
        # Ensure we have a power of 2 for binary tree
        if num_input_states & (num_input_states - 1) != 0:
            num_input_states = 2 ** int(np.log2(num_input_states))
            print(f"Adjusted to {num_input_states} states for binary tree")
        
        # Generate target state if not provided
        if target_state is None:
            dimension = getattr(noise_model, 'dimension', 2)
            target_state = generate_random_pure_state(dimension)
        
        # Generate initial noisy states
        initial_states = []
        for _ in range(num_input_states):
            noisy_state = noise_model.apply_noise(target_state)
            initial_states.append(noisy_state)
        
        # Track evolution
        logical_errors = [initial_states[0].get_logical_error()]
        fidelities = [initial_states[0].get_fidelity_with_target()]
        purities = [initial_states[0].get_purity_parameter()]
        
        # Reset counters
        self.total_operations = 0
        self.total_amplification_iterations = 0
        
        # Perform recursive purification
        final_state, levels_used = self._recursive_purification(
            initial_states, logical_errors, fidelities, purities)
        
        return PurificationResult(
            initial_state=initial_states[0],
            final_state=final_state,
            logical_error_evolution=logical_errors,
            fidelity_evolution=fidelities,
            purity_evolution=purities,
            total_swap_operations=self.total_operations,
            total_amplification_iterations=self.total_amplification_iterations,
            memory_levels_used=levels_used
        )
    
    def _recursive_purification(self, states: List[QuantumState],
                              logical_errors: List[float],
                              fidelities: List[float],
                              purities: List[float]) -> Tuple[QuantumState, int]:
        """
        Recursive binary tree purification using swap tests.
        
        This is the core streaming algorithm - applies swap tests in binary tree pattern.
        """
        if len(states) == 1:
            return states[0], 0
        
        # Apply swap tests to pairs of states
        next_level_states = []
        
        for i in range(0, len(states), 2):
            state1, state2 = states[i], states[i + 1]
            
            # Perform amplitude-amplified swap test
            swap_result = self.swap_processor.amplitude_amplified_swap(state1, state2)
            next_level_states.append(swap_result.output_state)
            
            # Update counters
            self.total_operations += 1
            self.total_amplification_iterations += swap_result.amplification_iterations
        
        # Track evolution at this level
        representative_state = next_level_states[0]
        logical_errors.append(representative_state.get_logical_error())
        fidelities.append(representative_state.get_fidelity_with_target())
        purities.append(representative_state.get_purity_parameter())
        
        # Recurse to next level
        final_state, deeper_levels = self._recursive_purification(
            next_level_states, logical_errors, fidelities, purities)
        
        return final_state, deeper_levels + 1
    
    def theoretical_purification_analysis(self, initial_error_rate: float,
                                        dimension: int = 2,
                                        num_levels: int = 4,
                                        noise_type: str = 'depolarizing') -> Tuple[List[float], List[float]]:
        """
        Theoretical analysis of purification without running full protocol.
        Useful for quick threshold analysis.
        """
        if noise_type == 'depolarizing':
            # For depolarizing noise, use exact purity evolution
            initial_purity = 1 - initial_error_rate
            purities = [initial_purity]
            
            for _ in range(num_levels):
                λ = purities[-1]
                d = dimension
                
                # Exact purity transformation for depolarizing noise
                numerator = λ * (1 + λ + 2 * (1 - λ) / d)
                denominator = 1 + λ**2 + (1 - λ**2) / d
                
                new_purity = numerator / denominator
                purities.append(new_purity)
            
            # Convert to logical errors
            logical_errors = [(1 - λ) * (dimension - 1) / dimension for λ in purities]
            
        else:
            # For Pauli noise, use approximate analysis
            # This is more complex and depends on specific error rates
            # For now, implement a simplified model
            # NEED TO UPDATE THIS WITH PROPER PAULI ANALYSIS
            logical_errors = [initial_error_rate]
            purities = [1 - initial_error_rate]
            
            for _ in range(num_levels):
                # Simplified Pauli purification model
                current_error = logical_errors[-1]
                # Assume some error reduction per level (rough approximation)
                reduction_factor = 0.7  # This should be derived from proper Pauli analysis
                new_error = current_error * reduction_factor
                logical_errors.append(new_error)
                purities.append(1 - new_error)
        
        return logical_errors, purities
    
    def threshold_analysis(self, error_rates: np.ndarray,
                          noise_model_factory,
                          num_levels: int = 4,
                          num_input_states: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyze threshold behavior across multiple error rates.
        
        Returns:
            error_rates: Input error rates tested
            final_logical_errors: Final logical errors achieved
        """
        final_errors = []
        
        for error_rate in error_rates:
            try:
                noise_model = noise_model_factory(error_rate)
                result = self.purify_stream(
                    initial_error_rate=error_rate,
                    noise_model=noise_model,
                    num_input_states=num_input_states
                )
                final_errors.append(result.logical_error_evolution[-1])
            except Exception as e:
                print(f"Failed at error rate {error_rate:.3f}: {e}")
                final_errors.append(float('inf'))  # Mark as failed
        
        return error_rates, np.array(final_errors)
