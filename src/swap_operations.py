"""
Core swap test operations and amplitude amplification for streaming protocol.
Updated swap test operations implementing exact Pauli formulas from Section II.E
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
    """Handles swap test operations with exact Pauli error formulas from Section II.E."""
    
    def __init__(self, max_amplification_iterations: int = 100):
        self.max_amplification_iterations = max_amplification_iterations
    
    def calculate_success_probability(self, state1: QuantumState, state2: QuantumState) -> float:
        """Calculate exact success probability for swap test."""
        if isinstance(state1, PurityParameterState) and isinstance(state2, PurityParameterState):
            return self._depolarizing_success_probability(state1, state2)
        elif isinstance(state1, BlochVectorState) and isinstance(state2, BlochVectorState):
            return self._pauli_success_probability_exact(state1, state2)
        else:
            raise ValueError("States must be of compatible types")
    
    def _depolarizing_success_probability(self, state1: PurityParameterState, 
                                        state2: PurityParameterState) -> float:
        """Success probability for depolarizing noise: P = 1/2(1 + Tr(ρ²))"""
        if state1.dimension != state2.dimension:
            raise ValueError("States must have same dimension")
        
        # For identical depolarized states with purity λ = 1-δ
        purity = state1.purity_parameter
        d = state1.dimension
        
        # Tr(ρ²) = (1-δ)² + δ(2-δ)/d from manuscript Eq. (9)
        delta = 1 - purity
        tr_rho_squared = purity**2 + delta * (2 - delta) / d
        
        return 0.5 * (1 + tr_rho_squared)
    
    def _pauli_success_probability_exact(self, state1: BlochVectorState, 
                                       state2: BlochVectorState) -> float:
        """
        Exact success probability for Pauli errors using Eq. (41) from manuscript.
        
        P_success = 1/2[2 - 2p_total + p²_total + Σp²_i]
        
        Note: Success probability depends only on error rate distribution,
        not the initial state itself (key insight from Section II.E).
        """
        if not state1.error_rates or not state2.error_rates:
            raise ValueError("Pauli states must have error_rates specified")
        
        # Extract error rates
        px = state1.error_rates['px']
        py = state1.error_rates['py'] 
        pz = state1.error_rates['pz']
        p_total = state1.error_rates['p_total']
        
        # From Eq. (40): Tr(ρ²) = 1 - 2p_total + p²_total + Σp²_i
        sum_squared_errors = px**2 + py**2 + pz**2
        tr_rho_squared = 1 - 2*p_total + p_total**2 + sum_squared_errors
        
        # From Eq. (41): P_success = 1/2[2 - 2p_total + p²_total + Σp²_i]
        return 0.5 * (2 - 2*p_total + p_total**2 + sum_squared_errors)
    
    def amplitude_amplified_swap(self, state1: QuantumState, state2: QuantumState) -> SwapResult:
        """Perform amplitude-amplified swap test."""
        # Calculate initial success probability
        p_success = self.calculate_success_probability(state1, state2)
        
        # Calculate optimal amplification iterations
        iterations = self._calculate_optimal_iterations(p_success)
        
        # Apply amplification (simulated deterministic outcome)
        amplified_probability = self._apply_amplitude_amplification(p_success, iterations)
        
        # Compute output state
        if isinstance(state1, PurityParameterState):
            output_state = self._compute_depolarizing_output(state1, state2)
        elif isinstance(state1, BlochVectorState):
            output_state = self._compute_pauli_output_exact(state1, state2)
        else:
            raise ValueError("Unsupported state type")
        
        # Estimate gate count (swap + amplification rounds)
        gate_count = 4 + 4 * iterations  # Base swap + amplification iterations
        
        return SwapResult(
            output_state=output_state,
            success_probability=amplified_probability,
            amplification_iterations=iterations,
            total_gate_count=gate_count
        )
    
    def _calculate_optimal_iterations(self, p_success: float) -> int:
        """Calculate optimal number of amplitude amplification iterations."""
        if p_success >= 1.0:
            return 0
        
        theta = 2 * np.arcsin(np.sqrt(p_success))
        return max(0, int(np.floor(np.pi / (4 * np.arcsin(np.sqrt(p_success))) - 0.5)))
    
    def _apply_amplitude_amplification(self, initial_prob: float, iterations: int) -> float:
        """Apply amplitude amplification returning final success probability."""
        if initial_prob >= 1.0:
            return 1.0
        
        theta = 2 * np.arcsin(np.sqrt(initial_prob))
        final_amplitude = np.sin((iterations + 0.5) * theta)
        
        return min(1.0, final_amplitude**2)
    
    def _compute_depolarizing_output(self, state1: PurityParameterState, 
                                   state2: PurityParameterState) -> PurityParameterState:
        """Compute output for depolarizing noise using exact purity transformation."""
        # Assuming identical states
        λ = state1.purity_parameter
        d = state1.dimension
        
        # Exact transformation from manuscript Eq. (15)
        numerator = λ * (1 + λ + 2 * (1 - λ) / d)
        denominator = 1 + λ**2 + (1 - λ**2) / d
        
        output_purity = numerator / denominator
        
        return PurityParameterState(output_purity, d, state1.target_vector)
    
    def _compute_pauli_output_exact(self, state1: BlochVectorState, 
                                  state2: BlochVectorState) -> BlochVectorState:
        """
        Compute exact output for Pauli errors using renormalization maps from Section II.E.
        
        Implements noise-specific renormalization transformations:
        - Z-dephasing: Anisotropic scaling with (1-2p_z) factors
        - X-bit-flip: Similar but affects different components
        - Symmetric Pauli: Uniform (1-4p/3) scaling
        """
        current_bloch = state1.bloch_vector.copy()
        target_bloch = state1.target_bloch_vector.copy()
        error_rates = state1.error_rates
        
        px = error_rates['px']
        py = error_rates['py']
        pz = error_rates['pz']
        
        # Determine dominant error type and apply appropriate renormalization
        if px == 0 and py == 0 and pz > 0:
            # Pure Z-dephasing case
            output_bloch = self._apply_z_dephasing_renormalization(current_bloch, pz)
        elif py == 0 and pz == 0 and px > 0:
            # Pure X-bit-flip case  
            output_bloch = self._apply_x_bitflip_renormalization(current_bloch, px)
        elif px == py == pz:
            # Symmetric Pauli case
            output_bloch = self._apply_symmetric_pauli_renormalization(current_bloch, px)
        else:
            # General case - use composite renormalization
            output_bloch = self._apply_general_pauli_renormalization(current_bloch, px, py, pz)
        
        return BlochVectorState(output_bloch, target_bloch, state1.target_pure_state, error_rates)
    
    def _apply_z_dephasing_renormalization(self, r: np.ndarray, pz: float) -> np.ndarray:
        """
        Apply Z-dephasing renormalization from Eq. (43).
        
        R_Z: (r_x,r_y,r_z) ↦ 4*r^noisy / (3 + |r^noisy|²)
        where r^noisy = ((1-2p_z)r_x, (1-2p_z)r_y, r_z)
        """
        rx, ry, rz = r
        
        # Apply Z-dephasing transformation to get noisy Bloch vector
        r_noisy_x = (1 - 2*pz) * rx
        r_noisy_y = (1 - 2*pz) * ry
        r_noisy_z = rz  # Z component unchanged by Z-dephasing
        
        r_noisy = np.array([r_noisy_x, r_noisy_y, r_noisy_z])
        r_noisy_magnitude_squared = np.sum(r_noisy**2)
        
        # Apply geometric renormalization
        renormalization_factor = 4 / (3 + r_noisy_magnitude_squared)
        
        return renormalization_factor * r_noisy
    
    def _apply_x_bitflip_renormalization(self, r: np.ndarray, px: float) -> np.ndarray:
        """Apply X-bit-flip renormalization (analogous to Z-dephasing)."""
        rx, ry, rz = r
        
        # X-bit-flip affects Y and Z components
        r_noisy_x = rx  # X component unchanged by X-bit-flip
        r_noisy_y = (1 - 2*px) * ry  
        r_noisy_z = (1 - 2*px) * rz
        
        r_noisy = np.array([r_noisy_x, r_noisy_y, r_noisy_z])
        r_noisy_magnitude_squared = np.sum(r_noisy**2)
        
        renormalization_factor = 4 / (3 + r_noisy_magnitude_squared)
        
        return renormalization_factor * r_noisy
    
    def _apply_symmetric_pauli_renormalization(self, r: np.ndarray, p: float) -> np.ndarray:
        """Apply symmetric Pauli renormalization with uniform (1-4p/3) scaling."""
        # For symmetric Pauli: r^noisy = (1-4p/3)*r
        coherence_factor = 1 - 4*p/3
        r_noisy = coherence_factor * r
        
        r_noisy_magnitude_squared = np.sum(r_noisy**2)
        renormalization_factor = 4 / (3 + r_noisy_magnitude_squared)
        
        return renormalization_factor * r_noisy
    
    def _apply_general_pauli_renormalization(self, r: np.ndarray, px: float, py: float, pz: float) -> np.ndarray:
        """
        Apply general Pauli renormalization using exact transformations from Eqs. (32)-(34).
        
        This handles the general case where px ≠ py ≠ pz.
        """
        rx, ry, rz = r
        
        # Apply exact Pauli transformations from manuscript
        r_noisy_x = (1 - 2*py - 2*pz) * rx  # Eq. (32)
        r_noisy_y = (1 - 2*px - 2*pz) * ry  # Eq. (33)
        r_noisy_z = (1 - 2*px - 2*py) * rz  # Eq. (34)
        
        r_noisy = np.array([r_noisy_x, r_noisy_y, r_noisy_z])
        r_noisy_magnitude_squared = np.sum(r_noisy**2)
        
        # Apply geometric renormalization: r^(n+1) = 4*r^noisy / (3 + |r^noisy|²)
        renormalization_factor = 4 / (3 + r_noisy_magnitude_squared)
        
        return renormalization_factor * r_noisy


def theoretical_purity_evolution_depolarizing(initial_purity: float, dimension: int, num_levels: int) -> list:
    """Compute theoretical purity evolution for depolarizing noise."""
    purities = [initial_purity]
    current_purity = initial_purity
    
    for _ in range(num_levels):
        # Apply purity transformation
        numerator = current_purity * (1 + current_purity + 2*(1-current_purity)/dimension)
        denominator = 1 + current_purity**2 + (1-current_purity**2)/dimension
        current_purity = numerator / denominator
        purities.append(current_purity)
    
    return purities


def theoretical_bloch_evolution_pauli(initial_bloch: np.ndarray, px: float, py: float, pz: float, 
                                    num_levels: int) -> Tuple[list, list]:
    """
    Compute theoretical Bloch vector evolution for Pauli errors.
    
    Returns:
        bloch_evolution: List of Bloch vectors at each level
        magnitude_evolution: List of Bloch vector magnitudes
    """
    bloch_evolution = [initial_bloch.copy()]
    magnitude_evolution = [np.linalg.norm(initial_bloch)]
    
    current_bloch = initial_bloch.copy()
    
    for level in range(num_levels):
        # Apply exact Pauli transformations (Eqs. 32-34)
        rx, ry, rz = current_bloch
        
        r_noisy_x = (1 - 2*py - 2*pz) * rx
        r_noisy_y = (1 - 2*px - 2*pz) * ry  
        r_noisy_z = (1 - 2*px - 2*py) * rz
        
        r_noisy = np.array([r_noisy_x, r_noisy_y, r_noisy_z])
        r_noisy_magnitude_squared = np.sum(r_noisy**2)
        
        # Apply renormalization: r^(n+1) = 4*r^noisy / (3 + |r^noisy|²)
        renormalization_factor = 4 / (3 + r_noisy_magnitude_squared)
        current_bloch = renormalization_factor * r_noisy
        
        bloch_evolution.append(current_bloch.copy())
        magnitude_evolution.append(np.linalg.norm(current_bloch))
    
    return bloch_evolution, magnitude_evolution


def analyze_noise_model_dependence(initial_bloch: np.ndarray, num_levels: int = 5) -> dict:
    """
    Analyze how different noise types affect purification performance.
    
    Demonstrates why purification preferentially corrects depolarizing over dephasing errors.
    """
    results = {}
    
    # Test different noise scenarios
    noise_scenarios = {
        'pure_z_dephasing': {'px': 0.0, 'py': 0.0, 'pz': 0.3},
        'pure_x_bitflip': {'px': 0.3, 'py': 0.0, 'pz': 0.0},
        'symmetric_pauli': {'px': 0.1, 'py': 0.1, 'pz': 0.1},
        'mixed_pauli': {'px': 0.1, 'py': 0.05, 'pz': 0.15}
    }
    
    for scenario_name, rates in noise_scenarios.items():
        bloch_evo, mag_evo = theoretical_bloch_evolution_pauli(
            initial_bloch, rates['px'], rates['py'], rates['pz'], num_levels)
        
        # Calculate logical error evolution
        logical_errors = []
        for bloch in bloch_evo:
            # Logical error = 1/2 * |r_final - r_target|
            target_bloch = initial_bloch / np.linalg.norm(initial_bloch)  # Normalize target
            logical_error = 0.5 * np.linalg.norm(bloch - target_bloch)
            logical_errors.append(logical_error)
        
        results[scenario_name] = {
            'bloch_evolution': bloch_evo,
            'magnitude_evolution': mag_evo,
            'logical_error_evolution': logical_errors,
            'error_rates': rates,
            'final_logical_error': logical_errors[-1],
            'error_reduction_ratio': logical_errors[-1] / logical_errors[0] if logical_errors[0] > 0 else 0
        }
    
    return results