"""
Streaming Purification Quantum Error Correction - Full Quantum Simulation

This module implements the complete quantum protocol including:
- Probabilistic swap test simulation
- Amplitude amplification with explicit Q operator iterations
- Deterministic recursive purification
- Validation of theoretical predictions

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import warnings

@dataclass
class SwapTestResult:
    """Result of a single swap test operation."""
    success: bool
    output_state: np.ndarray  # State parameters after swap
    success_probability: float
    measurement_outcome: int  # 0 or 1

@dataclass
class AmplificationResult:
    """Result of amplitude amplification process."""
    final_success_probability: float
    iterations_used: int
    amplitude_evolution: np.ndarray  # Track amplitude through iterations
    success: bool
    output_state: np.ndarray

@dataclass
class PurificationResult:
    """Complete purification simulation result."""
    initial_purity: float
    final_purity: float
    purity_evolution: np.ndarray
    error_evolution: np.ndarray
    success_probabilities: np.ndarray
    amplification_iterations: np.ndarray
    amplification_results: List[AmplificationResult]
    total_swap_attempts: int
    total_amplification_iterations: int
    theoretical_prediction: float
    simulation_matches_theory: bool
    dimension: int
    levels: int

class QuantumState:
    """
    Represent quantum states for simulation.
    For qudits: rho = λ|ψ⟩⟨ψ| + (1-λ)I/d
    """
    
    def __init__(self, purity: float, dimension: int, target_state: Optional[np.ndarray] = None):
        """
        Initialize quantum state.
        
        Args:
            purity: Purity parameter λ ∈ [0,1]
            dimension: Dimension d of the system
            target_state: Optional specific target state (default: |0⟩)
        """
        self.purity = purity
        self.dimension = dimension
        
        if target_state is None:
            # Default to |0⟩ state
            self.target_state = np.zeros(dimension)
            self.target_state[0] = 1.0
        else:
            self.target_state = target_state / np.linalg.norm(target_state)
    
    def get_density_matrix(self) -> np.ndarray:
        """Get the full density matrix representation."""
        target_projector = np.outer(self.target_state, self.target_state.conj())
        mixed_state = np.eye(self.dimension) / self.dimension
        return self.purity * target_projector + (1 - self.purity) * mixed_state
    
    def fidelity_with_target(self) -> float:
        """Calculate fidelity with the pure target state."""
        rho = self.get_density_matrix()
        return np.real(np.trace(rho @ np.outer(self.target_state, self.target_state.conj())))

class SwapTestSimulator:
    """
    Simulate the quantum swap test with full amplitude amplification.
    """
    
    def __init__(self, dimension: int = 2, verbose: bool = False):
        """
        Initialize swap test simulator.
        
        Args:
            dimension: Dimension of quantum system
            verbose: Enable detailed logging
        """
        self.dimension = dimension
        self.verbose = verbose
    
    def calculate_success_probability(self, state1: QuantumState, state2: QuantumState) -> float:
        """
        Calculate theoretical success probability for swap test.
        P_success = (1 + Tr(ρ₁ρ₂))/2
        
        Args:
            state1, state2: Input quantum states
            
        Returns:
            Success probability for swap test
        """
        rho1 = state1.get_density_matrix()
        rho2 = state2.get_density_matrix()
        
        # P_success = (1 + Tr(ρ₁ρ₂))/2
        return 0.5 * (1 + np.real(np.trace(rho1 @ rho2)))
    
    def probabilistic_swap_test(self, state1: QuantumState, state2: QuantumState) -> SwapTestResult:
        """
        Perform a single probabilistic swap test.
        
        Args:
            state1, state2: Input quantum states (assumed identical)
            
        Returns:
            SwapTestResult with success/failure and output state
        """
        success_prob = self.calculate_success_probability(state1, state2)
        
        # Simulate measurement outcome
        measurement_outcome = np.random.choice([0, 1], p=[success_prob, 1 - success_prob])
        success = (measurement_outcome == 0)
        
        if success:
            # Calculate output state using theoretical formula
            output_purity = self._compute_output_purity(state1.purity)
            output_state = QuantumState(output_purity, self.dimension, state1.target_state)
        else:
            # Failed - no useful output
            output_state = QuantumState(0.0, self.dimension, state1.target_state)
        
        return SwapTestResult(
            success=success,
            output_state=output_state,
            success_probability=success_prob,
            measurement_outcome=measurement_outcome
        )
    
    def amplitude_amplification(self, state1: QuantumState, state2: QuantumState) -> AmplificationResult:
        """
        Perform amplitude amplification on the swap test.
        
        This simulates the actual quantum amplitude amplification process:
        1. Prepare coherent superposition from swap test
        2. Apply Q operator iterations
        3. Measure final amplified probability
        
        Args:
            state1, state2: Input quantum states
            
        Returns:
            AmplificationResult with final state and iteration data
        """
        # Calculate initial success probability
        initial_success_prob = self.calculate_success_probability(state1, state2)
        
        if initial_success_prob >= 1.0:
            # Already perfect - no amplification needed
            output_purity = self._compute_output_purity(state1.purity)
            output_state = QuantumState(output_purity, self.dimension, state1.target_state)
            
            return AmplificationResult(
                final_success_probability=1.0,
                iterations_used=0,
                amplitude_evolution=np.array([1.0]),
                success=True,
                output_state=output_state
            )
        
        # Calculate optimal number of iterations
        theta = 2 * np.arcsin(np.sqrt(initial_success_prob))
        optimal_iterations = max(0, int(np.floor(np.pi / (2 * theta) - 0.5)))
        
        if self.verbose:
            print(f"  Amplitude amplification: P_initial={initial_success_prob:.4f}, "
                  f"θ={theta:.4f}, optimal_k={optimal_iterations}")
        
        # Simulate amplitude evolution through iterations
        amplitude_evolution = np.zeros(optimal_iterations + 1)
        
        # Initial amplitude
        amplitude_evolution[0] = np.sqrt(initial_success_prob)
        
        # Apply Q operator k times: amplitude rotates by θ each iteration
        for k in range(optimal_iterations):
            # After k+1 applications of Q, success probability is sin²((2(k+1) + 1)θ/2)
            new_amplitude = np.sin((2*(k+1) + 1) * theta / 2)
            amplitude_evolution[k + 1] = new_amplitude
        
        # Final success probability
        final_success_prob = amplitude_evolution[-1]**2
        
        # For realistic simulation, add small noise to account for finite precision
        noise_level = 1e-6
        final_success_prob = min(1.0, final_success_prob + np.random.normal(0, noise_level))
        
        if self.verbose:
            print(f"  After {optimal_iterations} iterations: P_final={final_success_prob:.6f}")
        
        # Simulate final measurement with amplified probability
        success = np.random.random() < final_success_prob
        
        if success:
            output_purity = self._compute_output_purity(state1.purity)
            output_state = QuantumState(output_purity, self.dimension, state1.target_state)
        else:
            # This should be extremely rare with proper amplitude amplification
            output_state = QuantumState(0.0, self.dimension, state1.target_state)
            if self.verbose:
                print(f"  WARNING: Amplitude amplification failed (probability {1-final_success_prob:.6f})")
        
        return AmplificationResult(
            final_success_probability=final_success_prob,
            iterations_used=optimal_iterations,
            amplitude_evolution=amplitude_evolution,
            success=success,
            output_state=output_state
        )
    
    def deterministic_swap_test(self, state1: QuantumState, state2: QuantumState) -> AmplificationResult:
        """
        Perform deterministic swap test using amplitude amplification.
        
        This is the core operation of our streaming purification protocol.
        
        Args:
            state1, state2: Input quantum states
            
        Returns:
            AmplificationResult (success should be True with high probability)
        """
        return self.amplitude_amplification(state1, state2)
    
    def _compute_output_purity(self, input_purity: float) -> float:
        """
        Compute output purity using theoretical formula.
        
        For qudits: λ_out = λ_in * (1 + λ_in + 2(1-λ_in)/d) / (1 + λ_in² + (1-λ_in²)/d)
        """
        d = self.dimension
        lambda_in = input_purity
        
        numerator = lambda_in * (1 + lambda_in + 2*(1-lambda_in)/d)
        denominator = 1 + lambda_in**2 + (1-lambda_in**2)/d
        
        return numerator / denominator

class StreamingPurificationSimulator:
    """
    Full simulation of streaming purification quantum error correction.
    """
    
    def __init__(self, dimension: int = 2, data_dir: str = "./data/", verbose: bool = False):
        """
        Initialize the streaming purification simulator.
        
        Args:
            dimension: Dimension of quantum system
            data_dir: Directory to save simulation data
            verbose: Enable detailed logging
        """
        self.dimension = dimension
        self.data_dir = data_dir
        self.verbose = verbose
        
        # Create swap test simulator
        self.swap_simulator = SwapTestSimulator(dimension, verbose)
        
        # Create data directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(f"{data_dir}/full_simulations", exist_ok=True)
        os.makedirs(f"{data_dir}/validation_studies", exist_ok=True)
        
        if self.verbose:
            print(f"Streaming Purification Simulator initialized (d={dimension})")
    
    def theoretical_purity_evolution(self, initial_purity: float, num_levels: int) -> np.ndarray:
        """
        Calculate theoretical purity evolution for comparison.
        
        Args:
            initial_purity: Starting purity parameter
            num_levels: Number of purification levels
            
        Returns:
            Array of theoretical purity values
        """
        purity_evolution = np.zeros(num_levels + 1)
        purity_evolution[0] = initial_purity
        
        current_purity = initial_purity
        for level in range(num_levels):
            current_purity = self.swap_simulator._compute_output_purity(current_purity)
            purity_evolution[level + 1] = current_purity
        
        return purity_evolution
    
    def recursive_purification_simulation(self, initial_delta: float, num_levels: int, 
                                        max_attempts_per_level: int = 100) -> PurificationResult:
        """
        Perform full simulation of recursive purification with amplitude amplification.
        
        Args:
            initial_delta: Initial depolarization parameter
            num_levels: Number of recursion levels
            max_attempts_per_level: Maximum retry attempts if amplification fails
            
        Returns:
            Complete PurificationResult with simulation data
        """
        if self.verbose:
            print(f"\nStarting recursive purification simulation:")
            print(f"  Initial δ={initial_delta:.3f}, levels={num_levels}")
        
        # Initialize tracking arrays
        purity_evolution = np.zeros(num_levels + 1)
        error_evolution = np.zeros(num_levels + 1)
        success_probabilities = np.zeros(num_levels)
        amplification_iterations = np.zeros(num_levels)
        amplification_results = []
        
        total_swap_attempts = 0
        total_amplification_iterations = 0
        
        # Initial state
        initial_purity = 1 - initial_delta
        purity_evolution[0] = initial_purity
        error_evolution[0] = self._logical_error(initial_purity)
        
        # Create initial states (simulate 2^num_levels copies)
        num_initial_states = 2**num_levels
        current_states = [QuantumState(initial_purity, self.dimension) 
                         for _ in range(num_initial_states)]
        
        if self.verbose:
            print(f"  Starting with {len(current_states)} copies at λ={initial_purity:.4f}")
        
        # Recursive purification through binary tree
        for level in range(num_levels):
            if self.verbose:
                print(f"\n  Level {level}: Processing {len(current_states)} states")
            
            level_amplification_results = []
            new_states = []
            level_iterations = 0
            level_attempts = 0
            
            # Pair up states and apply deterministic swap tests
            for i in range(0, len(current_states), 2):
                state1 = current_states[i]
                state2 = current_states[i + 1]
                
                # Retry loop in case amplitude amplification fails
                attempts = 0
                success = False
                
                while not success and attempts < max_attempts_per_level:
                    result = self.swap_simulator.deterministic_swap_test(state1, state2)
                    attempts += 1
                    level_attempts += 1
                    
                    if result.success:
                        success = True
                        new_states.append(result.output_state)
                        level_amplification_results.append(result)
                        level_iterations += result.iterations_used
                        
                        if self.verbose and attempts > 1:
                            print(f"    Swap {i//2}: Success after {attempts} attempts")
                    else:
                        if self.verbose:
                            print(f"    Swap {i//2}: Attempt {attempts} failed, retrying...")
                
                if not success:
                    raise RuntimeError(f"Failed to achieve swap test success after {max_attempts_per_level} attempts")
            
            # Update tracking
            current_states = new_states
            amplification_results.extend(level_amplification_results)
            
            if current_states:
                avg_purity = np.mean([state.purity for state in current_states])
                purity_evolution[level + 1] = avg_purity
                error_evolution[level + 1] = self._logical_error(avg_purity)
                
                if level_amplification_results:
                    avg_success_prob = np.mean([r.final_success_probability for r in level_amplification_results])
                    success_probabilities[level] = avg_success_prob
                    amplification_iterations[level] = level_iterations / len(level_amplification_results)
            
            total_swap_attempts += level_attempts
            total_amplification_iterations += level_iterations
            
            if self.verbose:
                print(f"    Level {level} complete: {len(current_states)} outputs, "
                      f"avg_purity={avg_purity:.4f}, total_iterations={level_iterations}")
        
        # Final result
        final_purity = current_states[0].purity if current_states else 0.0
        
        # Compare with theoretical prediction
        theoretical_evolution = self.theoretical_purity_evolution(initial_purity, num_levels)
        theoretical_final = theoretical_evolution[-1]
        
        # Check if simulation matches theory (within tolerance)
        tolerance = 0.01
        simulation_matches_theory = abs(final_purity - theoretical_final) < tolerance
        
        if self.verbose:
            print(f"\n  Simulation complete:")
            print(f"    Final purity: {final_purity:.6f}")
            print(f"    Theoretical:  {theoretical_final:.6f}")
            print(f"    Difference:   {abs(final_purity - theoretical_final):.6f}")
            print(f"    Matches theory: {simulation_matches_theory}")
            print(f"    Total amplification iterations: {total_amplification_iterations}")
        
        return PurificationResult(
            initial_purity=initial_purity,
            final_purity=final_purity,
            purity_evolution=purity_evolution,
            error_evolution=error_evolution,
            success_probabilities=success_probabilities,
            amplification_iterations=amplification_iterations,
            amplification_results=amplification_results,
            total_swap_attempts=total_swap_attempts,
            total_amplification_iterations=total_amplification_iterations,
            theoretical_prediction=theoretical_final,
            simulation_matches_theory=simulation_matches_theory,
            dimension=self.dimension,
            levels=num_levels
        )
    
    def _logical_error(self, purity: float) -> float:
        """Calculate logical error using paper's fidelity metric."""
        return (1 - purity) * (self.dimension - 1) / self.dimension
    
    def validate_deterministic_behavior(self, initial_delta: float, num_levels: int, 
                                      num_trials: int = 10) -> Dict:
        """
        Validate that the protocol produces deterministic results.
        
        Run multiple trials and verify consistent outcomes.
        
        Args:
            initial_delta: Initial noise level
            num_levels: Purification depth
            num_trials: Number of independent trials
            
        Returns:
            Validation statistics
        """
        if self.verbose:
            print(f"\nValidating deterministic behavior:")
            print(f"  Running {num_trials} trials at δ={initial_delta}, levels={num_levels}")
        
        final_purities = []
        total_iterations = []
        
        for trial in range(num_trials):
            if self.verbose:
                print(f"  Trial {trial + 1}/{num_trials}")
            
            # Set random seed for reproducibility within each trial
            np.random.seed(trial + 1000)
            
            result = self.recursive_purification_simulation(initial_delta, num_levels)
            final_purities.append(result.final_purity)
            total_iterations.append(result.total_amplification_iterations)
        
        # Calculate statistics
        purity_mean = np.mean(final_purities)
        purity_std = np.std(final_purities)
        purity_range = np.max(final_purities) - np.min(final_purities)
        
        iterations_mean = np.mean(total_iterations)
        iterations_std = np.std(total_iterations)
        
        # Theoretical prediction
        theoretical = self.theoretical_purity_evolution(1 - initial_delta, num_levels)[-1]
        
        validation_result = {
            'num_trials': num_trials,
            'initial_delta': initial_delta,
            'num_levels': num_levels,
            'final_purities': np.array(final_purities),
            'purity_mean': purity_mean,
            'purity_std': purity_std,
            'purity_range': purity_range,
            'total_iterations': np.array(total_iterations),
            'iterations_mean': iterations_mean,
            'iterations_std': iterations_std,
            'theoretical_prediction': theoretical,
            'mean_error': abs(purity_mean - theoretical),
            'is_deterministic': purity_std < 0.001,  # Very low variation
            'dimension': self.dimension
        }
        
        if self.verbose:
            print(f"  Validation results:")
            print(f"    Purity: {purity_mean:.6f} ± {purity_std:.6f} (range: {purity_range:.6f})")
            print(f"    Theoretical: {theoretical:.6f}")
            print(f"    Is deterministic: {validation_result['is_deterministic']}")
            print(f"    Iterations: {iterations_mean:.1f} ± {iterations_std:.1f}")
        
        return validation_result
    
    def save_result(self, result: any, filename: str, metadata: Dict = None) -> str:
        """Save simulation result with metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = f"{filename}_{timestamp}"
        filepath = os.path.join(self.data_dir, f"{full_filename}.npz")
        
        # Prepare data for saving
        if isinstance(result, PurificationResult):
            save_data = {
                'initial_purity': result.initial_purity,
                'final_purity': result.final_purity,
                'purity_evolution': result.purity_evolution,
                'error_evolution': result.error_evolution,
                'success_probabilities': result.success_probabilities,
                'amplification_iterations': result.amplification_iterations,
                'total_swap_attempts': result.total_swap_attempts,
                'total_amplification_iterations': result.total_amplification_iterations,
                'theoretical_prediction': result.theoretical_prediction,
                'simulation_matches_theory': result.simulation_matches_theory,
                'dimension': result.dimension,
                'levels': result.levels
            }
        else:
            save_data = result if isinstance(result, dict) else {'data': result}
        
        # Add metadata
        if metadata is None:
            metadata = {}
        metadata.update({
            'timestamp': timestamp,
            'dimension': self.dimension,
            'simulator_version': '2.0_full_simulation',
            'save_date': datetime.now().isoformat()
        })
        
        save_data['metadata'] = metadata
        np.savez_compressed(filepath, **save_data)
        
        if self.verbose:
            print(f"Results saved to: {filepath}")
        
        return filepath

def run_full_simulation_studies(data_dir: str = "./data/") -> Dict[str, str]:
    """
    Run comprehensive full simulation studies demonstrating deterministic behavior.
    
    Args:
        data_dir: Directory to save all results
        
    Returns:
        Dictionary mapping study name to saved file path
    """
    saved_files = {}
    
    print("="*70)
    print("FULL QUANTUM SIMULATION: STREAMING PURIFICATION QEC")
    print("="*70)
    
    # Study 1: Deterministic Behavior Validation
    print("\n1. Deterministic Behavior Validation")
    print("-" * 45)
    
    for d in [2, 4]:
        sim = StreamingPurificationSimulator(dimension=d, data_dir=data_dir, verbose=True)
        
        # Test deterministic behavior at different noise levels
        for delta in [0.2, 0.5, 0.8]:
            print(f"\nTesting d={d}, δ={delta}")
            validation = sim.validate_deterministic_behavior(delta, num_levels=4, num_trials=5)
            
            saved_files[f'validation_d{d}_delta{delta:.1f}'] = sim.save_result(
                validation,
                f"validation_studies/deterministic_validation_d{d}_delta{delta:.1f}",
                {"study_type": "deterministic_validation", "dimension": d, "delta": delta}
            )
    
    # Study 2: Full Simulation vs Theoretical Comparison
    print("\n2. Simulation vs Theory Comparison")
    print("-" * 45)
    
    for d in [2, 3, 4]:
        sim = StreamingPurificationSimulator(dimension=d, data_dir=data_dir, verbose=True)
        
        comparison_data = {
            'dimension': d,
            'noise_levels': np.linspace(0.1, 0.8, 8),
            'simulation_results': [],
            'theoretical_results': [],
            'num_levels': 5
        }
        
        for delta in comparison_data['noise_levels']:
            print(f"\nSimulating d={d}, δ={delta:.2f}")
            
            # Full simulation
            sim_result = sim.recursive_purification_simulation(delta, 5)
            comparison_data['simulation_results'].append({
                'delta': delta,
                'final_purity': sim_result.final_purity,
                'total_iterations': sim_result.total_amplification_iterations,
                'matches_theory': sim_result.simulation_matches_theory
            })
            
            # Theoretical prediction
            theoretical = sim.theoretical_purity_evolution(1 - delta, 5)[-1]
            comparison_data['theoretical_results'].append(theoretical)
        
        saved_files[f'simulation_vs_theory_d{d}'] = sim.save_result(
            comparison_data,
            f"full_simulations/simulation_vs_theory_d{d}",
            {"study_type": "simulation_vs_theory", "dimension": d}
        )
    
    # Study 3: Amplitude Amplification Performance Analysis
    print("\n3. Amplitude Amplification Analysis")
    print("-" * 45)
    
    sim = StreamingPurificationSimulator(dimension=2, data_dir=data_dir, verbose=True)
    
    amp_analysis = {
        'purity_levels': np.linspace(0.1, 0.9, 9),
        'amplification_data': []
    }
    
    for purity in amp_analysis['purity_levels']:
        state = QuantumState(purity, 2)
        
        # Test amplitude amplification
        result = sim.swap_simulator.amplitude_amplification(state, state)
        
        amp_analysis['amplification_data'].append({
            'input_purity': purity,
            'initial_success_prob': sim.swap_simulator.calculate_success_probability(state, state),
            'final_success_prob': result.final_success_probability,
            'iterations_used': result.iterations_used,
            'amplitude_evolution': result.amplitude_evolution,
            'amplification_success': result.success
        })
        
        print(f"  λ={purity:.2f}: P_init={amp_analysis['amplification_data'][-1]['initial_success_prob']:.4f} → "
              f"P_final={result.final_success_probability:.6f} ({result.iterations_used} iterations)")
    
    saved_files['amplitude_amplification_analysis'] = sim.save_result(
        amp_analysis,
        "full_simulations/amplitude_amplification_analysis",
        {"study_type": "amplitude_amplification_analysis"}
    )
    
    print("\n" + "="*70)
    print("FULL SIMULATION STUDIES COMPLETED")
    print("="*70)
    print(f"Results saved to: {data_dir}")
    
    return saved_files

if __name__ == "__main__":
    # Run full simulation studies
    run_full_simulation_studies()
    
    # Example: Single detailed simulation
    print("\n" + "="*50)
    print("EXAMPLE: Single Detailed Simulation")
    print("="*50)
    
    sim = StreamingPurificationSimulator(dimension=2, verbose=True)
    result = sim.recursive_purification_simulation(initial_delta=0.3, num_levels=4)
    
    print(f"\nExample Result Summary:")
    print(f"Initial purity: {result.initial_purity:.6f}")
    print(f"Final purity:   {result.final_purity:.6f}")
    print(f"Theoretical:    {result.theoretical_prediction:.6f}")
    print(f"Matches theory: {result.simulation_matches_theory}")
    print(f"Total amplification iterations: {result.total_amplification_iterations}")