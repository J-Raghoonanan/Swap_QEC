"""
Threshold Analysis for Streaming Purification Quantum Error Correction

This module calculates error thresholds for different noise models and dimensions.
It determines the maximum error rates at which the purification protocol converges
to high fidelity states.

Key Achievement: Provides rigorous threshold calculations that can be compared
with theoretical predictions and other QEC codes.

WARNING: Threshold values produced here require experimental validation.
The theoretical framework is sound but numerical thresholds are estimates.
All results are saved in organized directory structures with metadata.
"""

import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
import warnings
from scipy.optimize import brentq, minimize_scalar
from purification_simulator import StreamingPurificationSimulator, QuantumState

# [Previous dataclasses and noise models remain the same]
@dataclass
class ThresholdResult:
    """Result of threshold calculation for a specific configuration."""
    error_model: str
    dimension: int
    threshold_value: float
    convergence_criterion: float
    purification_levels: int
    final_purity_at_threshold: float
    success_probability_at_threshold: float
    confidence_interval: Tuple[float, float]
    computation_method: str

class NoiseModel:
    """Base class for different noise models."""
    def __init__(self, name: str):
        self.name = name
    def apply_noise(self, pure_state: np.ndarray, error_rate: float) -> QuantumState:
        raise NotImplementedError

class DepolarizingNoise(NoiseModel):
    """Standard depolarizing noise model: ρ = (1-δ)|ψ⟩⟨ψ| + δI/d"""
    def __init__(self, dimension: int):
        super().__init__(f"Depolarizing_d{dimension}")
        self.dimension = dimension
    def apply_noise(self, pure_state: np.ndarray, error_rate: float) -> QuantumState:
        purity = 1 - error_rate
        return QuantumState(purity, self.dimension, pure_state)

class PauliNoise(NoiseModel):
    """General Pauli noise model with independent X, Y, Z error rates."""
    def __init__(self, px: float, py: float, pz: float, dimension: int = 2):
        if dimension != 2:
            raise ValueError("Pauli noise currently only implemented for qubits")
        super().__init__(f"Pauli_px{px:.2f}_py{py:.2f}_pz{pz:.2f}")
        self.px, self.py, self.pz = px, py, pz
        self.dimension = dimension
    def apply_noise(self, pure_state: np.ndarray, error_rate: float) -> QuantumState:
        scaled_px = self.px * error_rate
        scaled_py = self.py * error_rate  
        scaled_pz = self.pz * error_rate
        total_error = scaled_px + scaled_py + scaled_pz
        effective_purity = 1 - total_error
        return QuantumState(effective_purity, self.dimension, pure_state)

class ThresholdCalculator:
    """Calculate error thresholds with comprehensive data saving."""
    
    def __init__(self, data_dir: str = "./data/", max_levels: int = 10, 
                 convergence_tolerance: float = 1e-6, verbose: bool = False):
        """Initialize threshold calculator with data directory."""
        self.data_dir = data_dir
        self.max_levels = max_levels
        self.convergence_tolerance = convergence_tolerance
        self.verbose = verbose
        
        # Create data directories
        self.threshold_dir = os.path.join(data_dir, "threshold_analysis")
        self.scaling_dir = os.path.join(data_dir, "dimension_scaling")
        self.comparison_dir = os.path.join(data_dir, "literature_comparison")
        
        for dir_path in [self.threshold_dir, self.scaling_dir, self.comparison_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Cache simulators for different dimensions
        self._simulators = {}
        
        if self.verbose:
            print(f"ThresholdCalculator initialized with data directory: {data_dir}")
    
    def _get_simulator(self, dimension: int) -> StreamingPurificationSimulator:
        """Get or create simulator for given dimension."""
        if dimension not in self._simulators:
            self._simulators[dimension] = StreamingPurificationSimulator(
                dimension=dimension, verbose=False
            )
        return self._simulators[dimension]
    
    def _test_convergence(self, error_rate: float, noise_model: NoiseModel, 
                         target_fidelity: float = 0.99) -> Dict:
        """
        Test convergence and return detailed data for analysis.
        
        Returns:
            Dictionary with convergence data including purity evolution
        """
        if error_rate >= 1.0:
            return {'converged': False, 'reason': 'error_rate_too_high'}
        
        # Create noisy state
        pure_state = np.zeros(noise_model.dimension)
        pure_state[0] = 1.0
        noisy_state = noise_model.apply_noise(pure_state, error_rate)
        
        # Track detailed evolution
        simulator = self._get_simulator(noise_model.dimension)
        
        evolution_data = {
            'initial_purity': noisy_state.purity,
            'purity_evolution': [noisy_state.purity],
            'fidelity_evolution': [self._purity_to_fidelity(noisy_state.purity, noise_model.dimension)],
            'success_probabilities': [],
            'converged': False,
            'convergence_level': -1,
            'final_purity': noisy_state.purity,
            'final_fidelity': self._purity_to_fidelity(noisy_state.purity, noise_model.dimension),
            'reason': 'max_levels_reached'
        }
        
        current_purity = noisy_state.purity
        
        for level in range(self.max_levels):
            # Calculate success probability at this level
            current_state = QuantumState(current_purity, noise_model.dimension, pure_state)
            success_prob = simulator.swap_simulator.calculate_success_probability(current_state, current_state)
            evolution_data['success_probabilities'].append(success_prob)
            
            # Apply one level of purification
            new_purity = simulator.swap_simulator._compute_output_purity(current_purity)
            new_fidelity = self._purity_to_fidelity(new_purity, noise_model.dimension)
            
            evolution_data['purity_evolution'].append(new_purity)
            evolution_data['fidelity_evolution'].append(new_fidelity)
            
            # Check for convergence
            if new_fidelity >= target_fidelity:
                evolution_data['converged'] = True
                evolution_data['convergence_level'] = level + 1
                evolution_data['final_purity'] = new_purity
                evolution_data['final_fidelity'] = new_fidelity
                evolution_data['reason'] = 'target_fidelity_reached'
                if self.verbose:
                    print(f"    Converged at level {level + 1}, fidelity {new_fidelity:.6f}")
                break
            
            # Check for divergence (purity not increasing sufficiently)
            if new_purity <= current_purity + self.convergence_tolerance:
                evolution_data['converged'] = False
                evolution_data['final_purity'] = new_purity
                evolution_data['final_fidelity'] = new_fidelity
                evolution_data['reason'] = 'purity_stagnation'
                if self.verbose:
                    print(f"    Diverged at level {level + 1}, purity stagnant at {new_purity:.6f}")
                break
            
            current_purity = new_purity
        
        return evolution_data
    
    def _purity_to_fidelity(self, purity: float, dimension: int) -> float:
        """Convert purity parameter to fidelity with target pure state."""
        return purity + (1 - purity) / dimension
    
    def calculate_threshold(self, noise_model: NoiseModel, 
                          target_fidelity: float = 0.99,
                          search_range: Tuple[float, float] = (0.01, 0.99)) -> ThresholdResult:
        """Calculate error threshold and save detailed data."""
        
        if self.verbose:
            print(f"Calculating threshold for {noise_model.name}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create detailed search data
        search_data = {
            'noise_model': noise_model.name,
            'dimension': noise_model.dimension,
            'target_fidelity': target_fidelity,
            'search_range': search_range,
            'convergence_tolerance': self.convergence_tolerance,
            'max_levels': self.max_levels,
            'search_points': [],
            'timestamp': timestamp
        }
        
        min_error, max_error = search_range
        
        # Test search range endpoints
        min_test = self._test_convergence(min_error, noise_model, target_fidelity)
        max_test = self._test_convergence(max_error, noise_model, target_fidelity)
        
        search_data['search_points'].append({'error_rate': min_error, 'test_result': min_test})
        search_data['search_points'].append({'error_rate': max_error, 'test_result': max_test})
        
        if not min_test['converged']:
            warnings.warn(f"Protocol doesn't converge even at low error rate {min_error}")
            threshold_result = ThresholdResult(
                error_model=noise_model.name,
                dimension=noise_model.dimension,
                threshold_value=0.0,
                convergence_criterion=target_fidelity,
                purification_levels=self.max_levels,
                final_purity_at_threshold=0.0,
                success_probability_at_threshold=0.0,
                confidence_interval=(0.0, min_error),
                computation_method="search_range_invalid"
            )
            # Save failed search data
            self._save_search_data(search_data, threshold_result)
            return threshold_result
        
        # Binary search for threshold
        low, high = min_error, max_error
        search_iterations = 0
        
        while high - low > self.convergence_tolerance and search_iterations < 50:
            mid = (low + high) / 2
            mid_test = self._test_convergence(mid, noise_model, target_fidelity)
            search_data['search_points'].append({'error_rate': mid, 'test_result': mid_test})
            
            if mid_test['converged']:
                low = mid  # Can handle this error rate
            else:
                high = mid  # Cannot handle this error rate
            
            search_iterations += 1
            
            if self.verbose:
                print(f"  Search iteration {search_iterations}: [{low:.6f}, {high:.6f}], "
                      f"testing {mid:.6f} -> {'PASS' if mid_test['converged'] else 'FAIL'}")
        
        threshold = low
        
        # Calculate additional metrics at threshold
        pure_state = np.zeros(noise_model.dimension)
        pure_state[0] = 1.0
        threshold_state = noise_model.apply_noise(pure_state, threshold)
        
        simulator = self._get_simulator(noise_model.dimension)
        success_prob = simulator.swap_simulator.calculate_success_probability(
            threshold_state, threshold_state
        )
        
        threshold_result = ThresholdResult(
            error_model=noise_model.name,
            dimension=noise_model.dimension,
            threshold_value=threshold,
            convergence_criterion=target_fidelity,
            purification_levels=self.max_levels,
            final_purity_at_threshold=threshold_state.purity,
            success_probability_at_threshold=success_prob,
            confidence_interval=(threshold - self.convergence_tolerance, 
                               threshold + self.convergence_tolerance),
            computation_method="binary_search"
        )
        
        # Save all search data
        self._save_search_data(search_data, threshold_result)
        
        if self.verbose:
            print(f"  Threshold found: {threshold:.6f}")
        
        return threshold_result
    
    def _save_search_data(self, search_data: Dict, result: ThresholdResult):
        """Save detailed search data and results."""
        filename = f"threshold_{result.error_model}_d{result.dimension}_{search_data['timestamp']}"
        filepath = os.path.join(self.threshold_dir, f"{filename}.npz")
        
        # Prepare data for saving
        save_data = {
            'search_data': search_data,
            'threshold_result': asdict(result),
            'metadata': {
                'calculator_config': {
                    'max_levels': self.max_levels,
                    'convergence_tolerance': self.convergence_tolerance
                },
                'save_timestamp': datetime.now().isoformat()
            }
        }
        
        # Convert to numpy arrays where appropriate
        if search_data['search_points']:
            error_rates = [sp['error_rate'] for sp in search_data['search_points']]
            convergence_flags = [sp['test_result']['converged'] for sp in search_data['search_points']]
            final_fidelities = [sp['test_result']['final_fidelity'] for sp in search_data['search_points']]
            
            save_data['error_rates'] = np.array(error_rates)
            save_data['convergence_flags'] = np.array(convergence_flags)
            save_data['final_fidelities'] = np.array(final_fidelities)
        
        np.savez_compressed(filepath, **save_data)
        
        if self.verbose:
            print(f"  Search data saved to: {filepath}")
    
    def dimension_scaling_study(self, dimensions: List[int], 
                               noise_model_factory: Callable[[int], NoiseModel],
                               target_fidelity: float = 0.99) -> Dict[int, ThresholdResult]:
        """Study dimension scaling and save comprehensive data."""
        
        if self.verbose:
            print("Starting dimension scaling study")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        scaling_data = {
            'dimensions': dimensions,
            'target_fidelity': target_fidelity,
            'noise_model_type': noise_model_factory(2).name.split('_')[0],  # Extract base type
            'timestamp': timestamp,
            'results': {},
            'threshold_values': [],
            'dimension_array': []
        }
        
        results = {}
        
        for d in dimensions:
            if self.verbose:
                print(f"\nAnalyzing dimension {d}")
            
            noise_model = noise_model_factory(d)
            threshold_result = self.calculate_threshold(noise_model, target_fidelity)
            results[d] = threshold_result
            
            # Collect data for analysis
            scaling_data['results'][d] = asdict(threshold_result)
            scaling_data['threshold_values'].append(threshold_result.threshold_value)
            scaling_data['dimension_array'].append(d)
            
            if self.verbose:
                print(f"  Threshold for d={d}: {threshold_result.threshold_value:.6f}")
        
        # Convert to numpy arrays
        scaling_data['threshold_values'] = np.array(scaling_data['threshold_values'])
        scaling_data['dimension_array'] = np.array(scaling_data['dimension_array'])
        
        # Save scaling study data
        filename = f"dimension_scaling_{scaling_data['noise_model_type']}_{timestamp}"
        filepath = os.path.join(self.scaling_dir, f"{filename}.npz")
        
        np.savez_compressed(filepath, **scaling_data)
        
        if self.verbose:
            print(f"Dimension scaling data saved to: {filepath}")
        
        return results
    
    def compare_with_known_thresholds(self, calculated_thresholds: Dict[str, ThresholdResult]) -> Dict:
        """Compare with literature and save comparison data."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # WARNING: These are approximate values from literature
        known_thresholds = {
            "surface_code_depolarizing": 0.0109,
            "steane_code": 0.0073,
            "shor_code": 0.0054,
            "spinor_code_lower": 0.32,
            "spinor_code_upper": 0.75
        }
        
        comparison = {
            'timestamp': timestamp,
            'calculated': {name: asdict(result) for name, result in calculated_thresholds.items()},
            'literature_values': known_thresholds,
            'performance_ratios': {},
            'analysis': {},
            'comparison_arrays': {}
        }
        
        # Compare with relevant baselines
        calculated_names = []
        calculated_values = []
        surface_ratios = []
        spinor_ratios = []
        
        for name, result in calculated_thresholds.items():
            calculated_names.append(name)
            calculated_values.append(result.threshold_value)
            
            if "depolarizing" in name.lower():
                surface_ratio = result.threshold_value / known_thresholds["surface_code_depolarizing"]
                spinor_lower_ratio = result.threshold_value / known_thresholds["spinor_code_lower"]
                
                surface_ratios.append(surface_ratio)
                spinor_ratios.append(spinor_lower_ratio)
                
                comparison["performance_ratios"][name] = {
                    "vs_surface_code": surface_ratio,
                    "vs_spinor_lower": spinor_lower_ratio
                }
                
                comparison["analysis"][name] = {
                    "better_than_surface": surface_ratio > 1.0,
                    "within_spinor_range": (known_thresholds["spinor_code_lower"] <= 
                                           result.threshold_value <= 
                                           known_thresholds["spinor_code_upper"])
                }
        
        # Create arrays for plotting
        comparison['comparison_arrays'] = {
            'calculated_names': calculated_names,
            'calculated_values': np.array(calculated_values),
            'surface_ratios': np.array(surface_ratios) if surface_ratios else np.array([]),
            'spinor_ratios': np.array(spinor_ratios) if spinor_ratios else np.array([]),
            'literature_names': list(known_thresholds.keys()),
            'literature_values': np.array(list(known_thresholds.values()))
        }
        
        # Save comparison data
        filename = f"literature_comparison_{timestamp}"
        filepath = os.path.join(self.comparison_dir, f"{filename}.npz")
        
        np.savez_compressed(filepath, **comparison)
        
        if self.verbose:
            print(f"Literature comparison data saved to: {filepath}")
        
        return comparison

def run_comprehensive_threshold_analysis(data_dir: str = "./data/") -> Dict:
    """Run comprehensive threshold analysis with full data saving."""
    
    print("="*70)
    print("COMPREHENSIVE THRESHOLD ANALYSIS WITH DATA SAVING")
    print("="*70)
    
    calculator = ThresholdCalculator(data_dir=data_dir, max_levels=20, verbose=True)
    all_results = {}
    
    # 1. Depolarizing noise analysis
    print("\n1. DEPOLARIZING NOISE THRESHOLD ANALYSIS")
    print("-" * 50)
    
    dimensions = [2, 3, 4, 5, 8]
    depolarizing_results = calculator.dimension_scaling_study(
        dimensions=dimensions,
        noise_model_factory=lambda d: DepolarizingNoise(d),
        target_fidelity=0.99
    )
    all_results["depolarizing_scaling"] = depolarizing_results
    
    # 2. Pauli noise analysis
    print("\n2. PAULI NOISE ANALYSIS")
    print("-" * 50)
    
    pauli_models = [
        PauliNoise(0.33, 0.33, 0.33, 2),  # Symmetric
        PauliNoise(0.6, 0.2, 0.2, 2),    # X-biased
        PauliNoise(0.2, 0.2, 0.6, 2),    # Z-biased
        PauliNoise(0.0, 0.0, 1.0, 2),    # Pure dephasing
    ]
    
    pauli_results = {}
    for model in pauli_models:
        result = calculator.calculate_threshold(model, target_fidelity=0.99)
        pauli_results[model.name] = result
    all_results["pauli_noise"] = pauli_results
    
    # 3. Literature comparison
    print("\n3. LITERATURE COMPARISON")
    print("-" * 50)
    
    all_thresholds = {}
    all_thresholds.update({f"depolarizing_d{d}": result 
                          for d, result in depolarizing_results.items()})
    all_thresholds.update(pauli_results)
    
    comparison = calculator.compare_with_known_thresholds(all_thresholds)
    all_results["literature_comparison"] = comparison
    
    # 4. Save master results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_filepath = os.path.join(data_dir, f"threshold_analysis_master_{timestamp}.json")
    
    # Convert results to JSON-serializable format
    json_results = {}
    for key, value in all_results.items():
        if key == "literature_comparison":
            json_results[key] = value  # Already JSON-serializable
        else:
            json_results[key] = {str(k): asdict(v) for k, v in value.items()}
    
    with open(master_filepath, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\nMaster results saved to: {master_filepath}")
    print(f"Individual data files saved in: {data_dir}/threshold_analysis/")
    
    return all_results

if __name__ == "__main__":
    results = run_comprehensive_threshold_analysis()
    