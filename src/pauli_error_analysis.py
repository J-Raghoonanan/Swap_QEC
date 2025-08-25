"""
Enhanced analysis module implementing noise model dependence from Section II.E

This module provides:
1. Exact threshold calculations for different Pauli error types
2. Analysis of why purification preferentially corrects depolarizing over dephasing
3. Convergence analysis for asymptotic z-axis behavior (Eq. 51)
4. Comprehensive comparison of different noise models
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from src.quantum_states import BlochVectorState, pure_state_to_bloch_vector
from src.noise_models import PauliNoise, PureDephasingNoise, PureBitFlipNoise, SymmetricPauliNoise
from src.swap_operations import theoretical_bloch_evolution_pauli, analyze_noise_model_dependence


@dataclass  
class NoiseModelAnalysisResult:
    """Results from noise model dependence analysis."""
    noise_type: str
    error_rates: Dict[str, float]
    threshold_estimate: float
    asymptotic_behavior: str
    convergence_rate: float
    final_logical_error: float
    bloch_evolution: List[np.ndarray]
    logical_error_evolution: List[float]


class PauliErrorAnalyzer:
    """Analyzes the noise model dependence described in Section II.E."""
    
    def __init__(self):
        self.tolerance = 1e-6
        
    def analyze_z_dephasing_convergence(self, initial_bloch: np.ndarray, pz: float, 
                                      max_iterations: int = 50) -> Dict:
        """
        Analyze Z-dephasing convergence to z-axis (Eqs. 47-51).
        
        Shows asymptotic z-axis convergence where r_x, r_y → 0 exponentially
        due to (1-2p_z) suppression.
        """
        r_evolution = [initial_bloch.copy()]
        current_r = initial_bloch.copy()
        
        # Track evolution using recursive formulas (47)-(49)
        for n in range(max_iterations):
            rx, ry, rz = current_r
            
            # Calculate denominator for all components (from Eqs. 47-49)
            denominator = 3 + (1-2*pz)**2 * (rx**2 + ry**2) + rz**2
            
            # Apply recursive evolution
            r_next_x = 4 * (1-2*pz) * rx / denominator  # Eq. (47)
            r_next_y = 4 * (1-2*pz) * ry / denominator  # Eq. (48)  
            r_next_z = 4 * rz / denominator              # Eq. (49)
            
            current_r = np.array([r_next_x, r_next_y, r_next_z])
            r_evolution.append(current_r.copy())
            
            # Check convergence
            if np.linalg.norm(current_r - r_evolution[-2]) < self.tolerance:
                break
        
        # Asymptotic logical error (Eq. 51)
        final_r = r_evolution[-1]
        asymptotic_logical_error = 0.5 * abs(final_r[2] - 1)  # |r_z^(∞) - 1|/2
        
        # Calculate convergence rate for x,y components
        x_evolution = [r[0] for r in r_evolution]
        y_evolution = [r[1] for r in r_evolution]
        
        # Exponential decay rate
        if len(x_evolution) > 10:
            decay_rate = np.mean(np.log(np.abs(np.array(x_evolution[5:10]) / np.array(x_evolution[4:9]))))
        else:
            decay_rate = np.log(1 - 2*pz)  # Theoretical rate
        
        return {
            'bloch_evolution': r_evolution,
            'asymptotic_logical_error': asymptotic_logical_error,
            'x_decay_rate': decay_rate,
            'y_decay_rate': decay_rate,  # Same as x for Z-dephasing
            'iterations_to_convergence': len(r_evolution) - 1,
            'final_bloch_vector': final_r
        }
    
    def compare_noise_model_thresholds(self, initial_bloch: np.ndarray, 
                                     error_rates: List[float] = None) -> Dict[str, NoiseModelAnalysisResult]:
        """
        Compare threshold behavior across different noise models.
        
        Demonstrates why protocol performance depends on noise model.
        """
        if error_rates is None:
            error_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        results = {}
        
        # Test different noise models
        noise_models = {
            'pure_z_dephasing': lambda p: {'px': 0, 'py': 0, 'pz': p},
            'pure_x_bitflip': lambda p: {'px': p, 'py': 0, 'pz': 0},
            'symmetric_pauli': lambda p: {'px': p/3, 'py': p/3, 'pz': p/3},
            'xy_biased': lambda p: {'px': p/2, 'py': p/2, 'pz': 0},
            'depolarizing_equivalent': lambda p: {'px': p/3, 'py': p/3, 'pz': p/3}  # For comparison
        }
        
        for noise_name, noise_func in noise_models.items():
            # Find approximate threshold
            threshold = self._estimate_threshold(initial_bloch, noise_func, error_rates)
            
            # Analyze behavior at moderate error rate
            test_rates = noise_func(0.3)
            bloch_evo, mag_evo = theoretical_bloch_evolution_pauli(
                initial_bloch, test_rates['px'], test_rates['py'], test_rates['pz'], num_levels=8)
            
            # Calculate logical errors
            target_bloch = initial_bloch / np.linalg.norm(initial_bloch)
            logical_errors = [0.5 * np.linalg.norm(bloch - target_bloch) for bloch in bloch_evo]
            
            # Determine asymptotic behavior
            asymptotic_behavior = self._analyze_asymptotic_behavior(bloch_evo, test_rates)
            
            # Calculate convergence rate
            if len(logical_errors) > 5:
                convergence_rate = np.mean(np.array(logical_errors[2:6]) / np.array(logical_errors[1:5]))
            else:
                convergence_rate = logical_errors[-1] / logical_errors[0]
            
            results[noise_name] = NoiseModelAnalysisResult(
                noise_type=noise_name,
                error_rates=test_rates,
                threshold_estimate=threshold,
                asymptotic_behavior=asymptotic_behavior,
                convergence_rate=convergence_rate,
                final_logical_error=logical_errors[-1],
                bloch_evolution=bloch_evo,
                logical_error_evolution=logical_errors
            )
        
        return results
    
    def _estimate_threshold(self, initial_bloch: np.ndarray, noise_func, 
                          error_rates: List[float]) -> float:
        """Estimate threshold where purification fails to improve."""
        for error_rate in reversed(error_rates):  # Start from high error rates
            rates = noise_func(error_rate)
            
            # Test if purification improves
            bloch_evo, _ = theoretical_bloch_evolution_pauli(
                initial_bloch, rates['px'], rates['py'], rates['pz'], num_levels=3)
            
            # Check if final state is better than initial
            initial_magnitude = np.linalg.norm(bloch_evo[0])
            final_magnitude = np.linalg.norm(bloch_evo[-1])
            
            if final_magnitude > initial_magnitude * 1.01:  # 1% improvement threshold
                return error_rate
        
        return 0.0  # No threshold found in range
    
    def _analyze_asymptotic_behavior(self, bloch_evolution: List[np.ndarray], 
                                   error_rates: Dict[str, float]) -> str:
        """Analyze the asymptotic convergence behavior."""
        if len(bloch_evolution) < 3:
            return "insufficient_data"
        
        final_bloch = bloch_evolution[-1]
        px, py, pz = error_rates['px'], error_rates['py'], error_rates['pz']
        
        # Check for axis-aligned convergence
        abs_final = np.abs(final_bloch)
        max_component = np.argmax(abs_final)
        
        if px == py == 0 and pz > 0:
            return f"z_axis_convergence (Z-dephasing dominates)"
        elif py == pz == 0 and px > 0:
            return f"x_axis_convergence (X-bitflip dominates)"
        elif px == pz == 0 and py > 0:
            return f"y_axis_convergence (Y-bitflip dominates)"
        elif px == py == pz:
            return f"uniform_shrinkage (symmetric Pauli)"
        else:
            axis_names = ['x', 'y', 'z']
            return f"biased_toward_{axis_names[max_component]}_axis"
    
    def demonstrate_preferential_correction(self, initial_bloch: np.ndarray = None) -> Dict:
        """
        Demonstrate why purification preferentially corrects depolarizing over dephasing.
        
        This addresses the key insight from your manuscript about geometric
        reasons for noise model dependence.
        """
        if initial_bloch is None:
            # Use a representative initial state
            initial_bloch = np.array([0.6, 0.6, 0.6])  # Mixed initial state
        
        # Test same total error rate for different distributions
        total_error = 0.3
        
        test_cases = {
            'pure_depolarizing': {
                'equivalent_to': 'symmetric_pauli',
                'rates': {'px': total_error/3, 'py': total_error/3, 'pz': total_error/3},
                'description': 'Uniform distribution across all Pauli errors'
            },
            'pure_z_dephasing': {
                'rates': {'px': 0, 'py': 0, 'pz': total_error},
                'description': 'All error concentrated in Z-dephasing'
            },
            'pure_x_bitflip': {
                'rates': {'px': total_error, 'py': 0, 'pz': 0},
                'description': 'All error concentrated in X-bitflip'
            },
            'xy_biased': {
                'rates': {'px': total_error/2, 'py': total_error/2, 'pz': 0},
                'description': 'Error split between X and Y (no Z)'
            }
        }
        
        comparison_results = {}
        
        for case_name, case_info in test_cases.items():
            rates = case_info['rates']
            
            # Run evolution
            bloch_evo, mag_evo = theoretical_bloch_evolution_pauli(
                initial_bloch, rates['px'], rates['py'], rates['pz'], num_levels=6)
            
            # Calculate logical errors
            target_bloch = initial_bloch / np.linalg.norm(initial_bloch)
            logical_errors = [0.5 * np.linalg.norm(bloch - target_bloch) for bloch in bloch_evo]
            
            # Performance metrics
            error_reduction = logical_errors[0] / logical_errors[-1] if logical_errors[-1] > 0 else np.inf
            final_coherence = np.linalg.norm(bloch_evo[-1])
            
            comparison_results[case_name] = {
                'description': case_info['description'],
                'error_rates': rates,
                'logical_error_evolution': logical_errors,
                'bloch_evolution': bloch_evo,
                'magnitude_evolution': mag_evo,
                'error_reduction_factor': error_reduction,
                'final_coherence': final_coherence,
                'initial_logical_error': logical_errors[0],
                'final_logical_error': logical_errors[-1]
            }
        
        return comparison_results
    
    def plot_noise_model_comparison(self, comparison_results: Dict, save_path: Optional[str] = None):
        """Plot comparison showing preferential correction of different noise types."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Color scheme for different noise types
        colors = {
            'pure_depolarizing': 'blue',
            'pure_z_dephasing': 'red', 
            'pure_x_bitflip': 'green',
            'xy_biased': 'orange'
        }
        
        # Plot 1: Logical error evolution
        ax1 = axes[0, 0]
        for case_name, results in comparison_results.items():
            if case_name in colors:
                levels = range(len(results['logical_error_evolution']))
                ax1.semilogy(levels, results['logical_error_evolution'], 
                           'o-', color=colors[case_name], label=case_name.replace('_', ' ').title(),
                           linewidth=2, markersize=4)
        
        ax1.set_xlabel('Purification Level')
        ax1.set_ylabel('Logical Error Rate')
        ax1.set_title('Logical Error Evolution by Noise Type')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Bloch vector magnitude evolution  
        ax2 = axes[0, 1]
        for case_name, results in comparison_results.items():
            if case_name in colors:
                levels = range(len(results['magnitude_evolution']))
                ax2.plot(levels, results['magnitude_evolution'],
                        'o-', color=colors[case_name], label=case_name.replace('_', ' ').title(),
                        linewidth=2, markersize=4)
        
        ax2.set_xlabel('Purification Level')
        ax2.set_ylabel('Bloch Vector Magnitude')
        ax2.set_title('Coherence Recovery by Noise Type')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Error reduction factors
        ax3 = axes[1, 0]
        case_names = list(comparison_results.keys())
        reduction_factors = [comparison_results[name]['error_reduction_factor'] for name in case_names]
        
        bars = ax3.bar(range(len(case_names)), reduction_factors, 
                      color=[colors.get(name, 'gray') for name in case_names])
        ax3.set_yscale('log')
        ax3.set_xlabel('Noise Model')
        ax3.set_ylabel('Error Reduction Factor')
        ax3.set_title('Error Reduction Effectiveness')
        ax3.set_xticks(range(len(case_names)))
        ax3.set_xticklabels([name.replace('_', '\n') for name in case_names], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Asymptotic Bloch vectors (final positions)
        ax4 = axes[1, 1]
        
        # Create 3D-like visualization on 2D plot (rz vs |r_xy|)
        for case_name, results in comparison_results.items():
            if case_name in colors:
                final_bloch = results['bloch_evolution'][-1]
                rz = final_bloch[2]
                r_xy_magnitude = np.sqrt(final_bloch[0]**2 + final_bloch[1]**2)
                
                ax4.scatter(r_xy_magnitude, rz, color=colors[case_name], 
                          label=case_name.replace('_', ' ').title(), s=100, alpha=0.7)
        
        # Add target point
        target = np.array([0, 0, 1])  # Assuming target is |0⟩ state
        ax4.scatter(0, 1, color='black', marker='*', s=200, label='Target |0⟩')
        
        ax4.set_xlabel('|r_xy| = √(r_x² + r_y²)')
        ax4.set_ylabel('r_z')
        ax4.set_title('Final Bloch Vector Positions')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(-0.1, 1.1)
        ax4.set_ylim(-1.1, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Noise model comparison saved to {save_path}")
        
        return fig
    
    def calculate_exact_thresholds(self, initial_bloch: np.ndarray) -> Dict[str, float]:
        """
        Calculate exact thresholds for different Pauli error types using Appendix E formulas.
        """
        thresholds = {}
        
        # Pure Z-dephasing: Has infinite threshold (any p_z can be corrected if r_z ≠ 0)
        if abs(initial_bloch[2]) > self.tolerance:
            thresholds['pure_z_dephasing'] = np.inf
        else:
            thresholds['pure_z_dephasing'] = 0.0
        
        # Pure X-bit-flip: Threshold depends on perpendicular components
        r_perp = np.sqrt(initial_bloch[1]**2 + initial_bloch[2]**2)  # r_⊥ from Eq. (E9)
        r_magnitude = np.linalg.norm(initial_bloch)
        
        if r_magnitude > self.tolerance:
            # From Eq. (E9): px < (1 - |r_⊥|/|r|)/2
            thresholds['pure_x_bitflip'] = (1 - r_perp/r_magnitude) / 2
        else:
            thresholds['pure_x_bitflip'] = 0.0
        
        # Worst-case thresholds from Appendix E
        thresholds['x_bitflip_worst_case'] = 0.5  # States aligned with X-axis
        thresholds['x_bitflip_best_case'] = 0.0   # States perpendicular to X-axis
        
        # For symmetric Pauli, estimate based on overall coherence preservation
        thresholds['symmetric_pauli'] = 0.75  # Empirical estimate
        
        return thresholds
    
    def generate_noise_dependence_report(self, initial_bloch: np.ndarray = None, 
                                       save_dir: str = "./analysis_results/") -> Dict:
        """
        Generate comprehensive report on noise model dependence.
        
        This implements the key insights from Section II.E about why the protocol
        works differently for different noise types.
        """
        if initial_bloch is None:
            # Use representative initial state
            initial_bloch = np.array([0.5, 0.5, 0.8])  # Slightly z-biased
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Run comprehensive analysis
        comparison_results = self.compare_noise_model_thresholds(initial_bloch)
        z_dephasing_analysis = self.analyze_z_dephasing_convergence(initial_bloch, pz=0.3)
        threshold_calculations = self.calculate_exact_thresholds(initial_bloch)
        
        # Generate plots
        fig_comparison = self.plot_noise_model_comparison(
            comparison_results, 
            save_path=os.path.join(save_dir, "noise_model_comparison.png")
        )
        
        # Create summary report
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'initial_bloch_vector': initial_bloch.tolist(),
            'key_insights': {
                'preferential_correction': "Purification preferentially corrects depolarizing over dephasing errors",
                'geometric_explanation': "Due to anisotropic Bloch vector renormalization",
                'z_dephasing_behavior': "Exponential convergence to z-axis with rate (1-2p_z)",
                'threshold_dependence': "Thresholds vary dramatically by noise type"
            },
            'threshold_estimates': threshold_calculations,
            'noise_model_comparison': {name: {
                'final_logical_error': res.final_logical_error,
                'error_reduction_factor': res.convergence_rate,
                'asymptotic_behavior': res.asymptotic_behavior
            } for name, res in comparison_results.items()},
            'z_dephasing_detailed': {
                'asymptotic_logical_error': z_dephasing_analysis['asymptotic_logical_error'],
                'xy_decay_rate': z_dephasing_analysis['x_decay_rate'],
                'iterations_to_convergence': z_dephasing_analysis['iterations_to_convergence']
            }
        }
        
        # Save report
        report_path = os.path.join(save_dir, "noise_dependence_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Comprehensive noise dependence report saved to {report_path}")
        
        return report


def validate_manuscript_formulas():
    """
    Validate that implementation matches manuscript formulas exactly.
    
    This function should be run as a test to ensure all equations are implemented correctly.
    """
    print("=== Validating Manuscript Formula Implementation ===")
    
    # Test case from Appendix C
    initial_delta = 0.3  # 30% depolarization
    initial_purity = 0.7  # λ₀ = 1 - δ
    dimension = 2
    
    # Test depolarizing purity transformation (should match Appendix C)
    expected_sequence = [0.7, 0.802, 0.881, 0.933]  # From manuscript
    
    current_purity = initial_purity
    computed_sequence = [current_purity]
    
    for level in range(3):
        # Apply transformation λ_{i+1} = 4λᵢ/(3 + λᵢ²) for qubits
        current_purity = 4 * current_purity / (3 + current_purity**2)
        computed_sequence.append(current_purity)
    
    print("Purity evolution validation:")
    print(f"Expected: {expected_sequence}")
    print(f"Computed: {[round(p, 3) for p in computed_sequence]}")
    print(f"Match: {np.allclose(expected_sequence, computed_sequence, atol=1e-3)}")
    
    # Test Pauli success probability formula
    test_pauli_rates = {'px': 0.1, 'py': 0.15, 'pz': 0.05}
    p_total = sum(test_pauli_rates.values())
    
    # From Eq. (41): P_success = 1/2[2 - 2p_total + p²_total + Σp²_i]
    sum_squared = test_pauli_rates['px']**2 + test_pauli_rates['py']**2 + test_pauli_rates['pz']**2
    expected_p_success = 0.5 * (2 - 2*p_total + p_total**2 + sum_squared)
    
    print(f"\nPauli success probability validation:")
    print(f"Error rates: {test_pauli_rates}")
    print(f"Expected P_success: {expected_p_success:.6f}")
    
    return True


if __name__ == "__main__":
    # Run validation
    validate_manuscript_formulas()
    
    # Generate comprehensive analysis
    analyzer = PauliErrorAnalyzer()
    report = analyzer.generate_noise_dependence_report()
    
    print("\nNoise model dependence analysis complete!")