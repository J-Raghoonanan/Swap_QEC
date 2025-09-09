"""
Threshold analysis and plotting for streaming QEC protocol.
Focused on producing Figure 4 analog results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from src.streaming_protocol import StreamingPurificationProtocol, create_depolarizing_noise_factory, create_pauli_noise_factory


@dataclass
class ThresholdResults:
    """Results from threshold analysis."""
    error_rates: np.ndarray
    final_logical_errors: np.ndarray
    code_sizes: List[int]
    noise_type: str
    dimension: int


class ThresholdAnalyzer:
    """Analyzes threshold behavior for streaming purification protocol."""
    
    def __init__(self):
        self.protocol = StreamingPurificationProtocol()
    
    def run_threshold_sweep(self, 
                           noise_type: str = 'depolarizing',
                           dimension: int = 2,
                           error_rates: np.ndarray = None,
                           code_sizes: List[int] = None,
                           num_levels: int = 4) -> ThresholdResults:
        """
        Run threshold analysis sweep similar to Grafe et al. Figure 4.
        
        Args:
            noise_type: 'depolarizing', 'symmetric_pauli', 'dephasing', 'bitflip'
            dimension: Qudit dimension (2 for qubits)
            error_rates: Array of physical error rates to test
            code_sizes: List of code sizes (number of input states)
            num_levels: Number of purification levels
        """
        if error_rates is None:
            error_rates = np.linspace(0.01, 0.95, 20)
        
        if code_sizes is None:
            code_sizes = [4, 8, 16, 32, 64]  # Powers of 2
        
        print(f"Running threshold analysis for {noise_type} noise")
        print(f"Error rates: {len(error_rates)} points from {error_rates[0]:.2f} to {error_rates[-1]:.2f}")
        print(f"Code sizes: {code_sizes}")
        
        # Create noise model factory
        if noise_type == 'depolarizing':
            noise_factory = create_depolarizing_noise_factory(dimension)
        elif noise_type == 'symmetric_pauli':
            noise_factory = create_pauli_noise_factory('symmetric')
        elif noise_type == 'dephasing':
            noise_factory = create_pauli_noise_factory('dephasing')
        elif noise_type == 'bitflip':
            noise_factory = create_pauli_noise_factory('bitflip')
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # Collect results for different code sizes
        all_results = {}
        
        for code_size in code_sizes:
            print(f"  Testing code size N={code_size}...")
            final_errors = []
            
            for error_rate in error_rates:
                try:
                    noise_model = noise_factory(error_rate)
                    result = self.protocol.purify_stream(
                        initial_error_rate=error_rate,
                        noise_model=noise_model,
                        num_input_states=code_size
                    )
                    final_errors.append(result.logical_error_evolution[-1])
                except Exception as e:
                    print(f"    Failed at error rate {error_rate:.3f}: {e}")
                    final_errors.append(float('inf'))
            
            all_results[code_size] = np.array(final_errors)
        
        return ThresholdResults(
            error_rates=error_rates,
            final_logical_errors=all_results,
            code_sizes=code_sizes,
            noise_type=noise_type,
            dimension=dimension
        )
    
    def analyze_single_purification_evolution(self,
                                            initial_error_rate: float = 0.3,
                                            noise_type: str = 'depolarizing',
                                            dimension: int = 2,
                                            num_input_states: int = 16) -> Dict[str, Any]:
        """
        Analyze the evolution of logical error through purification levels.
        Similar to Grafe et al. Figure 4(a).
        """
        if noise_type == 'depolarizing':
            noise_factory = create_depolarizing_noise_factory(dimension)
        else:
            noise_factory = create_pauli_noise_factory(noise_type.replace('_pauli', ''))
        
        noise_model = noise_factory(initial_error_rate)
        
        result = self.protocol.purify_stream(
            initial_error_rate=initial_error_rate,
            noise_model=noise_model,
            num_input_states=num_input_states
        )
        
        levels = np.arange(len(result.logical_error_evolution))
        
        return {
            'levels': levels,
            'logical_errors': result.logical_error_evolution,
            'fidelities': result.fidelity_evolution,
            'purities': result.purity_evolution,
            'initial_error_rate': initial_error_rate,
            'noise_type': noise_type,
            'final_error': result.logical_error_evolution[-1],
            'error_reduction_ratio': result.logical_error_evolution[-1] / result.logical_error_evolution[0]
        }


class ThresholdPlotter:
    """Creates publication-quality plots for threshold analysis."""
    
    def plot_grafe_style_figure4(self, 
                                evolution_data: Dict[str, Any],
                                threshold_results: ThresholdResults,
                                save_path: str = "figures/logical_vs_purification.pdf") -> plt.Figure:
        """
        Create a figure analogous to Grafe et al. Figure 4.
        
        Panel (a): Logical error vs purification levels
        Panel (b): Logical error vs physical error rate for different code sizes
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Panel (a): Error evolution through purification levels
        levels = evolution_data['levels']
        logical_errors = evolution_data['logical_errors']
        
        ax1.semilogy(levels, logical_errors, 'o-', linewidth=2, markersize=6, 
                     color='blue', label='Streaming Purification')
        
        ax1.set_xlabel('Purification Level')
        ax1.set_ylabel('Logical Error Rate')
        ax1.set_title(f'(a) Error Evolution\n({evolution_data["noise_type"]} noise, ' +
                     f'p = {evolution_data["initial_error_rate"]:.2f})')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add error reduction annotation
        initial_error = logical_errors[0]
        final_error = logical_errors[-1]
        reduction_ratio = final_error / initial_error
        ax1.text(0.5, 0.95, f'Error reduction: {reduction_ratio:.2e}', 
                transform=ax1.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Panel (b): Threshold behavior
        error_rates = threshold_results.error_rates
        code_sizes = threshold_results.code_sizes
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(code_sizes)))
        
        for i, code_size in enumerate(code_sizes):
            final_errors = threshold_results.final_logical_errors[code_size]
            
            # Filter out infinite values (failed runs)
            valid_mask = np.isfinite(final_errors)
            valid_error_rates = error_rates[valid_mask]
            valid_final_errors = final_errors[valid_mask]
            
            if len(valid_final_errors) > 0:
                ax2.semilogy(valid_error_rates, valid_final_errors, 'o-', 
                           color=colors[i], label=f'N = {code_size}', linewidth=2)
        
        # Add reference line for no error correction
        ax2.semilogy(error_rates, error_rates, '--', color='gray', alpha=0.7, 
                    label='No correction')
        
        ax2.set_xlabel('Physical Error Rate')
        ax2.set_ylabel('Final Logical Error Rate')
        ax2.set_title(f'(b) Threshold Analysis\n({threshold_results.noise_type} noise)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_comparison_different_noise_types(self,
                                            error_rates: np.ndarray,
                                            results_dict: Dict[str, ThresholdResults],
                                            code_size: int = 16,
                                            save_path: str = "figures/comparison_noise_types.pdf") -> plt.Figure:
        """
        Compare threshold behavior for different noise types.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (noise_type, results) in enumerate(results_dict.items()):
            if code_size in results.final_logical_errors:
                final_errors = results.final_logical_errors[code_size]
                
                # Filter out infinite values
                valid_mask = np.isfinite(final_errors)
                valid_error_rates = error_rates[valid_mask]
                valid_final_errors = final_errors[valid_mask]
                
                if len(valid_final_errors) > 0:
                    ax.semilogy(valid_error_rates, valid_final_errors, 'o-',
                              color=colors[i % len(colors)], label=noise_type,
                              linewidth=2, markersize=4)
        
        # Add reference line
        ax.semilogy(error_rates, error_rates, '--', color='gray', alpha=0.7,
                   label='No correction')
        
        ax.set_xlabel('Physical Error Rate')
        ax.set_ylabel('Final Logical Error Rate')
        ax.set_title(f'Threshold Comparison (N = {code_size})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison figure saved to {save_path}")
        
        return fig


def run_complete_threshold_analysis(save_plots: bool = True) -> Dict[str, Any]:
    """
    Run complete threshold analysis and generate plots.
    This is the main function to reproduce Figure 4 analog.
    """
    analyzer = ThresholdAnalyzer()
    plotter = ThresholdPlotter()
    
    print("=== Streaming QEC Threshold Analysis ===")
    
    # Define analysis parameters
    error_rates = np.linspace(0.05, 0.9, 15)
    code_sizes = [4, 8, 16, 32]
    
    # 1. Single purification evolution analysis
    print("\n1. Analyzing purification evolution...")
    evolution_data = analyzer.analyze_single_purification_evolution(
        initial_error_rate=0.3,
        noise_type='depolarizing',
        num_input_states=16
    )
    
    # 2. Threshold sweep for depolarizing noise
    print("\n2. Running threshold sweep for depolarizing noise...")
    depolarizing_results = analyzer.run_threshold_sweep(
        noise_type='depolarizing',
        error_rates=error_rates,
        code_sizes=code_sizes
    )
    
    # 3. Threshold sweep for Pauli noise (if desired)
    print("\n3. Running threshold sweep for symmetric Pauli noise...")
    pauli_results = analyzer.run_threshold_sweep(
        noise_type='symmetric_pauli',
        error_rates=error_rates,
        code_sizes=code_sizes
    )
    
    # 4. Generate main Figure 4 analog
    print("\n4. Generating plots...")
    fig_main = plotter.plot_grafe_style_figure4(
        evolution_data,
        depolarizing_results,
        save_path='figures/logical_vs_purification.pdf' if save_plots else None
    )
    
    # 5. Generate noise comparison plot
    results_dict = {
        'Depolarizing': depolarizing_results,
        'Symmetric Pauli': pauli_results
    }
    
    fig_comparison = plotter.plot_comparison_different_noise_types(
        error_rates,
        results_dict,
        code_size=16,
        save_path='figures/comparison_noise_types.pdf' if save_plots else None
    )
    
    plt.show()
    
    print("\n=== Analysis Complete ===")
    print(f"Final error reduction (depolarizing, p=0.3): {evolution_data['error_reduction_ratio']:.2e}")
    
    return {
        'evolution_data': evolution_data,
        'depolarizing_results': depolarizing_results,
        'pauli_results': pauli_results,
        'figures': {'main': fig_main, 'comparison': fig_comparison}
    }


if __name__ == "__main__":
    # Run the complete analysis
    results = run_complete_threshold_analysis()