"""
Example usage of the refactored streaming QEC protocol.
Run this to generate Figure 4 analog results.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.streaming_protocol import StreamingPurificationProtocol, create_depolarizing_noise_factory
from src.threshold_analysis import run_complete_threshold_analysis, ThresholdAnalyzer, ThresholdPlotter


def simple_example():
    """Simple example showing basic protocol usage."""
    print("=== Simple Streaming Purification Example ===")
    
    # Create protocol instance
    protocol = StreamingPurificationProtocol()
    
    # Create depolarizing noise model
    noise_factory = create_depolarizing_noise_factory(dimension=2)  # Qubits
    initial_error_rate = 0.3  # 30% depolarization
    noise_model = noise_factory(initial_error_rate)
    
    # Run streaming purification
    result = protocol.purify_stream(
        initial_error_rate=initial_error_rate,
        noise_model=noise_model,
        num_input_states=16  # 2^4 states for 4 levels of purification
    )
    
    # Display results
    print(f"Initial logical error: {result.logical_error_evolution[0]:.4f}")
    print(f"Final logical error: {result.logical_error_evolution[-1]:.4f}")
    print(f"Error reduction ratio: {result.logical_error_evolution[-1]/result.logical_error_evolution[0]:.2e}")
    print(f"Total swap operations: {result.total_swap_operations}")
    print(f"Memory levels used: {result.memory_levels_used}")
    
    # Plot evolution
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    levels = np.arange(len(result.logical_error_evolution))
    plt.semilogy(levels, result.logical_error_evolution, 'o-', linewidth=2)
    plt.xlabel('Purification Level')
    plt.ylabel('Logical Error Rate')
    plt.title('Error Evolution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(levels, result.purity_evolution, 's-', linewidth=2, color='green')
    plt.xlabel('Purification Level')
    plt.ylabel('Purity Parameter λ')
    plt.title('Purity Evolution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def compare_theoretical_vs_simulation():
    """Compare theoretical predictions with simulation results."""
    print("\n=== Theoretical vs Simulation Comparison ===")
    
    protocol = StreamingPurificationProtocol()
    initial_error_rate = 0.4
    dimension = 2
    num_levels = 5
    
    # Theoretical analysis
    theoretical_errors, theoretical_purities = protocol.theoretical_purification_analysis(
        initial_error_rate=initial_error_rate,
        dimension=dimension,
        num_levels=num_levels,
        noise_type='depolarizing'
    )
    
    # Simulation
    noise_model = create_depolarizing_noise_factory(dimension)(initial_error_rate)
    result = protocol.purify_stream(
        initial_error_rate=initial_error_rate,
        noise_model=noise_model,
        num_input_states=2**num_levels
    )
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    levels_theory = np.arange(len(theoretical_errors))
    levels_sim = np.arange(len(result.logical_error_evolution))
    
    plt.semilogy(levels_theory, theoretical_errors, 'o-', label='Theoretical', linewidth=2)
    plt.semilogy(levels_sim, result.logical_error_evolution, 's-', label='Simulation', linewidth=2)
    plt.xlabel('Purification Level')
    plt.ylabel('Logical Error Rate')
    plt.title('Logical Error Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(levels_theory, theoretical_purities, 'o-', label='Theoretical', linewidth=2)
    plt.plot(levels_sim, result.purity_evolution, 's-', label='Simulation', linewidth=2)
    plt.xlabel('Purification Level')
    plt.ylabel('Purity Parameter λ')
    plt.title('Purity Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Theoretical final error: {theoretical_errors[-1]:.4f}")
    print(f"Simulation final error: {result.logical_error_evolution[-1]:.4f}")
    print(f"Agreement: {abs(theoretical_errors[-1] - result.logical_error_evolution[-1]):.4f}")


def threshold_scan_example():
    """Example of threshold scanning for different code sizes."""
    print("\n=== Threshold Scan Example ===")
    
    analyzer = ThresholdAnalyzer()
    
    # Quick threshold scan
    error_rates = np.linspace(0.1, 0.8, 10)  # Reduced range for quick test
    code_sizes = [4, 8, 16]  # Smaller sizes for quick test
    
    results = analyzer.run_threshold_sweep(
        noise_type='depolarizing',
        error_rates=error_rates,
        code_sizes=code_sizes
    )
    
    # Plot results
    plt.figure(figsize=(8, 6))
    
    colors = ['blue', 'red', 'green']
    for i, code_size in enumerate(code_sizes):
        final_errors = results.final_logical_errors[code_size]
        valid_mask = np.isfinite(final_errors)
        
        plt.semilogy(error_rates[valid_mask], final_errors[valid_mask], 
                    'o-', color=colors[i], label=f'N = {code_size}', linewidth=2)
    
    # Add reference line
    plt.semilogy(error_rates, error_rates, '--', color='gray', alpha=0.7, 
                label='No correction')
    
    plt.xlabel('Physical Error Rate')
    plt.ylabel('Final Logical Error Rate')
    plt.title('Threshold Behavior (Depolarizing Noise)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def demonstrate_memory_scaling():
    """Demonstrate O(log N) memory scaling."""
    print("\n=== Memory Scaling Demonstration ===")
    
    protocol = StreamingPurificationProtocol()
    noise_factory = create_depolarizing_noise_factory(2)
    
    input_sizes = [4, 8, 16, 32, 64, 128]
    memory_levels = []
    
    for size in input_sizes:
        noise_model = noise_factory(0.3)
        result = protocol.purify_stream(
            initial_error_rate=0.3,
            noise_model=noise_model,
            num_input_states=size
        )
        memory_levels.append(result.memory_levels_used)
        print(f"N = {size:3d} → Memory levels: {result.memory_levels_used}")
    
    # Plot scaling
    plt.figure(figsize=(8, 6))
    plt.plot(input_sizes, memory_levels, 'o-', linewidth=2, markersize=8)
    plt.plot(input_sizes, np.log2(input_sizes), '--', alpha=0.7, label='log₂(N)')
    plt.xlabel('Number of Input States (N)')
    plt.ylabel('Memory Levels Used')
    plt.title('Memory Scaling: O(log N)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("Streaming QEC Protocol - Refactored Implementation")
    print("=" * 55)
    
    # Run examples
    simple_example()
    compare_theoretical_vs_simulation()
    threshold_scan_example()
    demonstrate_memory_scaling()
    
    # Option to run complete analysis
    print("\n" + "=" * 55)
    response = input("Run complete threshold analysis for Figure 4? (y/n): ")
    if response.lower() == 'y':
        run_complete_threshold_analysis()
    
    print("\nAll examples completed!")