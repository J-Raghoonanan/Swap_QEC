"""
Performance Comparison for Streaming Purification vs Existing QEC Codes

This module provides detailed performance comparisons between the streaming
purification protocol and established quantum error correction codes across
multiple metrics: sample complexity, memory usage, gate complexity, and
threshold performance.

Key Achievement: Generates quantitative comparisons that can be presented
in tabular form in the paper, showing where streaming purification excels
and where it has limitations.

CRITICAL LIMITATION: Performance comparisons are based on theoretical models
and may not reflect real-world performance under practical constraints.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import warnings
from abc import ABC, abstractmethod

@dataclass
class PerformanceMetrics:
    """Standard performance metrics for QEC code comparison."""
    sample_complexity: float
    memory_complexity: float
    gate_complexity: float
    threshold: float
    final_fidelity: float
    protocol_name: str
    dimension: int
    error_model: str

# [Previous protocol classes remain the same, just adding data saving to the comparator]

class PerformanceComparator:
    """Compare performance with comprehensive data saving."""
    
    def __init__(self, protocols: List[QECProtocol], data_dir: str = "./data/", verbose: bool = False):
        """Initialize with data directory."""
        self.protocols = protocols
        self.data_dir = data_dir
        self.verbose = verbose
        
        # Create data directories
        self.comparison_dir = os.path.join(data_dir, "performance_comparison")
        self.plots_dir = os.path.join("./figures/", "performance_plots")
        self.tables_dir = os.path.join(data_dir, "performance_tables")
        
        for dir_path in [self.comparison_dir, self.plots_dir, self.tables_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def compare_protocols(self, error_rates: List[float], target_fidelity: float = 0.99,
                         num_logical_qubits: int = 1) -> Dict[str, Any]:
        """Compare protocols and save comprehensive data."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        comparison_data = {
            'timestamp': timestamp,
            'protocols': [p.name for p in self.protocols],
            'error_rates': error_rates,
            'target_fidelity': target_fidelity,
            'num_logical_qubits': num_logical_qubits,
            'results': {},
            'performance_arrays': {},
            'summary_table': [],
            'analysis': {}
        }
        
        # Initialize arrays for plotting
        performance_arrays = {
            'error_rates': np.array(error_rates),
            'protocols': {}
        }
        
        # Calculate performance for each protocol
        for protocol in self.protocols:
            if self.verbose:
                print(f"Analyzing {protocol.name}")
            
            protocol_results = []
            sample_complexities = []
            memory_complexities = []
            gate_complexities = []
            thresholds = []
            fidelities = []
            
            for error_rate in error_rates:
                try:
                    metrics = protocol.calculate_performance(
                        error_rate, target_fidelity, num_logical_qubits
                    )
                    protocol_results.append(asdict(metrics))
                    
                    # Collect arrays for plotting
                    sample_complexities.append(metrics.sample_complexity)
                    memory_complexities.append(metrics.memory_complexity)
                    gate_complexities.append(metrics.gate_complexity)
                    thresholds.append(metrics.threshold)
                    fidelities.append(metrics.final_fidelity)
                    
                    if self.verbose:
                        print(f"  ε={error_rate:.3f}: mem={metrics.memory_complexity:.1f}, "
                              f"gates={metrics.gate_complexity:.0f}")
                
                except Exception as e:
                    if self.verbose:
                        print(f"  {protocol.name} failed at ε={error_rate:.3f}: {e}")
                    
                    # Create placeholder
                    failed_metrics = {
                        'sample_complexity': float('inf'),
                        'memory_complexity': float('inf'),
                        'gate_complexity': float('inf'),
                        'threshold': 0.0,
                        'final_fidelity': 0.0,
                        'protocol_name': protocol.name,
                        'dimension': protocol.dimension,
                        'error_model': "failed"
                    }
                    protocol_results.append(failed_metrics)
                    
                    sample_complexities.append(float('inf'))
                    memory_complexities.append(float('inf'))
                    gate_complexities.append(float('inf'))
                    thresholds.append(0.0)
                    fidelities.append(0.0)
            
            comparison_data['results'][protocol.name] = protocol_results
            
            # Store arrays for plotting
            performance_arrays['protocols'][protocol.name] = {
                'sample_complexities': np.array(sample_complexities),
                'memory_complexities': np.array(memory_complexities),
                'gate_complexities': np.array(gate_complexities),
                'thresholds': np.array(thresholds),
                'fidelities': np.array(fidelities)
            }
        
        comparison_data['performance_arrays'] = performance_arrays
        
        # Generate analysis and tables
        comparison_data['summary_table'] = self._generate_summary_table(comparison_data)
        comparison_data['analysis'] = self._analyze_results(comparison_data)
        
        # Save data
        self._save_comparison_data(comparison_data)
        
        return comparison_data
    
    def _save_comparison_data(self, comparison_data: Dict):
        """Save comparison data in multiple formats."""
        timestamp = comparison_data['timestamp']
        
        # 1. Save complete data as NPZ
        filepath_npz = os.path.join(self.comparison_dir, f"performance_comparison_{timestamp}.npz")
        np.savez_compressed(filepath_npz, **comparison_data)
        
        # 2. Save summary table as JSON
        table_filepath = os.path.join(self.tables_dir, f"performance_table_{timestamp}.json")
        with open(table_filepath, 'w') as f:
            json.dump({
                'summary_table': comparison_data['summary_table'],
                'analysis': comparison_data['analysis'],
                'metadata': {
                    'timestamp': timestamp,
                    'protocols': comparison_data['protocols'],
                    'error_rates': comparison_data['error_rates']
                }
            }, f, indent=2, default=str)
        
        # 3. Save plotting arrays separately for easy access
        arrays_filepath = os.path.join(self.comparison_dir, f"plotting_arrays_{timestamp}.npz")
        plot_data = comparison_data['performance_arrays'].copy()
        np.savez_compressed(arrays_filepath, **plot_data)
        
        if self.verbose:
            print(f"Comparison data saved:")
            print(f"  Complete data: {filepath_npz}")
            print(f"  Summary table: {table_filepath}")
            print(f"  Plotting arrays: {arrays_filepath}")
    
    def generate_comparison_plots(self, comparison_data: Dict, save_plots: bool = True) -> Dict[str, str]:
        """Generate and save comparison plots."""
        timestamp = comparison_data['timestamp']
        saved_plots = {}
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        error_rates = comparison_data['performance_arrays']['error_rates']
        protocols = comparison_data['performance_arrays']['protocols']
        
        # Plot 1: Memory complexity vs error rate
        ax1 = axes[0, 0]
        for protocol_name, arrays in protocols.items():
            memory_vals = arrays['memory_complexities']
            # Filter finite values
            finite_mask = np.isfinite(memory_vals)
            if np.any(finite_mask):
                ax1.plot(error_rates[finite_mask], memory_vals[finite_mask], 
                        'o-', label=protocol_name, linewidth=2, markersize=6)
        
        ax1.set_xlabel('Error Rate', fontsize=12)
        ax1.set_ylabel('Memory Complexity (qubits)', fontsize=12)
        ax1.set_title('Memory Scaling Comparison', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Gate complexity vs error rate
        ax2 = axes[0, 1]
        for protocol_name, arrays in protocols.items():
            gate_vals = arrays['gate_complexities']
            finite_mask = np.isfinite(gate_vals)
            if np.any(finite_mask):
                ax2.plot(error_rates[finite_mask], gate_vals[finite_mask], 
                        's-', label=protocol_name, linewidth=2, markersize=6)
        
        ax2.set_xlabel('Error Rate', fontsize=12)
        ax2.set_ylabel('Gate Complexity', fontsize=12)
        ax2.set_title('Gate Complexity Comparison', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Threshold comparison (bar chart)
        ax3 = axes[1, 0]
        protocol_names = []
        threshold_values = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, (protocol_name, arrays) in enumerate(protocols.items()):
            thresholds = arrays['thresholds']
            if len(thresholds) > 0 and thresholds[0] > 0:
                protocol_names.append(protocol_name.replace(' ', '\n'))
                threshold_values.append(thresholds[0] * 100)
        
        if threshold_values:
            bars = ax3.bar(range(len(protocol_names)), threshold_values, 
                          color=colors[:len(protocol_names)])
            ax3.set_xlabel('Protocol', fontsize=12)
            ax3.set_ylabel('Error Threshold (%)', fontsize=12)
            ax3.set_title('Error Threshold Comparison', fontsize=14, fontweight='bold')
            ax3.set_xticks(range(len(protocol_names)))
            ax3.set_xticklabels(protocol_names, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Memory vs Gates trade-off
        ax4 = axes[1, 1]
        for protocol_name, arrays in protocols.items():
            # Use middle error rate for trade-off analysis
            mid_idx = len(error_rates) // 2
            mem_val = arrays['memory_complexities'][mid_idx]
            gate_val = arrays['gate_complexities'][mid_idx]
            
            if np.isfinite(mem_val) and np.isfinite(gate_val):
                ax4.scatter(mem_val, gate_val, s=150, label=protocol_name, alpha=0.7)
        
        ax4.set_xlabel('Memory Complexity (qubits)', fontsize=12)
        ax4.set_ylabel('Gate Complexity', fontsize=12)
        ax4.set_title('Memory vs Gate Trade-off', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_filepath = os.path.join(self.plots_dir, f"performance_comparison_{timestamp}.pdf")
            plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
            saved_plots['comparison'] = plot_filepath
            
            if self.verbose:
                print(f"Comparison plots saved to: {plot_filepath}")
        
        # Generate individual metric plots
        individual_plots = self._generate_individual_plots(comparison_data, save_plots)
        saved_plots.update(individual_plots)
        
        return saved_plots
    
    def _generate_individual_plots(self, comparison_data: Dict, save_plots: bool) -> Dict[str, str]:
        """Generate individual plots for each metric."""
        timestamp = comparison_data['timestamp']
        saved_plots = {}
        
        error_rates = comparison_data['performance_arrays']['error_rates']
        protocols = comparison_data['performance_arrays']['protocols']
        
        metrics = ['memory_complexities', 'gate_complexities', 'sample_complexities']
        titles = ['Memory Complexity vs Error Rate', 'Gate Complexity vs Error Rate', 
                 'Sample Complexity vs Error Rate']
        ylabels = ['Memory (qubits)', 'Gate Count', 'Sample Count']
        
        for metric, title, ylabel in zip(metrics, titles, ylabels):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for protocol_name, arrays in protocols.items():
                values = arrays[metric]
                finite_mask = np.isfinite(values)
                if np.any(finite_mask):
                    ax.plot(error_rates[finite_mask], values[finite_mask], 
                           'o-', label=protocol_name, linewidth=2, markersize=8)
            
            ax.set_xlabel('Error Rate', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend()
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_plots:
                plot_filepath = os.path.join(self.plots_dir, f"{metric}_{timestamp}.pdf")
                plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
                saved_plots[metric] = plot_filepath
        
        return saved_plots

def run_comprehensive_performance_comparison(data_dir: str = "./data/") -> Dict:
    """Run comprehensive performance comparison with full data saving."""
    
    print("="*70)
    print("COMPREHENSIVE PERFORMANCE COMPARISON WITH DATA SAVING")
    print("="*70)
    
    # Define protocols
    protocols = [
        StreamingPurificationProtocol(dimension=2),
        StreamingPurificationProtocol(dimension=4),
        SurfaceCodeProtocol(distance=3),
        SurfaceCodeProtocol(distance=5), 
        SpinorCodeProtocol(dimension=2, num_qubits=10),
        SpinorCodeProtocol(dimension=2, num_qubits=20)
    ]
    
    # Create comparator
    comparator = PerformanceComparator(protocols, data_dir=data_dir, verbose=True)
    
    # Run comparison
    error_rates = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    comparison_results = comparator.compare_protocols(
        error_rates=error_rates,
        target_fidelity=0.99,
        num_logical_qubits=1
    )
    
    # Generate and save plots
    saved_plots = comparator.generate_comparison_plots(comparison_results, save_plots=True)
    
    # Print summary
    print(f"\nResults saved to: {data_dir}/performance_comparison/")
    print(f"Plots saved to: {data_dir}/performance_plots/")
    print(f"Tables saved to: {data_dir}/performance_tables/")
    
    return comparison_results

if __name__ == "__main__":
    results = run_comprehensive_performance_comparison()