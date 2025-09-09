"""
Resource Scaling Analysis for Streaming Purification QEC

Analyzes the practical resource requirements for implementing the streaming
purification protocol, including gate complexity, memory overhead, coherence
time requirements, and scaling with problem size.

Key Achievement: Provides concrete resource estimates for practical implementation
and identifies bottlenecks that may limit real-world performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
import warnings
from purification_simulator import StreamingPurificationSimulator, QuantumState

@dataclass
class ResourceMetrics:
    """Resource requirements for a specific configuration."""
    protocol_config: str
    dimension: int
    purification_levels: int
    total_gate_count: int
    quantum_memory_qubits: int
    classical_memory_bits: int
    coherence_time_required: float  # In arbitrary units
    wall_clock_time: float  # In arbitrary units
    success_probability: float
    error_rate: float

class ResourceAnalyzer:
    """
    Analyze resource requirements for streaming purification protocol.
    
    This class provides detailed analysis of computational resources needed
    for practical implementation of the streaming purification QEC protocol.
    """
    
    def __init__(self, data_dir: str = "./data/", verbose: bool = False):
        """Initialize resource analyzer with data saving."""
        self.data_dir = data_dir
        self.verbose = verbose
        
        # Create data directories
        self.resource_dir = os.path.join(data_dir, "resource_analysis")
        self.scaling_dir = os.path.join(data_dir, "resource_scaling")
        self.overhead_dir = os.path.join(data_dir, "overhead_analysis")
        
        for dir_path in [self.resource_dir, self.scaling_dir, self.overhead_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Gate operation costs (in arbitrary time units)
        self.gate_costs = {
            'single_qubit': 1.0,
            'two_qubit': 10.0,
            'controlled_swap': 20.0,
            'measurement': 5.0,
            'classical_processing': 0.1
        }
        
        # Memory coherence parameters
        self.coherence_params = {
            'T1': 100.0,  # Relaxation time
            'T2': 50.0,   # Dephasing time  
            'gate_time': 1.0  # Single gate time
        }
        
        if self.verbose:
            print(f"ResourceAnalyzer initialized with data directory: {data_dir}")
    
    def analyze_gate_complexity(self, dimension: int, purification_levels: int, 
                               error_rate: float) -> Dict:
        """
        Analyze detailed gate complexity for streaming purification.
        
        Args:
            dimension: Quantum system dimension
            purification_levels: Number of recursive purification levels
            error_rate: Physical error rate
            
        Returns:
            Detailed gate complexity breakdown
        """
        if self.verbose:
            print(f"Analyzing gate complexity: d={dimension}, levels={purification_levels}, ε={error_rate:.3f}")
        
        # Initialize simulator
        simulator = StreamingPurificationSimulator(dimension=dimension, verbose=False)
        
        gate_analysis = {
            'dimension': dimension,
            'purification_levels': purification_levels,
            'error_rate': error_rate,
            'level_breakdown': [],
            'total_gates': 0,
            'gate_type_counts': {
                'swap_tests': 0,
                'hadamard_gates': 0,
                'controlled_swaps': 0,
                'measurements': 0,
                'amplitude_amplification': 0
            },
            'time_estimates': {
                'total_time': 0.0,
                'swap_test_time': 0.0,
                'amplitude_amp_time': 0.0,
                'measurement_time': 0.0
            }
        }
        
        # Analyze each level
        current_purity = 1 - error_rate
        
        for level in range(purification_levels):
            level_data = {
                'level': level,
                'input_purity': current_purity,
                'num_parallel_operations': 2**(purification_levels - level - 1),
                'gates_per_operation': {},
                'total_level_gates': 0,
                'amplification_iterations': 0
            }
            
            # Calculate success probability and amplification needs
            state = QuantumState(current_purity, dimension)
            success_prob = simulator.swap_simulator.calculate_success_probability(state, state)
            
            # Amplitude amplification iterations needed
            if success_prob < 1.0:
                theta = 2 * np.arcsin(np.sqrt(success_prob))
                amp_iterations = max(0, int(np.pi / (2 * theta) - 0.5))
            else:
                amp_iterations = 0
            
            level_data['amplification_iterations'] = amp_iterations
            
            # Basic swap test gates per operation
            swap_gates = {
                'hadamard': 2,  # Initial and final Hadamard
                'controlled_swap': 1,
                'measurement': 1
            }
            
            # Additional gates for amplitude amplification
            amp_gates = amp_iterations * 4  # Approximate for reflection operators
            
            level_data['gates_per_operation'] = {
                'basic_swap': sum(swap_gates.values()),
                'amplitude_amplification': amp_gates,
                'total_per_operation': sum(swap_gates.values()) + amp_gates
            }
            
            # Total gates for this level
            level_gates = level_data['num_parallel_operations'] * level_data['gates_per_operation']['total_per_operation']
            level_data['total_level_gates'] = level_gates
            
            # Update global counters
            gate_analysis['total_gates'] += level_gates
            gate_analysis['gate_type_counts']['swap_tests'] += level_data['num_parallel_operations']
            gate_analysis['gate_type_counts']['hadamard_gates'] += level_data['num_parallel_operations'] * 2
            gate_analysis['gate_type_counts']['controlled_swaps'] += level_data['num_parallel_operations']
            gate_analysis['gate_type_counts']['measurements'] += level_data['num_parallel_operations']
            gate_analysis['gate_type_counts']['amplitude_amplification'] += level_data['num_parallel_operations'] * amp_iterations
            
            # Time estimates
            swap_time = level_data['num_parallel_operations'] * (
                2 * self.gate_costs['single_qubit'] +  # Hadamards
                self.gate_costs['controlled_swap'] + 
                self.gate_costs['measurement']
            )
            amp_time = level_data['num_parallel_operations'] * amp_iterations * 4 * self.gate_costs['single_qubit']
            
            gate_analysis['time_estimates']['swap_test_time'] += swap_time
            gate_analysis['time_estimates']['amplitude_amp_time'] += amp_time
            gate_analysis['time_estimates']['measurement_time'] += level_data['num_parallel_operations'] * self.gate_costs['measurement']
            
            gate_analysis['level_breakdown'].append(level_data)
            
            # Update purity for next level
            current_purity = simulator.swap_simulator._compute_output_purity(current_purity)
            
            if self.verbose:
                print(f"  Level {level}: {level_data['num_parallel_operations']} ops, "
                      f"{level_gates} gates, {amp_iterations} amp iterations")
        
        gate_analysis['time_estimates']['total_time'] = sum(gate_analysis['time_estimates'].values())
        
        return gate_analysis
    
    def analyze_memory_requirements(self, num_logical_qubits: int, purification_levels: int,
                                  dimension: int) -> Dict:
        """
        Analyze quantum and classical memory requirements.
        
        Args:
            num_logical_qubits: Number of logical qubits to protect
            purification_levels: Depth of purification tree
            dimension: System dimension
            
        Returns:
            Detailed memory analysis
        """
        memory_analysis = {
            'num_logical_qubits': num_logical_qubits,
            'purification_levels': purification_levels,
            'dimension': dimension,
            'quantum_memory': {
                'logical_qubits': num_logical_qubits * int(np.log2(dimension)),
                'purification_workspace': purification_levels,
                'amplitude_amplification_ancillas': 1,  # One ancilla per swap test
                'total_qubits': 0
            },
            'classical_memory': {
                'syndrome_storage': purification_levels * num_logical_qubits,
                'amplitude_tracking': purification_levels * 8,  # 8 bytes per amplitude
                'control_logic': 1024,  # 1KB for control logic
                'total_bytes': 0
            },
            'memory_scaling': {}
        }
        
        # Calculate totals
        quantum_total = (memory_analysis['quantum_memory']['logical_qubits'] +
                        memory_analysis['quantum_memory']['purification_workspace'] +
                        memory_analysis['quantum_memory']['amplitude_amplification_ancillas'])
        memory_analysis['quantum_memory']['total_qubits'] = quantum_total
        
        classical_total = (memory_analysis['classical_memory']['syndrome_storage'] +
                          memory_analysis['classical_memory']['amplitude_tracking'] +
                          memory_analysis['classical_memory']['control_logic'])
        memory_analysis['classical_memory']['total_bytes'] = classical_total
        
        # Scaling analysis
        memory_analysis['memory_scaling'] = {
            'quantum_vs_logical': quantum_total / num_logical_qubits,
            'quantum_vs_levels': quantum_total / purification_levels,
            'classical_vs_logical': classical_total / num_logical_qubits,
            'total_memory_efficiency': quantum_total / (num_logical_qubits * dimension)
        }
        
        return memory_analysis
    
    def analyze_coherence_requirements(self, gate_analysis: Dict, dimension: int) -> Dict:
        """
        Analyze quantum coherence time requirements.
        
        Args:
            gate_analysis: Output from analyze_gate_complexity
            dimension: System dimension
            
        Returns:
            Coherence requirement analysis
        """
        coherence_analysis = {
            'dimension': dimension,
            'total_protocol_time': gate_analysis['time_estimates']['total_time'],
            'longest_coherence_requirement': 0.0,
            'level_coherence_requirements': [],
            'decoherence_impact': {},
            'feasibility_assessment': {}
        }
        
        # Analyze coherence requirements per level
        for level_data in gate_analysis['level_breakdown']:
            level_time = (level_data['gates_per_operation']['total_per_operation'] * 
                         self.gate_costs['single_qubit'])
            
            coherence_req = {
                'level': level_data['level'],
                'level_duration': level_time,
                'required_T1': level_time * 10,  # Heuristic: need 10x longer T1
                'required_T2': level_time * 5,   # Heuristic: need 5x longer T2
                'decoherence_probability': self._estimate_decoherence_probability(level_time)
            }
            
            coherence_analysis['level_coherence_requirements'].append(coherence_req)
            
            if coherence_req['required_T1'] > coherence_analysis['longest_coherence_requirement']:
                coherence_analysis['longest_coherence_requirement'] = coherence_req['required_T1']
        
        # Overall decoherence impact
        total_decoherence_prob = sum(req['decoherence_probability'] 
                                   for req in coherence_analysis['level_coherence_requirements'])
        
        coherence_analysis['decoherence_impact'] = {
            'total_decoherence_probability': min(1.0, total_decoherence_prob),
            'effective_error_increase': total_decoherence_prob * 0.1,  # Heuristic
            'protocol_still_viable': total_decoherence_prob < 0.5
        }
        
        # Feasibility assessment
        coherence_analysis['feasibility_assessment'] = {
            'feasible_with_current_hardware': coherence_analysis['longest_coherence_requirement'] < self.coherence_params['T1'],
            'required_improvement_factor': coherence_analysis['longest_coherence_requirement'] / self.coherence_params['T1'],
            'recommendation': self._generate_coherence_recommendation(coherence_analysis)
        }
        
        return coherence_analysis
    
    def _estimate_decoherence_probability(self, operation_time: float) -> float:
        """Estimate probability of decoherence during operation."""
        # Simple exponential decay model
        T1 = self.coherence_params['T1']
        T2 = self.coherence_params['T2']
        
        # Probability of T1 decay
        p_T1 = 1 - np.exp(-operation_time / T1)
        
        # Probability of T2 decay  
        p_T2 = 1 - np.exp(-operation_time / T2)
        
        # Combined probability (approximate)
        return min(1.0, p_T1 + p_T2)
    
    def _generate_coherence_recommendation(self, coherence_analysis: Dict) -> str:
        """Generate recommendation based on coherence analysis."""
        if coherence_analysis['feasibility_assessment']['feasible_with_current_hardware']:
            return "Protocol feasible with current quantum hardware"
        elif coherence_analysis['feasibility_assessment']['required_improvement_factor'] < 10:
            return "Protocol feasible with modest hardware improvements"
        else:
            return "Protocol requires significant hardware advances"
    
    def scaling_study(self, dimensions: List[int], level_ranges: List[int], 
                     error_rates: List[float]) -> Dict:
        """
        Comprehensive resource scaling study across multiple parameters.
        
        Args:
            dimensions: List of system dimensions to analyze
            level_ranges: List of purification levels to test
            error_rates: List of error rates to consider
            
        Returns:
            Complete scaling analysis data
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        scaling_data = {
            'timestamp': timestamp,
            'dimensions': dimensions,
            'level_ranges': level_ranges,
            'error_rates': error_rates,
            'scaling_results': {},
            'scaling_arrays': {},
            'summary_statistics': {}
        }
        
        if self.verbose:
            print("Starting comprehensive resource scaling study")
            print(f"Testing {len(dimensions)} dimensions, {len(level_ranges)} level ranges, {len(error_rates)} error rates")
        
        # Initialize arrays for analysis
        total_configurations = len(dimensions) * len(level_ranges) * len(error_rates)
        
        results_array = np.zeros((len(dimensions), len(level_ranges), len(error_rates)), dtype=object)
        gate_counts = np.zeros((len(dimensions), len(level_ranges), len(error_rates)))
        memory_requirements = np.zeros((len(dimensions), len(level_ranges), len(error_rates)))
        coherence_requirements = np.zeros((len(dimensions), len(level_ranges), len(error_rates)))
        
        config_count = 0
        
        for d_idx, d in enumerate(dimensions):
            for l_idx, levels in enumerate(level_ranges):
                for e_idx, error_rate in enumerate(error_rates):
                    config_count += 1
                    
                    if self.verbose:
                        print(f"  Configuration {config_count}/{total_configurations}: "
                              f"d={d}, levels={levels}, ε={error_rate:.3f}")
                    
                    try:
                        # Gate complexity analysis
                        gate_analysis = self.analyze_gate_complexity(d, levels, error_rate)
                        
                        # Memory analysis
                        memory_analysis = self.analyze_memory_requirements(1, levels, d)
                        
                        # Coherence analysis
                        coherence_analysis = self.analyze_coherence_requirements(gate_analysis, d)
                        
                        # Combine results
                        combined_result = {
                            'dimension': d,
                            'levels': levels,
                            'error_rate': error_rate,
                            'gate_analysis': gate_analysis,
                            'memory_analysis': memory_analysis,
                            'coherence_analysis': coherence_analysis,
                            'resource_metrics': ResourceMetrics(
                                protocol_config=f"d{d}_l{levels}_e{error_rate:.3f}",
                                dimension=d,
                                purification_levels=levels,
                                total_gate_count=gate_analysis['total_gates'],
                                quantum_memory_qubits=memory_analysis['quantum_memory']['total_qubits'],
                                classical_memory_bits=memory_analysis['classical_memory']['total_bytes'],
                                coherence_time_required=coherence_analysis['longest_coherence_requirement'],
                                wall_clock_time=gate_analysis['time_estimates']['total_time'],
                                success_probability=0.99,  # Approximate for high-fidelity protocol
                                error_rate=error_rate
                            )
                        }
                        
                        results_array[d_idx, l_idx, e_idx] = combined_result
                        gate_counts[d_idx, l_idx, e_idx] = gate_analysis['total_gates']
                        memory_requirements[d_idx, l_idx, e_idx] = memory_analysis['quantum_memory']['total_qubits']
                        coherence_requirements[d_idx, l_idx, e_idx] = coherence_analysis['longest_coherence_requirement']
                        
                    except Exception as e:
                        if self.verbose:
                            print(f"    Error in configuration: {e}")
                        
                        # Fill with failure indicators
                        gate_counts[d_idx, l_idx, e_idx] = np.inf
                        memory_requirements[d_idx, l_idx, e_idx] = np.inf
                        coherence_requirements[d_idx, l_idx, e_idx] = np.inf
        
        # Store results
        scaling_data['scaling_results'] = results_array
        scaling_data['scaling_arrays'] = {
            'dimensions': np.array(dimensions),
            'level_ranges': np.array(level_ranges),
            'error_rates': np.array(error_rates),
            'gate_counts': gate_counts,
            'memory_requirements': memory_requirements,
            'coherence_requirements': coherence_requirements
        }
        
        # Generate summary statistics
        scaling_data['summary_statistics'] = self._generate_scaling_statistics(scaling_data)
        
        # Save data
        self._save_scaling_data(scaling_data)
        
        return scaling_data
    
    def _generate_scaling_statistics(self, scaling_data: Dict) -> Dict:
        """Generate summary statistics from scaling study."""
        arrays = scaling_data['scaling_arrays']
        
        stats = {
            'gate_complexity_scaling': {},
            'memory_scaling': {},
            'coherence_scaling': {},
            'bottleneck_analysis': {}
        }
        
        # Gate complexity scaling
        finite_gates = arrays['gate_counts'][np.isfinite(arrays['gate_counts'])]
        if len(finite_gates) > 0:
            stats['gate_complexity_scaling'] = {
                'min_gates': float(np.min(finite_gates)),
                'max_gates': float(np.max(finite_gates)),
                'median_gates': float(np.median(finite_gates)),
                'mean_gates': float(np.mean(finite_gates)),
                'std_gates': float(np.std(finite_gates))
            }
        
        # Memory scaling
        finite_memory = arrays['memory_requirements'][np.isfinite(arrays['memory_requirements'])]
        if len(finite_memory) > 0:
            stats['memory_scaling'] = {
                'min_memory': float(np.min(finite_memory)),
                'max_memory': float(np.max(finite_memory)),
                'median_memory': float(np.median(finite_memory)),
                'mean_memory': float(np.mean(finite_memory)),
                'std_memory': float(np.std(finite_memory))
            }
        
        # Coherence scaling
        finite_coherence = arrays['coherence_requirements'][np.isfinite(arrays['coherence_requirements'])]
        if len(finite_coherence) > 0:
            stats['coherence_scaling'] = {
                'min_coherence': float(np.min(finite_coherence)),
                'max_coherence': float(np.max(finite_coherence)),
                'median_coherence': float(np.median(finite_coherence)),
                'mean_coherence': float(np.mean(finite_coherence)),
                'std_coherence': float(np.std(finite_coherence))
            }
        
        # Bottleneck analysis
        stats['bottleneck_analysis'] = {
            'gate_count_bottleneck_fraction': np.sum(arrays['gate_counts'] > 1e6) / arrays['gate_counts'].size,
            'memory_bottleneck_fraction': np.sum(arrays['memory_requirements'] > 100) / arrays['memory_requirements'].size,
            'coherence_bottleneck_fraction': np.sum(arrays['coherence_requirements'] > 1000) / arrays['coherence_requirements'].size
        }
        
        return stats
    
    def _save_scaling_data(self, scaling_data: Dict):
        """Save scaling study data."""
        timestamp = scaling_data['timestamp']
        
        # Save complete data
        filepath = os.path.join(self.scaling_dir, f"resource_scaling_study_{timestamp}.npz")
        
        # Prepare data for saving (handle object arrays)
        save_data = {
            'timestamp': timestamp,
            'dimensions': scaling_data['dimensions'],
            'level_ranges': scaling_data['level_ranges'],
            'error_rates': scaling_data['error_rates'],
            'gate_counts': scaling_data['scaling_arrays']['gate_counts'],
            'memory_requirements': scaling_data['scaling_arrays']['memory_requirements'],
            'coherence_requirements': scaling_data['scaling_arrays']['coherence_requirements'],
            'summary_statistics': scaling_data['summary_statistics']
        }
        
        np.savez_compressed(filepath, **save_data)
        
        # Save summary as JSON
        summary_filepath = os.path.join(self.scaling_dir, f"scaling_summary_{timestamp}.json")
        with open(summary_filepath, 'w') as f:
            json.dump({
                'summary_statistics': scaling_data['summary_statistics'],
                'metadata': {
                    'timestamp': timestamp,
                    'total_configurations': len(scaling_data['dimensions']) * len(scaling_data['level_ranges']) * len(scaling_data['error_rates']),
                    'dimensions_tested': scaling_data['dimensions'],
                    'levels_tested': scaling_data['level_ranges'],
                    'error_rates_tested': scaling_data['error_rates']
                }
            }, f, indent=2, default=str)
        
        if self.verbose:
            print(f"Scaling data saved:")
            print(f"  Complete data: {filepath}")
            print(f"  Summary: {summary_filepath}")

def run_comprehensive_resource_analysis(data_dir: str = "./data/") -> Dict:
    """Run comprehensive resource analysis with data saving."""
    
    print("="*70)
    print("COMPREHENSIVE RESOURCE SCALING ANALYSIS")
    print("Streaming Purification QEC Resource Requirements")
    print("="*70)
    
    analyzer = ResourceAnalyzer(data_dir=data_dir, verbose=True)
    
    # Define parameter ranges for scaling study
    dimensions = [2, 3, 4, 6, 8]
    level_ranges = [2, 3, 4, 5, 6, 8]
    error_rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    
    # Run comprehensive scaling study
    scaling_results = analyzer.scaling_study(dimensions, level_ranges, error_rates)
    
    # Generate specific analysis examples
    print("\n" + "="*50)
    print("DETAILED ANALYSIS EXAMPLES")
    print("="*50)
    
    # Example 1: Qubit system with moderate parameters
    print("\nExample 1: Qubit system (d=2, 4 levels, ε=0.1)")
    gate_analysis = analyzer.analyze_gate_complexity(dimension=2, purification_levels=4, error_rate=0.1)
    memory_analysis = analyzer.analyze_memory_requirements(num_logical_qubits=1, purification_levels=4, dimension=2)
    coherence_analysis = analyzer.analyze_coherence_requirements(gate_analysis, dimension=2)
    
    print(f"  Total gates: {gate_analysis['total_gates']}")
    print(f"  Quantum memory: {memory_analysis['quantum_memory']['total_qubits']} qubits")
    print(f"  Protocol time: {gate_analysis['time_estimates']['total_time']:.1f} time units")
    print(f"  Coherence requirement: {coherence_analysis['longest_coherence_requirement']:.1f}")
    print(f"  Feasible: {coherence_analysis['feasibility_assessment']['feasible_with_current_hardware']}")
    
    # Example 2: Higher dimension system
    print("\nExample 2: Qutrit system (d=3, 5 levels, ε=0.05)")
    gate_analysis_2 = analyzer.analyze_gate_complexity(dimension=3, purification_levels=5, error_rate=0.05)
    memory_analysis_2 = analyzer.analyze_memory_requirements(num_logical_qubits=1, purification_levels=5, dimension=3)
    
    print(f"  Total gates: {gate_analysis_2['total_gates']}")
    print(f"  Quantum memory: {memory_analysis_2['quantum_memory']['total_qubits']} qubits")
    print(f"  Protocol time: {gate_analysis_2['time_estimates']['total_time']:.1f} time units")
    
    # Print scaling summary
    stats = scaling_results['summary_statistics']
    print("\n" + "="*50)
    print("SCALING STUDY SUMMARY")
    print("="*50)
    
    if 'gate_complexity_scaling' in stats:
        gc = stats['gate_complexity_scaling']
        print(f"Gate Complexity Range: {gc['min_gates']:.0f} - {gc['max_gates']:.0f}")
        print(f"Median Gate Count: {gc['median_gates']:.0f}")
    
    if 'memory_scaling' in stats:
        ms = stats['memory_scaling']
        print(f"Memory Range: {ms['min_memory']:.0f} - {ms['max_memory']:.0f} qubits")
        print(f"Median Memory: {ms['median_memory']:.0f} qubits")
    
    if 'bottleneck_analysis' in stats:
        ba = stats['bottleneck_analysis']
        print(f"Gate Count Bottleneck: {ba['gate_count_bottleneck_fraction']:.1%} of configurations")
        print(f"Memory Bottleneck: {ba['memory_bottleneck_fraction']:.1%} of configurations")
        print(f"Coherence Bottleneck: {ba['coherence_bottleneck_fraction']:.1%} of configurations")
    
    print(f"\nResults saved to: {data_dir}/resource_analysis/")
    
    return scaling_results

if __name__ == "__main__":
    results = run_comprehensive_resource_analysis()