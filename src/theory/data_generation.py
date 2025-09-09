"""
Systematic data generation for streaming QEC protocol.
Generates and saves all data needed for paper figures.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import itertools

from src.streaming_protocol import StreamingPurificationProtocol, create_depolarizing_noise_factory, create_pauli_noise_factory


@dataclass
class EvolutionData:
    """Data for error/fidelity evolution through purification levels."""
    noise_type: str
    dimension: int
    N: int
    physical_error_rate: float
    iterations: List[int]
    logical_errors: List[float]
    fidelities: List[float]
    purities: List[float]
    total_swap_operations: int
    total_amplification_iterations: int
    memory_levels_used: int


@dataclass
class ThresholdData:
    """Data for threshold analysis."""
    noise_type: str
    dimension: int
    N: int
    physical_error_rates: List[float]
    final_logical_errors: List[float]
    initial_logical_errors: List[float]
    error_reduction_ratios: List[float]


@dataclass
class ResourceData:
    """Resource overhead data."""
    noise_type: str
    dimension: int
    N: int
    physical_error_rate: float
    total_swap_operations: int
    total_amplification_iterations: int
    memory_levels_used: int
    theoretical_memory: int  # log2(N)
    memory_efficiency: float


class StreamingQECDataGenerator:
    """Main data generation class for streaming QEC protocol."""
    
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = data_dir
        self.protocol = StreamingPurificationProtocol()
        
        # Create data directory structure
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "evolution"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "threshold"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "resources"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)
    
    def generate_evolution_data(self, 
                              noise_types: List[str] = None,
                              dimensions: List[int] = None,
                              N_values: List[int] = None,
                              physical_error_rates: List[float] = None) -> List[EvolutionData]:
        """
        Generate evolution data for errors/fidelity vs iterations plots.
        
        Args:
            noise_types: ['depolarizing', 'symmetric_pauli', 'dephasing', 'bitflip']
            dimensions: [2, 3, 4, 5] for qudits
            N_values: [4, 8, 16, 32, 64] number of input states
            physical_error_rates: [0.1, 0.3, 0.5, 0.7] initial error rates to test
        """
        if noise_types is None:
            noise_types = ['depolarizing', 'symmetric_pauli', 'dephasing']
        if dimensions is None:
            dimensions = [2, 3, 4]
        if N_values is None:
            N_values = [8, 16, 32]
        if physical_error_rates is None:
            physical_error_rates = [0.1, 0.3, 0.5, 0.7]
        
        evolution_data = []
        total_runs = len(noise_types) * len(dimensions) * len(N_values) * len(physical_error_rates)
        current_run = 0
        
        print(f"Generating evolution data: {total_runs} total runs...")
        
        for noise_type, dimension, N, error_rate in itertools.product(
            noise_types, dimensions, N_values, physical_error_rates):
            
            current_run += 1
            print(f"  Run {current_run}/{total_runs}: {noise_type}, d={dimension}, N={N}, p={error_rate:.2f}")
            
            try:
                # Skip Pauli noise for d > 2
                if noise_type != 'depolarizing' and dimension > 2:
                    continue
                
                # Create appropriate noise model
                if noise_type == 'depolarizing':
                    noise_factory = create_depolarizing_noise_factory(dimension)
                else:
                    noise_factory = create_pauli_noise_factory(noise_type.replace('_pauli', ''))
                
                noise_model = noise_factory(error_rate)
                
                # Run purification
                result = self.protocol.purify_stream(
                    initial_error_rate=error_rate,
                    noise_model=noise_model,
                    num_input_states=N
                )
                
                # Store evolution data
                data = EvolutionData(
                    noise_type=noise_type,
                    dimension=dimension,
                    N=N,
                    physical_error_rate=error_rate,
                    iterations=list(range(len(result.logical_error_evolution))),
                    logical_errors=result.logical_error_evolution,
                    fidelities=result.fidelity_evolution,
                    purities=result.purity_evolution,
                    total_swap_operations=result.total_swap_operations,
                    total_amplification_iterations=result.total_amplification_iterations,
                    memory_levels_used=result.memory_levels_used
                )
                evolution_data.append(data)
                
            except Exception as e:
                print(f"    Failed: {e}")
                continue
        
        return evolution_data
    
    def generate_threshold_data(self,
                              noise_types: List[str] = None,
                              dimensions: List[int] = None, 
                              N_values: List[int] = None,
                              error_rate_ranges: Dict[str, np.ndarray] = None) -> List[ThresholdData]:
        """
        Generate threshold analysis data.
        
        Args:
            error_rate_ranges: Custom error rate ranges for different noise types
        """
        if noise_types is None:
            noise_types = ['depolarizing', 'symmetric_pauli']
        if dimensions is None:
            dimensions = [2, 3, 4]
        if N_values is None:
            N_values = [8, 16, 32, 64]
        if error_rate_ranges is None:
            error_rate_ranges = {
                'depolarizing': np.linspace(0.05, 0.95, 15),
                'symmetric_pauli': np.linspace(0.05, 0.6, 12),
                'dephasing': np.linspace(0.05, 0.8, 12),
                'bitflip': np.linspace(0.05, 0.6, 12)
            }
        
        threshold_data = []
        
        print("Generating threshold data...")
        
        for noise_type in noise_types:
            for dimension in dimensions:
                # Skip Pauli noise for d > 2
                if noise_type != 'depolarizing' and dimension > 2:
                    continue
                
                error_rates = error_rate_ranges.get(noise_type, np.linspace(0.05, 0.8, 12))
                
                for N in N_values:
                    print(f"  {noise_type}, d={dimension}, N={N}")
                    
                    final_errors = []
                    initial_errors = []
                    
                    # Create noise factory
                    if noise_type == 'depolarizing':
                        noise_factory = create_depolarizing_noise_factory(dimension)
                    else:
                        noise_factory = create_pauli_noise_factory(noise_type.replace('_pauli', ''))
                    
                    for error_rate in error_rates:
                        try:
                            noise_model = noise_factory(error_rate)
                            result = self.protocol.purify_stream(
                                initial_error_rate=error_rate,
                                noise_model=noise_model,
                                num_input_states=N
                            )
                            
                            initial_errors.append(result.logical_error_evolution[0])
                            final_errors.append(result.logical_error_evolution[-1])
                            
                        except Exception as e:
                            print(f"    Failed at p={error_rate:.3f}: {e}")
                            initial_errors.append(np.nan)
                            final_errors.append(np.nan)
                    
                    # Calculate error reduction ratios
                    error_reduction_ratios = []
                    for init, final in zip(initial_errors, final_errors):
                        if not (np.isnan(init) or np.isnan(final)) and init > 0:
                            error_reduction_ratios.append(final / init)
                        else:
                            error_reduction_ratios.append(np.nan)
                    
                    data = ThresholdData(
                        noise_type=noise_type,
                        dimension=dimension,
                        N=N,
                        physical_error_rates=error_rates.tolist(),
                        final_logical_errors=final_errors,
                        initial_logical_errors=initial_errors,
                        error_reduction_ratios=error_reduction_ratios
                    )
                    threshold_data.append(data)
        
        return threshold_data
    
    def generate_resource_data(self,
                             noise_types: List[str] = None,
                             dimensions: List[int] = None,
                             N_values: List[int] = None,
                             physical_error_rate: float = 0.3) -> List[ResourceData]:
        """Generate resource overhead data."""
        if noise_types is None:
            noise_types = ['depolarizing', 'symmetric_pauli']
        if dimensions is None:
            dimensions = [2, 3, 4]
        if N_values is None:
            N_values = [4, 8, 16, 32, 64, 128]
        
        resource_data = []
        
        print(f"Generating resource data at p={physical_error_rate}...")
        
        for noise_type in noise_types:
            for dimension in dimensions:
                # Skip Pauli noise for d > 2
                if noise_type != 'depolarizing' and dimension > 2:
                    continue
                
                print(f"  {noise_type}, d={dimension}")
                
                # Create noise factory
                if noise_type == 'depolarizing':
                    noise_factory = create_depolarizing_noise_factory(dimension)
                else:
                    noise_factory = create_pauli_noise_factory(noise_type.replace('_pauli', ''))
                
                for N in N_values:
                    try:
                        noise_model = noise_factory(physical_error_rate)
                        result = self.protocol.purify_stream(
                            initial_error_rate=physical_error_rate,
                            noise_model=noise_model,
                            num_input_states=N
                        )
                        
                        theoretical_memory = int(np.log2(N)) if N > 1 else 1
                        memory_efficiency = result.memory_levels_used / max(theoretical_memory, 1)
                        
                        data = ResourceData(
                            noise_type=noise_type,
                            dimension=dimension,
                            N=N,
                            physical_error_rate=physical_error_rate,
                            total_swap_operations=result.total_swap_operations,
                            total_amplification_iterations=result.total_amplification_iterations,
                            memory_levels_used=result.memory_levels_used,
                            theoretical_memory=theoretical_memory,
                            memory_efficiency=memory_efficiency
                        )
                        resource_data.append(data)
                        
                    except Exception as e:
                        print(f"    Failed for N={N}: {e}")
                        continue
        
        return resource_data
    
    def save_data(self, evolution_data: List[EvolutionData] = None,
                  threshold_data: List[ThresholdData] = None, 
                  resource_data: List[ResourceData] = None,
                  timestamp: str = None) -> Dict[str, str]:
        """Save all generated data to files."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = {}
        
        # Save evolution data
        if evolution_data:
            # Convert to DataFrame for easy analysis
            df_evolution = pd.DataFrame([asdict(data) for data in evolution_data])
            
            # Save as CSV
            evolution_csv = os.path.join(self.data_dir, "evolution", f"evolution_data_{timestamp}.csv")
            df_evolution.to_csv(evolution_csv, index=False)
            
            # Save as JSON for full precision
            evolution_json = os.path.join(self.data_dir, "evolution", f"evolution_data_{timestamp}.json")
            with open(evolution_json, 'w') as f:
                json.dump([asdict(data) for data in evolution_data], f, indent=2)
            
            saved_files['evolution_csv'] = evolution_csv
            saved_files['evolution_json'] = evolution_json
            print(f"Saved evolution data: {len(evolution_data)} records")
        
        # Save threshold data
        if threshold_data:
            df_threshold = pd.DataFrame([asdict(data) for data in threshold_data])
            
            threshold_csv = os.path.join(self.data_dir, "threshold", f"threshold_data_{timestamp}.csv")
            df_threshold.to_csv(threshold_csv, index=False)
            
            threshold_json = os.path.join(self.data_dir, "threshold", f"threshold_data_{timestamp}.json")
            with open(threshold_json, 'w') as f:
                json.dump([asdict(data) for data in threshold_data], f, indent=2)
            
            saved_files['threshold_csv'] = threshold_csv
            saved_files['threshold_json'] = threshold_json
            print(f"Saved threshold data: {len(threshold_data)} records")
        
        # Save resource data
        if resource_data:
            df_resource = pd.DataFrame([asdict(data) for data in resource_data])
            
            resource_csv = os.path.join(self.data_dir, "resources", f"resource_data_{timestamp}.csv")
            df_resource.to_csv(resource_csv, index=False)
            
            resource_json = os.path.join(self.data_dir, "resources", f"resource_data_{timestamp}.json")
            with open(resource_json, 'w') as f:
                json.dump([asdict(data) for data in resource_data], f, indent=2)
            
            saved_files['resource_csv'] = resource_csv
            saved_files['resource_json'] = resource_json
            print(f"Saved resource data: {len(resource_data)} records")
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'generation_date': datetime.now().isoformat(),
            'evolution_records': len(evolution_data) if evolution_data else 0,
            'threshold_records': len(threshold_data) if threshold_data else 0,
            'resource_records': len(resource_data) if resource_data else 0,
            'files': saved_files
        }
        
        metadata_file = os.path.join(self.data_dir, f"metadata_{timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        saved_files['metadata'] = metadata_file
        
        return saved_files
    
    def generate_all_data(self, 
                         quick_run: bool = False,
                         save_immediately: bool = True) -> Dict[str, Any]:
        """
        Generate all data needed for paper figures.
        
        Args:
            quick_run: If True, use reduced parameter sets for testing
            save_immediately: If True, save data as it's generated
        """
        print("="*60)
        print("STREAMING QEC DATA GENERATION")
        print("="*60)
        
        if quick_run:
            print("Running in QUICK mode with reduced parameters...")
            noise_types = ['depolarizing', 'symmetric_pauli']
            dimensions = [2, 3]
            N_values = [8, 16]
            error_rates = [0.1, 0.3, 0.5]
            N_resource = [4, 8, 16, 32]
        else:
            print("Running FULL data generation...")
            noise_types = ['depolarizing', 'symmetric_pauli', 'dephasing']
            dimensions = [2, 3, 4, 5]
            N_values = [8, 16, 32, 64]
            error_rates = [0.1, 0.2, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
            N_resource = [4, 8, 16, 32, 64, 128]
        
        # Generate all data
        print("\n1. Generating evolution data...")
        evolution_data = self.generate_evolution_data(
            noise_types=noise_types,
            dimensions=dimensions,
            N_values=N_values,
            physical_error_rates=error_rates
        )
        
        print("\n2. Generating threshold data...")
        threshold_data = self.generate_threshold_data(
            noise_types=noise_types,
            dimensions=dimensions,
            N_values=N_values
        )
        
        print("\n3. Generating resource data...")
        resource_data = self.generate_resource_data(
            noise_types=noise_types,
            dimensions=dimensions,
            N_values=N_resource
        )
        
        # Save data
        if save_immediately:
            print("\n4. Saving data...")
            saved_files = self.save_data(evolution_data, threshold_data, resource_data)
            print(f"Data saved to: {self.data_dir}")
            print("Files saved:")
            for key, path in saved_files.items():
                print(f"  {key}: {path}")
        else:
            saved_files = {}
        
        return {
            'evolution_data': evolution_data,
            'threshold_data': threshold_data,
            'resource_data': resource_data,
            'saved_files': saved_files
        }


def main():
    """Main execution function."""
    import sys
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parse command line arguments
    quick_run = '--quick' in sys.argv
    data_dir = "../data"
    
    if '--data-dir' in sys.argv:
        idx = sys.argv.index('--data-dir')
        if idx + 1 < len(sys.argv):
            data_dir = sys.argv[idx + 1]
    
    # Create data generator
    generator = StreamingQECDataGenerator(data_dir)
    
    # Generate all data
    results = generator.generate_all_data(quick_run=quick_run)
    
    print("\n" + "="*60)
    print("DATA GENERATION COMPLETE")
    print("="*60)
    print(f"Evolution records: {len(results['evolution_data'])}")
    print(f"Threshold records: {len(results['threshold_data'])}")
    print(f"Resource records: {len(results['resource_data'])}")
    
    return results


if __name__ == "__main__":
    main()