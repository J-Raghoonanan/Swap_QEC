"""
True streaming purification protocol with O(log N) memory scaling.
Implements stack-based processing where states are processed online.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple, Iterator
from src.quantum_states import QuantumState, PurityParameterState, BlochVectorState, generate_random_pure_state
from src.noise_models import NoiseModel, DepolarizingNoise, PauliNoise
from src.swap_operations import SwapTestProcessor, SwapResult


@dataclass
class StreamingState:
    """Wrapper for states in the streaming stack with metadata."""
    state: QuantumState
    level: int
    arrival_time: int
    processing_history: List[int] = field(default_factory=list)


@dataclass 
class StreamingResult:
    """Result from streaming purification protocol."""
    total_states_processed: int
    output_states: List[StreamingState]
    max_stack_depth_used: int
    memory_efficiency: float
    total_swap_operations: int
    total_amplification_iterations: int
    error_evolution_trace: List[Tuple[int, float, int]]  # (time, error, level)


class TrueStreamingProtocol:
    """
    Streaming purification protocol with O(log N) memory scaling.
    
    Uses stack-based processing where states are processed online as they arrive,
    never storing more than O(log N) states simultaneously.
    """
    
    def __init__(self, max_stack_levels: int = 30, max_amplification_iterations: int = 100):
        """
        Initialize streaming protocol.
        
        Args:
            max_stack_levels: Maximum depth of purification stack (sets memory bound)
            max_amplification_iterations: Maximum iterations for amplitude amplification
        """
        self.max_stack_levels = max_stack_levels
        self.stack = [None] * max_stack_levels  # Stack[i] = state at purification level i
        self.swap_processor = SwapTestProcessor(max_amplification_iterations)
        
        # Tracking variables
        self.states_processed = 0
        self.swap_operations = 0
        self.amplification_iterations = 0
        self.output_states = []
        self.max_depth_used = 0
        self.error_trace = []
    
    def process_state_stream(self, 
                           noise_model: NoiseModel,
                           num_states: int,
                           initial_error_rate: float,
                           target_state: np.ndarray = None) -> StreamingResult:
        """
        Process a stream of N identical noisy states with O(log N) memory.
        
        Args:
            noise_model: Noise model to apply to generate states
            num_states: Total number of states to process
            initial_error_rate: Physical error rate for state generation
            target_state: Target pure state (random if None)
        """
        # Initialize
        self._reset_protocol()
        
        if target_state is None:
            dimension = getattr(noise_model, 'dimension', 2)
            target_state = generate_random_pure_state(dimension)
        
        # Process states one by one (simulating online arrival)
        for arrival_time in range(num_states):
            # Generate new noisy state (simulates online arrival)
            noisy_state = noise_model.apply_noise(target_state)
            streaming_state = StreamingState(
                state=noisy_state,
                level=0,
                arrival_time=arrival_time
            )
            
            # Process through stack
            self._process_single_state(streaming_state)
            
            # Track error evolution
            if arrival_time % max(1, num_states // 20) == 0:  # Sample periodically
                self._record_error_snapshot(arrival_time)
        
        # Flush remaining states from stack
        self._flush_stack()
        
        return self._generate_result()
    
    def _process_single_state(self, new_state: StreamingState):
        """
        Process a single state through the purification stack.
        
        This is the core streaming algorithm implementing O(log N) memory.
        """
        current_state = new_state
        level = 0
        
        while level < self.max_stack_levels:
            if self.stack[level] is None:
                # Empty slot found - store state here
                self.stack[level] = current_state
                self.max_depth_used = max(self.max_depth_used, level)
                break
            else:
                # Level occupied - pair states and swap
                partner_state = self.stack[level]
                self.stack[level] = None  # Clear this level
                
                # Perform amplitude-amplified swap test
                swap_result = self.swap_processor.amplitude_amplified_swap(
                    current_state.state, partner_state.state)
                
                # Update tracking
                self.swap_operations += 1
                self.amplification_iterations += swap_result.amplification_iterations
                
                # Create new state for next level
                current_state = StreamingState(
                    state=swap_result.output_state,
                    level=level + 1,
                    arrival_time=max(current_state.arrival_time, partner_state.arrival_time),
                    processing_history=current_state.processing_history + partner_state.processing_history + [level]
                )
                
                level += 1
        
        # If we've reached max stack depth, output the state
        if level >= self.max_stack_levels:
            self.output_states.append(current_state)
    
    def _flush_stack(self):
        """Flush remaining states from stack at end of processing."""
        for level in range(self.max_stack_levels):
            if self.stack[level] is not None:
                self.output_states.append(self.stack[level])
                self.stack[level] = None
    
    def _record_error_snapshot(self, time: int):
        """Record current state of purification for analysis."""
        for level, state in enumerate(self.stack):
            if state is not None:
                error = state.state.get_logical_error()
                self.error_trace.append((time, error, level))
    
    def _reset_protocol(self):
        """Reset protocol state for new run."""
        self.stack = [None] * self.max_stack_levels
        self.states_processed = 0
        self.swap_operations = 0
        self.amplification_iterations = 0
        self.output_states = []
        self.max_depth_used = 0
        self.error_trace = []
    
    def _generate_result(self) -> StreamingResult:
        """Generate final result summary."""
        return StreamingResult(
            total_states_processed=self.states_processed,
            output_states=self.output_states,
            max_stack_depth_used=self.max_depth_used,
            memory_efficiency=self.max_depth_used / max(1, np.log2(self.states_processed)),
            total_swap_operations=self.swap_operations,
            total_amplification_iterations=self.amplification_iterations,
            error_evolution_trace=self.error_trace
        )
    
    def get_memory_usage(self) -> int:
        """Get current memory usage (number of states stored)."""
        return sum(1 for state in self.stack if state is not None)
    
    def get_theoretical_memory_bound(self, num_states: int) -> int:
        """Get theoretical O(log N) memory bound."""
        return min(self.max_stack_levels, int(np.ceil(np.log2(num_states))) + 1)
    
    def analyze_memory_scaling(self, state_counts: List[int], 
                             noise_model: NoiseModel, 
                             initial_error_rate: float = 0.3) -> dict:
        """
        Analyze memory scaling across different numbers of input states.
        
        Returns:
            Dictionary with scaling analysis results
        """
        results = {
            'N_values': state_counts,
            'max_memory_used': [],
            'theoretical_bounds': [],
            'memory_efficiency': [],
            'total_swaps': [],
            'output_states_count': []
        }
        
        for N in state_counts:
            result = self.process_state_stream(
                noise_model=noise_model,
                num_states=N,
                initial_error_rate=initial_error_rate
            )
            
            theoretical_bound = self.get_theoretical_memory_bound(N)
            
            results['max_memory_used'].append(result.max_stack_depth_used)
            results['theoretical_bounds'].append(theoretical_bound)
            results['memory_efficiency'].append(result.memory_efficiency)
            results['total_swaps'].append(result.total_swap_operations)
            results['output_states_count'].append(len(result.output_states))
        
        return results


# Convenience functions for integration with existing code
def create_streaming_protocol(max_stack_levels: int = 30) -> TrueStreamingProtocol:
    """Create streaming protocol instance."""
    return TrueStreamingProtocol(max_stack_levels=max_stack_levels)


def run_streaming_comparison(batch_protocol, streaming_protocol, 
                           noise_model: NoiseModel,
                           num_states: int = 64,
                           initial_error_rate: float = 0.3):
    """
    Compare batch vs streaming implementations.
    
    Returns comparison of memory usage, performance, and results.
    """
    # Run batch protocol
    batch_result = batch_protocol.purify_stream(
        initial_error_rate=initial_error_rate,
        noise_model=noise_model,
        num_input_states=num_states
    )
    
    # Run streaming protocol  
    streaming_result = streaming_protocol.process_state_stream(
        noise_model=noise_model,
        num_states=num_states,
        initial_error_rate=initial_error_rate
    )
    
    return {
        'batch_memory': num_states,  # Stores all states simultaneously
        'streaming_memory': streaming_result.max_stack_depth_used,
        'theoretical_streaming_bound': streaming_protocol.get_theoretical_memory_bound(num_states),
        'memory_improvement_factor': num_states / streaming_result.max_stack_depth_used,
        'batch_final_error': batch_result.logical_error_evolution[-1],
        'streaming_output_count': len(streaming_result.output_states),
        'streaming_best_error': min(s.state.get_logical_error() for s in streaming_result.output_states) if streaming_result.output_states else float('inf')
    }