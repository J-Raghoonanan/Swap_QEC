"""
Batch SWAP purification implementation for IBM Quantum hardware.

This implementation uses a practical batch approach:
1. Create all N noisy Hadamard states simultaneously
2. Apply noise to all N copies
3. Perform tree of pairwise SWAP tests to get 1 final state
4. Measure final state fidelity with post-selection on successful SWAP tests
5. Retry whole experiment if too many SWAP tests fail

Author: Based on "From Noisy to Nice: Sequential SWAP Purification as Purification Error Correction (PEC)"
"""
from __future__ import annotations

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import copy

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# IBM imports
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2
    IBM_RUNTIME_AVAILABLE = True
except ImportError:
    IBM_RUNTIME_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PurificationConfig:
    """Configuration for batch purification experiment."""
    M: int  # Number of qubits in target Hadamard state
    N: int  # Total number of input copies (must be power of 2)
    p: float  # Error probability for depolarizing noise
    backend_name: str = "ibm_torino"
    shots: int = 8192
    transpilation_level: int = 2
    max_retry_attempts: int = 5  # Retry if too many SWAP failures
    min_success_rate: float = 0.1  # Minimum SWAP success rate to accept
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.M <= 0 or self.M > 8:  # Practical limit for NISQ devices
            raise ValueError(f"M must be in [1, 8], got {self.M}")
        if self.N <= 1 or (self.N & (self.N - 1)) != 0:  # Must be power of 2
            raise ValueError(f"N must be a power of 2 and > 1, got {self.N}")
        if not (0.0 <= self.p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {self.p}")
        if self.shots <= 0:
            raise ValueError(f"shots must be positive, got {self.shots}")
        
        # Estimate maximum qubits needed for this configuration
        # N copies of M qubits each, plus log2(N) ancillas for SWAP tests
        max_qubits = self.N * self.M + int(np.log2(self.N))
        if max_qubits > 127:  # Current IBM limits
            raise ValueError(f"Configuration requires ~{max_qubits} qubits, exceeds backend limit")
    
    def synthesize_run_id(self) -> str:
        """Create a unique identifier for this run."""
        return f"batch_M{self.M}_N{self.N}_p{self.p:.4f}"


# =============================================================================
# State Preparation
# =============================================================================

def create_hadamard_target_circuit(M: int) -> QuantumCircuit:
    """
    Create preparation circuit for |+⟩^⊗M Hadamard target state.
    
    Args:
        M: Number of qubits
        
    Returns:
        Circuit that prepares |+⟩^⊗M from |0⟩^⊗M
    """
    if M <= 0:
        raise ValueError("M must be positive")
    
    qc = QuantumCircuit(M, name=f"hadamard_target_M{M}")
    for q in range(M):
        qc.h(q)
    return qc


def apply_depolarizing_noise(qc: QuantumCircuit, qubits: List[int], p: float, 
                            seed: Optional[int] = None) -> None:
    """
    Apply depolarizing noise using random Pauli operations.
    
    For depolarizing parameter p:
    - With probability (1-p): no error
    - With probability p/3 each: apply X, Y, or Z error
    
    Args:
        qc: Circuit to modify in-place
        qubits: Qubits to apply noise to
        p: Depolarizing error probability
        seed: Random seed for reproducibility
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")
    
    if p == 0.0:
        return  # No noise to apply
    
    rng = np.random.default_rng(seed)
    
    for qubit in qubits:
        rand_val = rng.random()
        if rand_val < p / 3.0:  # X error
            qc.x(qubit)
        elif rand_val < 2 * p / 3.0:  # Y error  
            qc.y(qubit)
        elif rand_val < p:  # Z error
            qc.z(qubit)
        # else: no error (probability 1-p)


# =============================================================================
# Batch Circuit Construction
# =============================================================================

def create_batch_purification_circuit(config: PurificationConfig) -> QuantumCircuit:
    """
    Create complete batch purification circuit.
    
    Circuit structure:
    1. N registers of M qubits each for noisy copies
    2. log2(N) ancilla qubits for SWAP tests
    3. Prepare all N noisy Hadamard states
    4. Tree of pairwise SWAP tests
    5. Final state in register 0
    
    Args:
        config: Purification configuration
        
    Returns:
        Complete circuit for batch purification
    """
    M, N = config.M, config.N
    num_levels = int(np.log2(N))  # Number of purification levels
    
    # Total qubits: N registers of M qubits + ancillas for SWAP tests
    total_data_qubits = N * M
    total_ancillas = num_levels
    total_qubits = total_data_qubits + total_ancillas
    
    # Create circuit
    qc = QuantumCircuit(total_qubits, name=f"batch_purification_M{M}_N{N}")
    
    # Define qubit assignments
    # Data registers: [0:M], [M:2M], [2M:3M], ..., [(N-1)M:NM]
    data_registers = [list(range(i*M, (i+1)*M)) for i in range(N)]
    # Ancillas: [NM:NM+num_levels]  
    ancillas = list(range(total_data_qubits, total_qubits))
    
    logger.info(f"Circuit layout: {N} data registers of {M} qubits, {total_ancillas} ancillas")
    logger.info(f"Data registers: {data_registers}")
    logger.info(f"Ancillas: {ancillas}")
    
    # Step 1: Prepare all N noisy Hadamard states
    qc.barrier()
    qc.barrier(label="Prepare N noisy Hadamard states")
    
    for i in range(N):
        # Prepare Hadamard state on register i
        for q in data_registers[i]:
            qc.h(q)
        
        # Apply depolarizing noise to register i
        apply_depolarizing_noise(qc, data_registers[i], config.p, seed=42 + i)
    
    # Step 2: Tree of pairwise SWAP purifications
    qc.barrier()
    qc.barrier(label="Tree of SWAP purifications")
    
    # Track which registers are "active" at each level
    active_registers = list(range(N))
    
    for level in range(num_levels):
        ancilla_idx = ancillas[level]
        num_pairs = len(active_registers) // 2
        
        logger.debug(f"Level {level}: {len(active_registers)} active registers, {num_pairs} SWAP tests")
        
        for pair_idx in range(num_pairs):
            # Get register indices for this pair
            reg_a_idx = active_registers[2 * pair_idx]
            reg_b_idx = active_registers[2 * pair_idx + 1]
            
            reg_a_qubits = data_registers[reg_a_idx]
            reg_b_qubits = data_registers[reg_b_idx]
            
            # SWAP test between register A and register B
            qc.h(ancilla_idx)
            for i in range(M):
                qc.cswap(ancilla_idx, reg_a_qubits[i], reg_b_qubits[i])
            qc.h(ancilla_idx)
            
            logger.debug(f"  SWAP test {pair_idx}: reg{reg_a_idx} ⊕ reg{reg_b_idx} → ancilla{ancilla_idx}")
        
        # Update active registers (only register A from each pair survives)
        active_registers = [active_registers[2*i] for i in range(num_pairs)]
        
        qc.barrier()
    
    # Final state should be in register 0
    assert len(active_registers) == 1 and active_registers[0] == 0
    
    logger.info(f"Batch purification circuit created: {qc.depth()} depth, {qc.num_qubits} qubits")
    return qc


def add_measurements(qc: QuantumCircuit, config: PurificationConfig) -> QuantumCircuit:
    """
    Add measurements for post-selection and fidelity calculation.
    Compatible with SamplerV2 result structure.
    
    Args:
        qc: Purification circuit
        config: Configuration
        
    Returns:
        Circuit with measurements added
    """
    M, N = config.M, config.N
    num_levels = int(np.log2(N))
    
    total_data_qubits = N * M
    ancilla_start = total_data_qubits
    
    # Use single classical register for all measurements (SamplerV2 compatible)
    total_measurements = M + num_levels
    meas_register = ClassicalRegister(total_measurements, 'meas')
    
    # Create new circuit with single classical register
    measured_qc = QuantumCircuit(qc.num_qubits)
    measured_qc.add_register(meas_register)
    measured_qc.compose(qc, inplace=True)
    
    # Measure final state (register 0) first - these go in first M bits
    for i in range(M):
        measured_qc.measure(i, meas_register[i])
    
    # Measure ancillas for post-selection - these go in last num_levels bits
    for i in range(num_levels):
        measured_qc.measure(ancilla_start + i, meas_register[M + i])
    
    return measured_qc


# =============================================================================
# Execution and Analysis
# =============================================================================

def analyze_results(counts: Dict[str, int], config: PurificationConfig) -> Tuple[float, float]:
    """
    Analyze measurement results with post-selection.
    Updated for single classical register structure.
    
    Args:
        counts: Raw measurement counts from circuit execution
        config: Configuration parameters
        
    Returns:
        (fidelity, success_probability) tuple where:
        - fidelity: Measured fidelity conditioned on all SWAP tests succeeding
        - success_probability: Probability that all SWAP tests succeeded
    """
    M, N = config.M, config.N
    num_levels = int(np.log2(N))
    
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0, 0.0
    
    successful_shots = 0
    perfect_final_state_count = 0
    
    for outcome_str, count in counts.items():
        # Parse outcome: first M bits are final state, last num_levels bits are ancillas
        expected_length = M + num_levels
        if len(outcome_str) != expected_length:
            logger.warning(f"Unexpected outcome format: '{outcome_str}' (expected {expected_length} bits)")
            continue
        
        final_state_bits = outcome_str[:M]           # First M bits
        ancilla_bits = outcome_str[M:]              # Last num_levels bits
        
        # Check if all SWAP tests succeeded (all ancillas = 0)
        if ancilla_bits == "0" * num_levels:
            successful_shots += count
            
            # Check if final state is perfect |+⟩^⊗M (should measure as |0⟩^⊗M)
            if final_state_bits == "0" * M:
                perfect_final_state_count += count
    
    # Calculate probabilities
    success_probability = successful_shots / total_shots
    
    if successful_shots > 0:
        fidelity = perfect_final_state_count / successful_shots
    else:
        fidelity = 0.0
    
    logger.debug(f"Analysis: {successful_shots}/{total_shots} successful shots "
                f"({100*success_probability:.1f}%), "
                f"{perfect_final_state_count}/{successful_shots} perfect final states "
                f"({100*fidelity:.1f}% fidelity)")
    
    return fidelity, success_probability


def execute_with_retry(circuit: QuantumCircuit, config: PurificationConfig,
                      service, backend) -> Tuple[float, float, Dict]:
    """
    Execute circuit with retry on low success rate.
    
    Args:
        circuit: Circuit to execute  
        config: Configuration
        service: IBM service
        backend: IBM backend
        
    Returns:
        (fidelity, success_probability, final_counts) tuple
    """
    logger.info("Executing batch purification circuit")
    
    # Transpile circuit
    transpiled = transpile_circuit_for_backend(circuit, backend, config.transpilation_level)
    logger.info(f"Transpiled circuit: depth={transpiled.depth()}, qubits={transpiled.num_qubits}")
    
    total_attempts = 0
    all_counts = {}
    
    # Use SamplerV2 with correct syntax (per IBM docs)
    sampler = SamplerV2(backend)
    
    while total_attempts < config.max_retry_attempts:
        total_attempts += 1
        logger.info(f"Execution attempt {total_attempts}/{config.max_retry_attempts}")
        
        # Execute circuit
        job = sampler.run([transpiled], shots=config.shots)
        result = job.result()
        
        # SamplerV2 has different result structure - try multiple access patterns
        pub_result = result[0]
        
        # Try to get measurement counts from the 'meas' register
        if hasattr(pub_result.data, 'meas'):
            counts = pub_result.data.meas.get_counts()
            logger.debug(f"Found measurement counts via 'meas' register")
        else:
            # Debug: print available attributes
            data_attrs = [attr for attr in dir(pub_result.data) if not attr.startswith('_')]
            logger.info(f"Available data attributes: {data_attrs}")
            
            # Try each attribute to find one with get_counts method
            counts = None
            for attr_name in data_attrs:
                try:
                    attr = getattr(pub_result.data, attr_name)
                    if hasattr(attr, 'get_counts'):
                        counts = attr.get_counts()
                        logger.info(f"Found counts via '{attr_name}.get_counts()'")
                        break
                except Exception as e:
                    logger.debug(f"Failed to access {attr_name}: {e}")
            
            if counts is None:
                raise RuntimeError(f"Cannot find measurement counts in SamplerV2 result. "
                                 f"Available data attributes: {data_attrs}")
        
        logger.debug(f"Retrieved {len(counts)} unique measurement outcomes, "
                    f"total shots: {sum(counts.values())}")
        
        # Accumulate counts across attempts
        for outcome, count in counts.items():
            if outcome not in all_counts:
                all_counts[outcome] = 0
            all_counts[outcome] += count
        
        # Analyze current results
        fidelity, success_prob = analyze_results(all_counts, config)
        total_shots_so_far = sum(all_counts.values())
        successful_shots = total_shots_so_far * success_prob
        
        logger.info(f"  Attempt {total_attempts}: success_rate={success_prob:.4f}, "
                   f"successful_shots={successful_shots:.0f}")
        
        # Check if we have acceptable success rate or hit retry limit
        if success_prob >= config.min_success_rate or total_attempts >= config.max_retry_attempts:
            break
    
    final_fidelity, final_success_prob = analyze_results(all_counts, config)
    
    logger.info(f"Final results after {total_attempts} attempts:")
    logger.info(f"  Fidelity: {final_fidelity:.4f}")
    logger.info(f"  Success probability: {final_success_prob:.4f}")
    logger.info(f"  Total shots: {sum(all_counts.values())}")
    
    return final_fidelity, final_success_prob, all_counts


# =============================================================================
# Backend Setup  
# =============================================================================

def setup_ibm_backend(backend_name: str):
    """Setup IBM quantum backend."""
    if not IBM_RUNTIME_AVAILABLE:
        raise ImportError("IBM Runtime not available. Install qiskit-ibm-runtime.")
    
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    
    logger.info(f"Connected to backend: {backend_name}")
    logger.info(f"Backend status: {backend.status()}")
    
    return service, backend


def transpile_circuit_for_backend(circuit: QuantumCircuit, backend, optimization_level: int = 2):
    """Transpile circuit allowing full backend topology for connectivity."""
    
    num_circuit_qubits = circuit.num_qubits
    backend_qubits = backend.configuration().n_qubits
    
    logger.info(f"Circuit requires {num_circuit_qubits} qubits, backend has {backend_qubits}")
    
    if num_circuit_qubits > backend_qubits:
        raise ValueError(f"Circuit needs {num_circuit_qubits} qubits but backend only has {backend_qubits}")
    
    # Let transpiler use full backend topology for optimal routing
    # This allows it to route around connectivity constraints
    pass_manager = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend
        # No layout constraints - let transpiler find best mapping!
    )
    
    transpiled = pass_manager.run(circuit)
    
    logger.info(f"Transpiled: {num_circuit_qubits} logical → {transpiled.num_qubits} physical qubits")
    logger.info(f"Circuit depth: {circuit.depth()} → {transpiled.depth()}")
    
    # This is fine - using extra qubits for routing is normal and expected
    return transpiled


# =============================================================================
# Main Experiment Function
# =============================================================================

def run_complete_purification_experiment(config: PurificationConfig, 
                                       service=None, backend=None) -> Dict:
    """
    Run the complete batch purification experiment.
    
    Args:
        config: Experiment configuration
        service: IBM quantum service (optional)
        backend: IBM quantum backend (optional) 
        
    Returns:
        Dictionary with experimental results
    """
    config.validate()
    
    logger.info(f"Starting batch purification experiment: {config.synthesize_run_id()}")
    logger.info(f"Configuration: M={config.M}, N={config.N}, p={config.p}")
    
    # Step 1: Create batch purification circuit
    purification_circuit = create_batch_purification_circuit(config)
    
    # Step 2: Add measurements
    measured_circuit = add_measurements(purification_circuit, config)
    
    # Step 3: Setup backend if needed
    if service is None or backend is None:
        service, backend = setup_ibm_backend(config.backend_name)
    
    # Step 4: Execute with retry
    fidelity, success_prob, final_counts = execute_with_retry(
        measured_circuit, config, service, backend)
    
    # Step 5: Package results
    results = {
        'run_id': config.synthesize_run_id(),
        'M': config.M,
        'N': config.N, 
        'p': config.p,
        'max_purification_level': int(np.log2(config.N)),
        'final_fidelity': fidelity,
        'swap_success_probability': success_prob,
        'total_shots': sum(final_counts.values()),
        'backend_name': config.backend_name,
        'circuit_depth': measured_circuit.depth(),
        'circuit_qubits': measured_circuit.num_qubits,
    }
    
    logger.info(f"Experiment complete: fidelity={fidelity:.4f}, success_prob={success_prob:.4f}")
    return results


# =============================================================================
# Export
# =============================================================================

__all__ = [
    # Configuration
    "PurificationConfig",
    
    # Core components
    "create_hadamard_target_circuit",
    "apply_depolarizing_noise",
    "create_batch_purification_circuit",
    "add_measurements",
    
    # Analysis and execution
    "analyze_results",
    "execute_with_retry",
    
    # Backend integration
    "setup_ibm_backend",
    "transpile_circuit_for_backend",
    
    # Complete experiment
    "run_complete_purification_experiment",
]