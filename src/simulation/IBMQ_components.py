"""
IBMQ implementation components for SWAP-based purification error correction.

This module provides hardware-executable functions that closely mirror the 
simulation logic from the main codebase, adapted for IBM Quantum devices.
"""
from __future__ import annotations

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import json
import csv
from datetime import datetime

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers import Backend, Job
from qiskit.result import Result
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit.library import IGate, XGate, YGate, ZGate, HGate, SGate, SdgGate
from qiskit.quantum_info import random_unitary

# IBM imports - modern runtime approach only
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2
    from qiskit_aer import AerSimulator
    from qiskit.compiler import transpile
    QISKIT_AVAILABLE = True
    IBM_RUNTIME_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Qiskit imports failed: {e}")
    QISKIT_AVAILABLE = False
    IBM_RUNTIME_AVAILABLE = False
    # Create dummy classes to avoid import errors
    QiskitRuntimeService = None
    Session = None
    SamplerV2 = None
    AerSimulator = None

# Local imports (reuse configs from simulation)
from .configs import NoiseSpec, NoiseType, NoiseMode, TwirlingSpec, TargetSpec, StateKind

logger = logging.getLogger(__name__)


@dataclass
class IBMQRunSpec:
    """Configuration for IBM Quantum hardware runs."""
    M: int  # Number of qubits per copy
    N: int  # Number of input copies for sequential purification
    noise: NoiseSpec
    target_type: str = "hadamard"  # or "ghz"
    backend_name: str = "ibm_torino"
    shots: int = 8192
    use_amplitude_amplification: bool = False
    transpilation_level: int = 2
    out_dir: Path = field(default_factory=lambda: Path("data/IBMQ"))
    run_id: Optional[str] = None
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.M <= 0 or self.M > 10:  # Reasonable limit for IBMQ
            raise ValueError(f"M must be in [1, 10], got {self.M}")
        if self.N <= 0 or (self.N & (self.N - 1)) != 0:  # Must be power of 2
            raise ValueError(f"N must be a positive power of 2, got {self.N}")
        if self.N > 256:  # Reasonable upper limit
            raise ValueError(f"N must be <= 256, got {self.N}")
        if not (0.0 <= self.noise.p <= 1.0):
            raise ValueError(f"noise.p must be in [0,1], got {self.noise.p}")
        if self.target_type not in ["hadamard", "ghz"]:
            raise ValueError(f"target_type must be 'hadamard' or 'ghz', got {self.target_type}")
        if self.shots <= 0:
            raise ValueError(f"shots must be positive, got {self.shots}")
        
        # Check total qubit requirements
        total_qubits = self.N * self.M + self._max_ancilla_qubits()
        if total_qubits > 100:  # Conservative limit for current hardware
            raise ValueError(f"Total qubits ({total_qubits}) exceeds hardware limits")
    
    def _max_ancilla_qubits(self) -> int:
        """Calculate maximum ancilla qubits needed across all rounds."""
        rounds = int(np.log2(self.N))
        max_parallel_swaps = self.N // 2  # First round has most parallel SWAPs
        return max_parallel_swaps
    
    def num_rounds(self) -> int:
        """Number of purification rounds = log₂ N."""
        return int(np.log2(self.N))


# =============================
# Phase 1: Core Functions
# =============================

def prepare_hadamard_state(M: int) -> QuantumCircuit:
    """
    Prepare |+⟩^⊗M state on M qubits.
    
    Args:
        M: Number of qubits
        
    Returns:
        QuantumCircuit preparing the Hadamard product state
    """
    if M <= 0:
        raise ValueError("M must be positive")
    
    qc = QuantumCircuit(M, name=f"prep_hadamard_M{M}")
    for q in range(M):
        qc.h(q)
    
    logger.debug(f"Created Hadamard preparation circuit for M={M}")
    return qc


def prepare_ghz_state(M: int) -> QuantumCircuit:
    """
    Prepare GHZ state (|0...0⟩ + |1...1⟩)/√2 on M qubits.
    
    Args:
        M: Number of qubits
        
    Returns:
        QuantumCircuit preparing the GHZ state
    """
    if M <= 0:
        raise ValueError("M must be positive")
    
    qc = QuantumCircuit(M, name=f"prep_ghz_M{M}")
    qc.h(0)  # Create superposition on first qubit
    for q in range(1, M):
        qc.cx(0, q)  # Entangle with all other qubits
    
    logger.debug(f"Created GHZ preparation circuit for M={M}")
    return qc


def _sample_clifford_gate_ibmq(mode: str, index: int, seed: Optional[int] = None) -> str:
    """
    Sample a single-qubit Clifford gate for twirling.
    
    Args:
        mode: 'random' or 'cyclic'
        index: Index for deterministic sampling in cyclic mode
        seed: Random seed
        
    Returns:
        Gate name string
    """
    # Use same options as simulation
    options = ['i', 'h', 's', 'sdg', 'sh', 'sdgh']
    
    if mode == "random":
        rng = np.random.default_rng(seed)
        return str(rng.choice(options))
    else:  # cyclic
        return options[index % len(options)]


def _apply_clifford_gate_ibmq(qc: QuantumCircuit, qubit: int, gate_name: str) -> None:
    """Apply a single-qubit Clifford gate to the circuit."""
    if gate_name == 'i':
        pass  # identity, do nothing
    elif gate_name == 'h':
        qc.h(qubit)
    elif gate_name == 's':
        qc.s(qubit)
    elif gate_name == 'sdg':
        qc.sdg(qubit)
    elif gate_name == 'sh':
        qc.s(qubit)
        qc.h(qubit)
    elif gate_name == 'sdgh':
        qc.sdg(qubit)
        qc.h(qubit)
    else:
        raise ValueError(f"Unknown Clifford gate: {gate_name}")


def _apply_inverse_clifford_gate_ibmq(qc: QuantumCircuit, qubit: int, gate_name: str) -> None:
    """Apply the inverse of a Clifford gate."""
    if gate_name == 'i':
        pass
    elif gate_name == 'h':
        qc.h(qubit)  # H† = H
    elif gate_name == 's':
        qc.sdg(qubit)  # S† = Sdg
    elif gate_name == 'sdg':
        qc.s(qubit)  # Sdg† = S
    elif gate_name == 'sh':
        qc.h(qubit)  # Reverse order
        qc.sdg(qubit)
    elif gate_name == 'sdgh':
        qc.h(qubit)
        qc.s(qubit)
    else:
        raise ValueError(f"Unknown Clifford gate: {gate_name}")


def apply_depolarizing_noise(qc: QuantumCircuit, qubits: List[int], p: float, seed: Optional[int] = None) -> None:
    """
    Apply depolarizing noise to specified qubits.
    
    For hardware implementation, we approximate the depolarizing channel
    using probabilistic Pauli gates. Each qubit gets X, Y, or Z with 
    probability p/3 each.
    
    Args:
        qc: Circuit to modify in-place
        qubits: List of qubit indices to apply noise to
        p: Depolarizing probability parameter
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0,1], got {p}")
    
    # For hardware, we approximate by probabilistically applying Paulis
    # This matches the Kraus operator structure from simulation
    rng = np.random.default_rng(seed)
    for qubit in qubits:
        # Create a probabilistic mixture approximating the depolarizing channel
        # In practice, we'll apply each Pauli with probability p/3
        r = rng.random()
        
        if r < p / 3:
            qc.x(qubit)  # Pauli X
        elif r < 2 * p / 3:
            qc.y(qubit)  # Pauli Y  
        elif r < p:
            qc.z(qubit)  # Pauli Z
        # else: identity (do nothing)
    
    logger.debug(f"Applied depolarizing noise with p={p:.4f} to qubits {qubits}")


def apply_twirled_dephasing_noise(qc: QuantumCircuit, qubits: List[int], p: float, 
                                 twirling_seed: Optional[int] = None) -> None:
    """
    Apply twirled dephasing noise to specified qubits.
    
    Implements channel twirling exactly as in simulation:
    1. Apply random Clifford C
    2. Apply dephasing channel (Z with probability p)
    3. Apply C†
    
    Args:
        qc: Circuit to modify in-place
        qubits: List of qubit indices to apply noise to  
        p: Dephasing probability parameter
        twirling_seed: Seed for Clifford sampling
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0,1], got {p}")
    
    clifford_gates = []
    
    # Step 1: Apply random Cliffords
    for i, qubit in enumerate(qubits):
        qubit_seed = (twirling_seed + i) if twirling_seed is not None else None
        gate_name = _sample_clifford_gate_ibmq("random", index=i, seed=qubit_seed)
        _apply_clifford_gate_ibmq(qc, qubit, gate_name)
        clifford_gates.append(gate_name)
    
    # Step 2: Apply dephasing noise (Z-rotations) 
    for qubit in qubits:
        if np.random.random() < p:
            qc.z(qubit)
    
    # Step 3: Undo Clifford frame
    for i, qubit in enumerate(qubits):
        _apply_inverse_clifford_gate_ibmq(qc, qubit, clifford_gates[i])
    
    logger.debug(f"Applied twirled dephasing noise with p={p:.4f} to qubits {qubits}")


def build_swap_test_circuit(M: int) -> QuantumCircuit:
    """
    Build SWAP test circuit for M-qubit registers.
    
    Circuit structure:
    - 1 ancilla qubit
    - M qubits for register A  
    - M qubits for register B
    - Total: 1 + 2*M qubits
    
    Args:
        M: Number of qubits per register
        
    Returns:
        QuantumCircuit implementing H-CSWAP-H pattern
    """
    if M <= 0:
        raise ValueError("M must be positive")
    
    total_qubits = 1 + 2 * M
    qc = QuantumCircuit(total_qubits, name=f"swap_test_M{M}")
    
    ancilla = 0
    reg_A = list(range(1, M + 1))
    reg_B = list(range(M + 1, 2 * M + 1))
    
    # First Hadamard on ancilla
    qc.h(ancilla)
    
    # Controlled SWAP between registers A and B
    for i in range(M):
        qc.cswap(ancilla, reg_A[i], reg_B[i])
    
    # Second Hadamard on ancilla  
    qc.h(ancilla)
    
    logger.debug(f"Built SWAP test circuit for M={M} (total {total_qubits} qubits)")
    return qc


def add_fidelity_measurement_hadamard(qc: QuantumCircuit, qubits: List[int], 
                                     classical_reg: Optional[ClassicalRegister] = None) -> None:
    """
    Add fidelity measurement for Hadamard state |+⟩^⊗M.
    
    Measurement strategy:
    1. Apply H gate to each qubit (rotate to Z-basis)
    2. Measure all qubits
    3. Fidelity = probability of measuring all |0⟩
    
    Args:
        qc: Circuit to modify in-place
        qubits: Qubits to measure for fidelity
        classical_reg: Classical register for measurements (auto-created if None)
    """
    if not qubits:
        raise ValueError("qubits list cannot be empty")
    
    # Rotate from X-basis to Z-basis
    for qubit in qubits:
        qc.h(qubit)
    
    # Add classical register if not provided
    if classical_reg is None:
        classical_reg = ClassicalRegister(len(qubits), f"fidelity_c")
        qc.add_register(classical_reg)
    
    # Measure all qubits
    for i, qubit in enumerate(qubits):
        qc.measure(qubit, classical_reg[i])
    
    logger.debug(f"Added Hadamard fidelity measurement for qubits {qubits}")


def add_fidelity_measurement_ghz(qc: QuantumCircuit, qubits: List[int],
                                classical_reg: Optional[ClassicalRegister] = None) -> None:
    """
    Add fidelity measurement for GHZ state (|00...0⟩ + |11...1⟩)/√2.
    
    Measurement strategy:
    1. Measure directly in Z-basis
    2. Calculate fidelity from parity measurements
    
    Args:
        qc: Circuit to modify in-place  
        qubits: Qubits to measure for fidelity
        classical_reg: Classical register for measurements (auto-created if None)
    """
    if not qubits:
        raise ValueError("qubits list cannot be empty")
    
    # Add classical register if not provided
    if classical_reg is None:
        classical_reg = ClassicalRegister(len(qubits), f"fidelity_c")
        qc.add_register(classical_reg)
    
    # Measure all qubits directly (no rotation needed for GHZ)
    for i, qubit in enumerate(qubits):
        qc.measure(qubit, classical_reg[i])
    
    logger.debug(f"Added GHZ fidelity measurement for qubits {qubits}")


def add_ancilla_measurement(qc: QuantumCircuit, ancilla: int,
                           classical_reg: Optional[ClassicalRegister] = None) -> None:
    """
    Add measurement of ancilla qubit for SWAP test success.
    
    Args:
        qc: Circuit to modify in-place
        ancilla: Ancilla qubit index
        classical_reg: Classical register (auto-created if None)
    """
    if classical_reg is None:
        classical_reg = ClassicalRegister(1, "ancilla_c")
        qc.add_register(classical_reg)
    
    qc.measure(ancilla, classical_reg[0])
    logger.debug(f"Added ancilla measurement for qubit {ancilla}")


def calculate_fidelity_from_counts(counts: Dict[str, int], M: int, 
                                  target_type: str) -> float:
    """
    Calculate fidelity from measurement counts.
    
    Args:
        counts: Dictionary of measurement outcomes and their counts
        M: Number of target qubits
        target_type: 'hadamard' or 'ghz'
        
    Returns:
        Estimated fidelity
    """
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0
    
    if target_type == "hadamard":
        # For |+⟩^⊗M, fidelity = P(all zeros after H rotation)
        target_outcome = "0" * M
        target_count = counts.get(target_outcome, 0)
        fidelity = target_count / total_shots
        
    elif target_type == "ghz":
        # For GHZ, fidelity related to P(all zeros) + P(all ones)
        all_zeros = "0" * M
        all_ones = "1" * M
        coherent_count = counts.get(all_zeros, 0) + counts.get(all_ones, 0)
        # For perfect GHZ: P(00...0) = P(11...1) = 0.5, so total = 1.0
        # Fidelity is the fraction of coherent outcomes
        fidelity = coherent_count / total_shots
        
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
    
    logger.debug(f"Calculated fidelity = {fidelity:.4f} for {target_type} state")
    return fidelity


# =============================
# Phase 2: Backend Integration  
# =============================

def setup_ibm_backend(backend_name: str = "ibm_torino") -> Tuple[Optional[QiskitRuntimeService], Backend]:
    """
    Setup quantum backend for experiments.
    
    Args:
        backend_name: 'aer_simulator' or IBM hardware name (e.g., 'ibm_torino')
        
    Returns:
        (service, backend) tuple where service is None for local simulator
    """
    if not QISKIT_AVAILABLE:
        logger.warning("Qiskit not available, returning None backend")
        return None, None
        
    if backend_name.lower() in ['aer_simulator', 'simulator']:
        backend = AerSimulator()
        service = None
        logger.info("Using local AerSimulator")
        return service, backend
    else:
        # Real IBM hardware
        if not IBM_RUNTIME_AVAILABLE:
            raise RuntimeError("IBM Runtime not available. Install with: pip install qiskit-ibm-runtime")
        
        try:
            service = QiskitRuntimeService()  # Uses default authentication
            backend = service.backend(backend_name)
            logger.info(f"Connected to IBM hardware: {backend_name}")
            logger.info(f"Backend status: {backend.status()}")
            return service, backend
        except Exception as e:
            logger.error(f"Failed to connect to {backend_name}: {e}")
            logger.warning("Falling back to local simulator")
            return None, AerSimulator()


def transpile_for_backend(qc: QuantumCircuit, backend: Backend, 
                         optimization_level: int = 2) -> QuantumCircuit:
    """
    Transpile circuit for specific IBM backend.
    
    Args:
        qc: Circuit to transpile
        backend: Target backend
        optimization_level: Qiskit optimization level (0-3)
        
    Returns:
        Transpiled circuit
    """
    try:
        # Generate transpilation pass manager
        pass_manager = generate_preset_pass_manager(
            optimization_level=optimization_level,
            backend=backend
        )
        
        # Transpile the circuit
        transpiled_qc = pass_manager.run(qc)
        
        logger.info(f"Transpiled circuit: {qc.num_qubits} qubits, depth {qc.depth()} → depth {transpiled_qc.depth()}")
        return transpiled_qc
        
    except Exception as e:
        logger.error(f"Transpilation failed: {e}")
        raise


def execute_circuits_with_backend(circuits: List[QuantumCircuit], backend, service, 
                                shots: int = 1024) -> List[Dict[str, int]]:
    """
    Execute quantum circuits and return measurement counts.
    
    Args:
        circuits: List of quantum circuits to execute
        backend: Quantum backend
        service: IBM service (None for local simulator)
        shots: Number of measurement shots
        
    Returns:
        List of count dictionaries, one per circuit
    """
    if not QISKIT_AVAILABLE or backend is None:
        # Mock execution for testing
        logger.warning("Using mock circuit execution")
        mock_counts = []
        for qc in circuits:
            if qc.num_clbits == 1:  # SWAP test
                mock_counts.append({'0': int(0.7 * shots), '1': int(0.3 * shots)})
            else:  # Multi-qubit measurement
                # Mock mostly |0...0⟩ outcome
                all_zero = '0' * qc.num_clbits
                mock_counts.append({all_zero: int(0.8 * shots), '1' * qc.num_clbits: int(0.2 * shots)})
        return mock_counts
    
    if service is not None:
        # Use IBM Runtime for hardware with SamplerV2
        try:
            with Session(service=service, backend=backend) as session:
                sampler = SamplerV2(session=session)
                
                # Convert circuits to primitive unified blocks (PUBs)
                pubs = [(circuit, None, shots) for circuit in circuits]
                
                job = sampler.run(pubs)
                result = job.result()
                
                counts_list = []
                for pub_result in result:
                    # Extract counts from PUB result
                    counts_dict = {}
                    if hasattr(pub_result, 'data') and hasattr(pub_result.data, 'meas'):
                        # Convert measurement data to counts
                        meas_data = pub_result.data.meas
                        for outcome, count in zip(*np.unique(meas_data, return_counts=True)):
                            # Convert outcome to bitstring
                            if isinstance(outcome, (int, np.integer)):
                                bitstring = format(int(outcome), f'0{circuits[0].num_clbits}b')
                            else:
                                bitstring = str(outcome)
                            counts_dict[bitstring] = int(count)
                    
                    counts_list.append(counts_dict)
                
                return counts_list
                
        except Exception as e:
            logger.error(f"IBM Runtime execution failed: {e}")
            logger.warning("Falling back to local simulator")
            # Fall back to local simulator
            backend = AerSimulator()
    
    # Use local simulator
    try:
        transpiled_circuits = transpile(circuits, backend, optimization_level=1)
        job = backend.run(transpiled_circuits, shots=shots)
        result = job.result()
        
        return [result.get_counts(i) for i in range(len(circuits))]
    except Exception as e:
        logger.error(f"Circuit execution failed: {e}")
        raise


def run_circuit_with_retries(qc: QuantumCircuit, service, backend, 
                            shots: int, max_retries: int = 3) -> Result:
    """
    Run circuit on backend with retry logic using modern execution pattern.
    
    Args:
        qc: Circuit to execute (should be pre-transpiled for hardware)
        service: IBM Runtime service (None for local simulator)
        backend: Target backend
        shots: Number of measurement shots
        max_retries: Maximum retry attempts
        
    Returns:
        Result object or counts dictionary
        
    Raises:
        RuntimeError: If all retries fail
    """
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Submitting job to {backend.name} (attempt {attempt + 1}/{max_retries + 1})")
            
            # Use the modern execution function
            counts_list = execute_circuits_with_backend([qc], backend, service, shots)
            
            # Create a mock Result object for compatibility
            class MockResult:
                def __init__(self, counts):
                    self._counts = counts
                    
                def get_counts(self, experiment=None):
                    if experiment is None:
                        return self._counts[0]
                    return self._counts[experiment]
            
            result = MockResult(counts_list)
            logger.info(f"Job completed successfully")
            return result
            
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"Job failed (attempt {attempt + 1}): {e}. Retrying...")
            else:
                logger.error(f"All retry attempts failed: {e}")
                raise RuntimeError(f"Failed to execute circuit after {max_retries + 1} attempts: {e}")


def save_ibmq_results(results: Dict, filepath: Path, run_spec: IBMQRunSpec) -> None:
    """
    Save IBMQ results to CSV file in format compatible with simulation data.
    
    Args:
        results: Dictionary containing experimental results
        filepath: Output file path  
        run_spec: Run configuration
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data row matching simulation format
    data_row = {
        'M': run_spec.M,
        'N': run_spec.N,  # Number of input copies
        'noise_type': run_spec.noise.noise_type.value,
        'noise_mode': run_spec.noise.mode.value, 
        'p': run_spec.noise.p,
        'target_type': run_spec.target_type,
        'backend': run_spec.backend_name,
        'shots': run_spec.shots,
        'timestamp': datetime.now().isoformat(),
        'run_id': run_spec.run_id or f"ibmq_{run_spec.M}_{run_spec.noise.noise_type.value}_{run_spec.noise.p:.3f}",
        **results  # Include all experimental results
    }
    
    # Write to CSV
    fieldnames = list(data_row.keys())
    file_exists = filepath.exists()
    
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data_row)
    
    logger.info(f"Saved results to {filepath}")


# =============================
# Phase 3: High-level Assembly
# =============================

def build_sequential_purification_circuit(N: int, M: int, noise_spec: NoiseSpec, 
                                        target_type: str = "hadamard") -> QuantumCircuit:
    """
    Build complete sequential purification circuit for N input copies.
    
    This implements the batch version of the sequential protocol:
    1. Create N copies of the target state
    2. Apply noise to all copies
    3. Perform log₂ N rounds of pairwise SWAP purification
    4. Measure final state fidelity
    
    Args:
        N: Number of input copies (must be power of 2)
        M: Number of qubits per copy
        noise_spec: Noise configuration
        target_type: 'hadamard' or 'ghz'
        
    Returns:
        Complete sequential purification circuit
    """
    if N <= 0 or (N & (N - 1)) != 0:
        raise ValueError(f"N must be a positive power of 2, got {N}")
    if M <= 0:
        raise ValueError(f"M must be positive, got {M}")
    
    num_rounds = int(np.log2(N))
    logger.info(f"Building sequential circuit: N={N} copies, M={M} qubits, {num_rounds} rounds")
    
    # Qubit allocation
    copy_qubits = N * M  # N copies of M qubits each
    max_ancilla = N // 2  # Maximum ancilla qubits needed (first round)
    total_qubits = copy_qubits + max_ancilla
    
    qc = QuantumCircuit(total_qubits, name=f"sequential_purification_N{N}_M{M}")
    
    # Define qubit ranges
    # Copies: [0, N*M)
    # Ancillas: [N*M, N*M + max_ancilla)
    copy_start = 0
    ancilla_start = copy_qubits
    
    # Step 1: Prepare N copies of the target state
    logger.debug(f"Preparing {N} copies of {target_type} state")
    if target_type == "hadamard":
        prep_circuit = prepare_hadamard_state(M)
    elif target_type == "ghz":
        prep_circuit = prepare_ghz_state(M)
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
    
    for copy_idx in range(N):
        copy_qubits_range = list(range(copy_idx * M, (copy_idx + 1) * M))
        qc.compose(prep_circuit, qubits=copy_qubits_range, inplace=True)
    
    # Step 2: Apply noise to all copies
    logger.debug(f"Applying {noise_spec.noise_type.value} noise with p={noise_spec.p}")
    for copy_idx in range(N):
        copy_qubits_range = list(range(copy_idx * M, (copy_idx + 1) * M))
        
        if noise_spec.noise_type == NoiseType.depolarizing:
            apply_depolarizing_noise(qc, copy_qubits_range, noise_spec.p)
        elif noise_spec.noise_type in [NoiseType.dephase_z, NoiseType.dephase_x]:
            # Use copy index as seed offset for consistent but different twirling per copy
            twirl_seed = 42 + copy_idx
            apply_twirled_dephasing_noise(qc, copy_qubits_range, noise_spec.p, twirl_seed)
        else:
            raise ValueError(f"Unsupported noise type: {noise_spec.noise_type}")
    
    # Step 3: Sequential SWAP purification rounds
    active_copies = list(range(N))  # Track which copies are still active
    
    for round_idx in range(num_rounds):
        num_pairs = len(active_copies) // 2
        logger.debug(f"Round {round_idx + 1}/{num_rounds}: {num_pairs} SWAP tests")
        
        # Perform SWAP tests in parallel for this round
        new_active_copies = []
        
        for pair_idx in range(num_pairs):
            # Get the two copies to merge
            copy_a_idx = active_copies[2 * pair_idx]
            copy_b_idx = active_copies[2 * pair_idx + 1]
            
            # Define qubit ranges for this SWAP test
            reg_a = list(range(copy_a_idx * M, (copy_a_idx + 1) * M))
            reg_b = list(range(copy_b_idx * M, (copy_b_idx + 1) * M))
            ancilla = ancilla_start + pair_idx
            
            # Build SWAP test for this pair
            # H-CSWAP-H pattern
            qc.h(ancilla)
            
            # Controlled SWAP between reg_a and reg_b
            for qubit_idx in range(M):
                # Fredkin gate: controlled SWAP
                qc.cswap(ancilla, reg_a[qubit_idx], reg_b[qubit_idx]) 
            
            qc.h(ancilla)
            
            # For now, assume all SWAP tests succeed (ancilla = 0)
            # The purified state remains in reg_a
            new_active_copies.append(copy_a_idx)
        
        active_copies = new_active_copies
        logger.debug(f"After round {round_idx + 1}: {len(active_copies)} active copies")
    
    # Step 4: Add final fidelity measurement
    final_copy_idx = active_copies[0]
    final_qubits = list(range(final_copy_idx * M, (final_copy_idx + 1) * M))
    
    # Add classical register for fidelity measurement
    fidelity_creg = ClassicalRegister(M, "final_fidelity")
    qc.add_register(fidelity_creg)
    
    if target_type == "hadamard":
        add_fidelity_measurement_hadamard(qc, final_qubits, fidelity_creg)
    else:  # ghz
        add_fidelity_measurement_ghz(qc, final_qubits, fidelity_creg)
    
    logger.info(f"Built sequential circuit: {qc.num_qubits} total qubits, depth {qc.depth()}")
    return qc


def build_full_purification_experiment(M: int, noise_spec: NoiseSpec, 
                                      target_type: str = "hadamard", N: int = 2) -> QuantumCircuit:
    """
    Build complete purification experiment circuit.
    
    For N=2: Single SWAP test (original behavior)
    For N>2: Sequential purification with log₂ N rounds
    
    Args:
        M: Number of qubits per copy
        noise_spec: Noise configuration
        target_type: 'hadamard' or 'ghz'
        N: Number of input copies (must be power of 2)
        
    Returns:
        Complete experimental circuit
    """
    if N == 2:
        # Original single SWAP test implementation
        return _build_single_swap_experiment(M, noise_spec, target_type)
    else:
        # Sequential purification for N > 2
        return build_sequential_purification_circuit(N, M, noise_spec, target_type)


def _build_single_swap_experiment(M: int, noise_spec: NoiseSpec, 
                                 target_type: str = "hadamard") -> QuantumCircuit:
    """
    Build single SWAP test experiment (original implementation).
    
    Circuit structure:
    1. Prepare two copies of target state
    2. Apply noise to both copies
    3. Perform SWAP test
    4. Measure ancilla for success
    5. Measure fidelity of remaining state
    
    Args:
        M: Number of qubits per copy
        noise_spec: Noise configuration
        target_type: 'hadamard' or 'ghz'
        
    Returns:
        Complete experimental circuit
    """
    total_qubits = 1 + 2 * M  # ancilla + 2 registers
    qc = QuantumCircuit(total_qubits, name=f"purification_exp_M{M}")
    
    # Define qubit assignments
    ancilla = 0
    reg_A = list(range(1, M + 1))
    reg_B = list(range(M + 1, 2 * M + 1))
    
    # Step 1: Prepare target states on both registers
    if target_type == "hadamard":
        prep_circuit = prepare_hadamard_state(M)
    elif target_type == "ghz":
        prep_circuit = prepare_ghz_state(M)
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
    
    # Apply preparation to both registers
    qc.compose(prep_circuit, qubits=reg_A, inplace=True)
    qc.compose(prep_circuit, qubits=reg_B, inplace=True)
    
    # Step 2: Apply noise to both registers
    if noise_spec.noise_type == NoiseType.depolarizing:
        apply_depolarizing_noise(qc, reg_A, noise_spec.p)
        apply_depolarizing_noise(qc, reg_B, noise_spec.p)
    elif noise_spec.noise_type in [NoiseType.dephase_z, NoiseType.dephase_x]:
        # Use same seed for both copies to ensure identical twirling
        twirl_seed = 42  # Fixed seed for reproducibility
        apply_twirled_dephasing_noise(qc, reg_A, noise_spec.p, twirl_seed)
        apply_twirled_dephasing_noise(qc, reg_B, noise_spec.p, twirl_seed)
    else:
        raise ValueError(f"Unsupported noise type: {noise_spec.noise_type}")
    
    # Step 3: SWAP test
    swap_circuit = build_swap_test_circuit(M)
    qc.compose(swap_circuit, qubits=list(range(total_qubits)), inplace=True)
    
    # Step 4: Add measurements
    # Ancilla measurement for SWAP success
    ancilla_creg = ClassicalRegister(1, "ancilla")
    qc.add_register(ancilla_creg)
    add_ancilla_measurement(qc, ancilla, ancilla_creg)
    
    # Fidelity measurement on register A (after successful SWAP)
    fidelity_creg = ClassicalRegister(M, "fidelity")
    qc.add_register(fidelity_creg)
    if target_type == "hadamard":
        add_fidelity_measurement_hadamard(qc, reg_A, fidelity_creg)
    else:  # ghz
        add_fidelity_measurement_ghz(qc, reg_A, fidelity_creg)
    
    logger.info(f"Built single SWAP experiment: M={M}, noise={noise_spec.noise_type.value}, p={noise_spec.p}")
    return qc


def analyze_sequential_results(counts: Dict[str, int], N: int, M: int, 
                              target_type: str) -> Dict[str, float]:
    """
    Analyze results from sequential purification experiment.
    
    Args:
        counts: Measurement outcome dictionary
        N: Number of input copies
        M: Number of qubits per copy  
        target_type: Target state type
        
    Returns:
        Analysis results dictionary
    """
    import math
    
    num_rounds = int(math.log2(N))
    total_ancillas = sum(N // (2**(i+1)) for i in range(num_rounds))
    
    total_shots = sum(counts.values())
    if total_shots == 0:
        return {'fidelity': 0.0, 'success_probability': 0.0, 'all_swap_success_prob': 0.0}
    
    # Parse measurement outcomes
    # Format: "fidelity_bits ancilla_bits" (bit order may be reversed by Qiskit)
    successful_swap_counts = {}
    all_ancilla_success_count = 0
    
    for outcome, count in counts.items():
        # Split outcome: first M bits are fidelity, rest are ancillas
        if len(outcome) != M + total_ancillas:
            continue  # Skip malformed outcomes
            
        fidelity_bits = outcome[:M]  
        ancilla_bits = outcome[M:]
        
        # Check if ALL SWAP tests succeeded (all ancillas = 0)
        all_swaps_successful = all(bit == '0' for bit in ancilla_bits)
        
        if all_swaps_successful:
            all_ancilla_success_count += count
            if fidelity_bits not in successful_swap_counts:
                successful_swap_counts[fidelity_bits] = 0
            successful_swap_counts[fidelity_bits] += count
    
    # Calculate metrics
    all_swap_success_prob = all_ancilla_success_count / total_shots
    
    # Calculate fidelity (conditioned on all SWAPs succeeding)
    if successful_swap_counts:
        fidelity = calculate_fidelity_from_counts(
            successful_swap_counts, M, target_type
        )
    else:
        fidelity = 0.0
    
    return {
        'fidelity': fidelity,
        'success_probability': all_swap_success_prob,  # Probability all SWAPs succeed
        'total_shots': total_shots,
        'success_shots': all_ancilla_success_count,
        'num_rounds': num_rounds,
    }


__all__ = [
    # Configuration
    "IBMQRunSpec",
    # State preparation
    "prepare_hadamard_state",
    "prepare_ghz_state", 
    # Noise application
    "apply_depolarizing_noise",
    "apply_twirled_dephasing_noise",
    # SWAP test
    "build_swap_test_circuit",
    # Measurements
    "add_fidelity_measurement_hadamard",
    "add_fidelity_measurement_ghz", 
    "add_ancilla_measurement",
    "calculate_fidelity_from_counts",
    # Backend integration
    "setup_ibm_backend",
    "transpile_for_backend",
    "run_circuit_with_retries",
    "save_ibmq_results",
    # High-level assembly
    "build_full_purification_experiment",
    "build_sequential_purification_circuit",
    "analyze_sequential_results",
]