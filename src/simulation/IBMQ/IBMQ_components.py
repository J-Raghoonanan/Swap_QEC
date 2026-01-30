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
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Qiskit imports
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# IBM imports
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
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
    p: float  # Error probability for noise
    noise_type: str = "depolarizing"  # "depolarizing", "dephasing", or "twirled_dephasing"
    backend_name: str = "ibm_torino"
    shots: int = 8192
    transpilation_level: int = 2
    max_retry_attempts: int = 5  # Retry if too many SWAP failures
    min_success_rate: float = 0.1  # Minimum SWAP success rate to accept
    num_noise_realizations: int = 1  # Monte Carlo samples over Pauli noise patterns

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.M <= 0 or self.M > 8:  # Practical limit for NISQ devices
            raise ValueError(f"M must be in [1, 8], got {self.M}")
        if self.N <= 1 or (self.N & (self.N - 1)) != 0:  # Must be power of 2
            raise ValueError(f"N must be a power of 2 and > 1, got {self.N}")
        if not (0.0 <= self.p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {self.p}")
        if self.noise_type not in ["depolarizing", "dephasing", "twirled_dephasing"]:
            raise ValueError(
                "noise_type must be 'depolarizing', 'dephasing', or 'twirled_dephasing', "
                f"got {self.noise_type}"
            )
        if self.shots <= 0:
            raise ValueError(f"shots must be positive, got {self.shots}")
        if self.num_noise_realizations <= 0:
            raise ValueError(
                f"num_noise_realizations must be positive, got {self.num_noise_realizations}"
            )

        # Estimate maximum qubits needed for this configuration:
        # N copies of M qubits each, plus (N-1) ancillas for SWAP tests (one per test)
        max_qubits = self.N * self.M + (self.N - 1)
        if max_qubits > 127:  # Current IBM limits
            raise ValueError(
                f"Configuration requires ~{max_qubits} qubits, exceeds backend limit"
            )

    def synthesize_run_id(self) -> str:
        """Create a unique identifier for this run (physical parameters only)."""
        return f"batch_M{self.M}_N{self.N}_p{self.p:.4f}_{self.noise_type}"


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


# =============================================================================
# Noise Channels (Pauli Realizations)
# =============================================================================

def apply_depolarizing_noise(
    qc: QuantumCircuit, qubits: List[int], p: float, seed: Optional[int] = None
) -> None:
    """
    Apply a single-shot Pauli realization of the depolarizing channel
    to the specified qubits.

    Ideal single-qubit depolarizing channel with strength p is
        Λ(ρ) = (1-p) ρ + (p/3)(XρX + YρY + ZρZ).

    Here we mimic this by, for each qubit independently:
        - With probability 1-p: apply no gate
        - With probability p/3: apply X, Y, or Z

    This produces one concrete Pauli pattern corresponding to a sample
    from the depolarizing channel.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")

    if p == 0.0:
        return  # No noise to apply

    rng = np.random.default_rng(seed)

    for qubit in qubits:
        rand_val = rng.random()
        if rand_val < p / 3.0:        # X error
            qc.x(qubit)
        elif rand_val < 2 * p / 3.0:  # Y error
            qc.y(qubit)
        elif rand_val < p:            # Z error
            qc.z(qubit)
        # else: no error with probability 1-p


def apply_dephasing_noise(
    qc: QuantumCircuit, qubits: List[int], p: float, seed: Optional[int] = None
) -> None:
    """
    Apply a single-shot Pauli realization of a pure dephasing channel.

    Ideal single-qubit dephasing with parameter p is
        Λ(ρ) = (1-p) ρ + p ZρZ.

    We mimic this by, for each qubit independently:
        - With probability p: apply Z
        - With probability 1-p: apply identity
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")

    if p == 0.0:
        return  # No noise to apply

    rng = np.random.default_rng(seed)

    for qubit in qubits:
        if rng.random() < p:
            qc.z(qubit)


def apply_clifford_twirled_dephasing(
    qc: QuantumCircuit, qubits: List[int], p: float, seed: Optional[int] = None
) -> None:
    """
    Apply a Clifford-twirled version of the dephasing channel.

    For each qubit:
        1. Sample a single-qubit Clifford C from a finite generating set
        2. Apply C
        3. Apply dephasing noise as above (Z with probability p)
        4. Apply C†

    This implements a Pauli-twirled variant of dephasing, approximating
    an effectively more isotropic noise model.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")

    if p == 0.0:
        return  # No noise to apply

    rng = np.random.default_rng(seed)

    # Simple Clifford generating set
    clifford_gates = ["i", "h", "s", "sh", "shs", "hs"] 

    for qubit in qubits:
        # Step 1: Apply random Clifford
        clifford = rng.choice(clifford_gates)
        if clifford == "h":
            qc.h(qubit)
        elif clifford == "s":
            qc.s(qubit)
        elif clifford == "sh":
            qc.s(qubit)
            qc.h(qubit)
        elif clifford == "shs":
            qc.s(qubit)
            qc.h(qubit)
            qc.s(qubit)
        elif clifford == "hs":
            qc.h(qubit)
            qc.s(qubit)
        # "i" does nothing

        # Step 2: Apply dephasing noise (Z errors)
        if rng.random() < p:
            qc.z(qubit)

        # Step 3: Apply Clifford inverse
        if clifford == "h":
            qc.h(qubit)      # H† = H
        elif clifford == "s":
            qc.sdg(qubit)    # S† = S†
        elif clifford == "sh":
            qc.h(qubit)      # (SH)† = H†S† = HS†
            qc.sdg(qubit)
        elif clifford == "shs":
            qc.sdg(qubit)    # (SHS)† = S†H†S†
            qc.h(qubit)
            qc.sdg(qubit)
        elif clifford == "hs":
            qc.sdg(qubit)    # (HS)† = S†H†
            qc.h(qubit)


def apply_noise(
    qc: QuantumCircuit,
    qubits: List[int],
    noise_type: str,
    p: float,
    seed: Optional[int] = None,
) -> None:
    """
    Apply specified noise type to qubits as a Pauli-sampled realization.

    Args:
        qc: Circuit to modify in-place
        qubits: Qubits to apply noise to
        noise_type: "depolarizing", "dephasing", or "twirled_dephasing"
        p: Error probability
        seed: Random seed for reproducibility
    """
    if noise_type == "depolarizing":
        apply_depolarizing_noise(qc, qubits, p, seed)
    elif noise_type == "dephasing":
        apply_dephasing_noise(qc, qubits, p, seed)
    elif noise_type == "twirled_dephasing":
        apply_clifford_twirled_dephasing(qc, qubits, p, seed)
    else:
        raise ValueError(
            f"Unknown noise type: {noise_type}. "
            "Must be 'depolarizing', 'dephasing', or 'twirled_dephasing'"
        )


# =============================================================================
# Batch Circuit Construction
# =============================================================================

def create_batch_purification_circuit(
    config: PurificationConfig,
    base_noise_seed: Optional[int] = None,
) -> QuantumCircuit:
    """
    Create complete batch purification circuit.

    Circuit structure:
    1. N registers of M qubits each for noisy copies
    2. (N - 1) ancilla qubits, one per SWAP test in the tree
    3. Prepare all N noisy Hadamard states
    4. Tree of pairwise SWAP tests
    5. Final state in register 0

    Args:
        config: Purification configuration
        base_noise_seed: Base seed for Pauli noise sampling. Each *realization*
                         should use a different base_noise_seed so we Monte Carlo
                         over distinct noise patterns. Within a given realization,
                         we use the *same* seed for all N registers so the N
                         copies are identical, as required by purification theory.

    Returns:
        Complete circuit for batch purification
    """
    M, N = config.M, config.N
    num_levels = int(np.log2(N))  # Number of purification levels

    # Total qubits: N registers of M qubits + ancillas for all SWAP tests
    total_data_qubits = N * M
    total_ancillas = N - 1
    total_qubits = total_data_qubits + total_ancillas

    # Default base seed if not provided
    if base_noise_seed is None:
        base_noise_seed = 12345

    # Create circuit
    qc = QuantumCircuit(total_qubits, name=f"batch_purification_M{M}_N{N}")

    # Define qubit assignments
    # Data registers: [0:M], [M:2M], [2M:3M], ..., [(N-1)M:NM]
    data_registers = [list(range(i * M, (i + 1) * M)) for i in range(N)]
    # Ancillas: [NM : NM + (N-1)]
    ancillas = list(range(total_data_qubits, total_qubits))

    logger.info(
        f"Circuit layout: {N} data registers of {M} qubits, {total_ancillas} ancillas"
    )
    logger.info(f"Data registers: {data_registers}")
    logger.info(f"Ancillas: {ancillas}")

    # Step 1: Prepare all N noisy Hadamard states
    qc.barrier()
    qc.barrier(label="Prepare N noisy Hadamard states")

    for i in range(N):
        # Prepare Hadamard state on register i: |+>^⊗M
        for q in data_registers[i]:
            qc.h(q)

        # Apply identical noise pattern to ALL registers in this realization
        apply_noise(
            qc,
            data_registers[i],
            config.noise_type,
            config.p,
            seed=base_noise_seed,
        )

    # Step 2: Tree of pairwise SWAP purifications
    qc.barrier()
    qc.barrier(label="Tree of SWAP purifications")

    active_registers = list(range(N))
    ancilla_counter = 0

    for level in range(num_levels):
        num_pairs = len(active_registers) // 2
        new_active: List[int] = []

        logger.debug(
            f"Level {level}: {len(active_registers)} active registers, "
            f"{num_pairs} SWAP tests"
        )

        for pair_idx in range(num_pairs):
            reg_a_idx = active_registers[2 * pair_idx]
            reg_b_idx = active_registers[2 * pair_idx + 1]

            reg_a_qubits = data_registers[reg_a_idx]
            reg_b_qubits = data_registers[reg_b_idx]

            ancilla_idx = ancillas[ancilla_counter]
            ancilla_counter += 1

            # SWAP test between register A and register B using its own ancilla
            qc.h(ancilla_idx)
            for i in range(M):
                qc.cswap(ancilla_idx, reg_a_qubits[i], reg_b_qubits[i])
            qc.h(ancilla_idx)

            logger.debug(
                f"  SWAP test {pair_idx}: reg{reg_a_idx} ⊕ reg{reg_b_idx} "
                f"→ ancilla{ancilla_idx}"
            )

            # Only register A's index persists to the next level
            new_active.append(reg_a_idx)

        active_registers = new_active
        qc.barrier()

    # Sanity checks
    assert len(active_registers) == 1 and active_registers[0] == 0
    assert ancilla_counter == total_ancillas

    logger.info(
        f"Batch purification circuit created: depth={qc.depth()}, "
        f"qubits={qc.num_qubits}"
    )
    return qc


def add_measurements(qc: QuantumCircuit, config: PurificationConfig) -> QuantumCircuit:
    """
    Add measurements for post-selection and fidelity calculation.

    We measure:
        - The final purified register (register 0) in the X basis (|+/->)
          by first applying H and then measuring in Z.
        - All ancillas in Z to detect SWAP-test success (ancilla=0).

    The measurement results are stored in a single ClassicalRegister
    compatible with SamplerV2.
    """
    M, N = config.M, config.N
    num_ancillas = N - 1

    total_data_qubits = N * M
    ancilla_start = total_data_qubits

    # Use single classical register for all measurements (SamplerV2 compatible)
    total_measurements = M + num_ancillas
    meas_register = ClassicalRegister(total_measurements, "meas")

    # Create new circuit with single classical register
    measured_qc = QuantumCircuit(qc.num_qubits)
    measured_qc.add_register(meas_register)
    measured_qc.compose(qc, inplace=True)

    # Rotate target |+>^⊗M back to |0>^⊗M for fidelity estimation
    for i in range(M):
        measured_qc.h(i)

    # Measure final state (register 0) into the first M classical bits
    for i in range(M):
        measured_qc.measure(i, meas_register[i])

    # Measure all ancillas into the remaining bits
    for a_idx in range(num_ancillas):
        measured_qc.measure(ancilla_start + a_idx, meas_register[M + a_idx])

    return measured_qc


# =============================================================================
# Execution and Analysis
# =============================================================================

def analyze_results(
    counts: Dict[str, int], config: PurificationConfig
) -> Tuple[float, float]:
    """
    Analyze measurement results with post-selection.

    Args:
        counts: Raw measurement counts from circuit execution
        config: Configuration parameters

    Returns:
        (fidelity, success_probability) tuple where:
        - fidelity: Measured fidelity of |+⟩^⊗M conditioned on all SWAP
                    tests succeeding
        - success_probability: Probability that all SWAP tests succeeded
    """
    M, N = config.M, config.N
    num_ancillas = N - 1

    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0, 0.0

    successful_shots = 0
    perfect_final_state_count = 0

    expected_length = M + num_ancillas

    for outcome_str, count in counts.items():
        if len(outcome_str) != expected_length:
            logger.warning(
                f"Unexpected outcome format: '{outcome_str}' "
                f"(expected {expected_length} bits)"
            )
            continue

        # Qiskit uses little-endian ordering for classical bits:
        # bitstring[0] corresponds to highest classical index.
        # We reverse so index 0 is classical bit 0.
        bitstring = outcome_str[::-1]

        final_state_bits = bitstring[:M]                 # meas[0..M-1]
        ancilla_bits = bitstring[M : M + num_ancillas]   # meas[M..]

        # All SWAP tests succeeded iff all ancillas measured 0
        if ancilla_bits == "0" * num_ancillas:
            successful_shots += count

            # After the final H layer, the ideal |+>^⊗M gives |0>^⊗M in Z.
            if final_state_bits == "0" * M:
                perfect_final_state_count += count

    success_probability = successful_shots / total_shots

    if successful_shots > 0:
        fidelity = perfect_final_state_count / successful_shots
    else:
        fidelity = 0.0

    logger.debug(
        f"Analysis: {successful_shots}/{total_shots} successful shots "
        f"({100*success_probability:.1f}%), "
        f"{perfect_final_state_count}/{successful_shots} perfect final states "
        f"({100*fidelity:.1f}% fidelity)"
    )

    return fidelity, success_probability


def execute_with_retry(
    circuit: QuantumCircuit, config: PurificationConfig, service, backend
) -> Tuple[float, float, Dict[str, int]]:
    """
    Execute circuit with retry on low success rate.

    This function does NOT perform Monte Carlo over noise patterns; it
    assumes the circuit (including noise gates) is fixed. Monte Carlo
    is handled at a higher level by re-building the circuit with
    different noise seeds.

    Args:
        circuit: Circuit to execute
        config: Configuration
        service: IBM service
        backend: IBM quantum backend

    Returns:
        (fidelity, success_probability, all_counts) tuple where counts
        aggregate all shots from all retries of this circuit.
    """
    logger.info("Executing batch purification circuit")

    # Transpile circuit
    transpiled = transpile_circuit_for_backend(
        circuit, backend, config.transpilation_level
    )
    logger.info(
        f"Transpiled circuit: depth={transpiled.depth()}, "
        f"qubits={transpiled.num_qubits}"
    )

    total_attempts = 0
    all_counts: Dict[str, int] = {}

    sampler = SamplerV2(backend)

    while total_attempts < config.max_retry_attempts:
        total_attempts += 1
        logger.info(
            f"Execution attempt {total_attempts}/{config.max_retry_attempts}"
        )

        try:
            logger.info("📤 Submitting job to quantum backend...")
            job = sampler.run([transpiled], shots=config.shots)

            logger.info(
                f"⏳ Waiting for job {job.job_id()} to complete (max 300s)..."
            )
            result = job.result(timeout=300)

            logger.info("✅ Job completed successfully")

        except Exception as e:
            if "timeout" in str(e).lower():
                logger.error(
                    "⏰ Job timed out after 300 seconds - backend queue likely busy"
                )
                raise RuntimeError(
                    "Job timeout - try different backend or smaller experiment"
                )
            else:
                logger.error(f"💥 Job execution failed: {e}")
                raise

        pub_result = result[0]

        # Try to get measurement counts from the 'meas' register
        if hasattr(pub_result.data, "meas"):
            counts = pub_result.data.meas.get_counts()
            logger.debug("Found measurement counts via 'meas' register")
        else:
            data_attrs = [
                attr for attr in dir(pub_result.data) if not attr.startswith("_")
            ]
            logger.info(f"Available data attributes: {data_attrs}")

            counts = None
            for attr_name in data_attrs:
                try:
                    attr = getattr(pub_result.data, attr_name)
                    if hasattr(attr, "get_counts"):
                        counts = attr.get_counts()
                        logger.info(
                            f"Found counts via '{attr_name}.get_counts()'"
                        )
                        break
                except Exception as e:
                    logger.debug(f"Failed to access {attr_name}: {e}")

            if counts is None:
                raise RuntimeError(
                    "Cannot find measurement counts in SamplerV2 result. "
                    f"Available data attributes: {data_attrs}"
                )

        logger.debug(
            f"Retrieved {len(counts)} unique measurement outcomes, "
            f"total shots: {sum(counts.values())}"
        )

        # Accumulate counts across attempts for this fixed circuit
        for outcome, count in counts.items():
            all_counts[outcome] = all_counts.get(outcome, 0) + count

        # Analyze current results for this circuit
        fidelity, success_prob = analyze_results(all_counts, config)
        total_shots_so_far = sum(all_counts.values())
        successful_shots = total_shots_so_far * success_prob

        logger.info(
            f"  Attempt {total_attempts}: success_rate={success_prob:.4f}, "
            f"successful_shots={successful_shots:.0f}"
        )

        if (
            success_prob >= config.min_success_rate
            or total_attempts >= config.max_retry_attempts
        ):
            break

    final_fidelity, final_success_prob = analyze_results(all_counts, config)

    logger.info(f"Final results after {total_attempts} attempts:")
    logger.info(f"  Fidelity (this circuit): {final_fidelity:.4f}")
    logger.info(f"  Success probability (this circuit): {final_success_prob:.4f}")
    logger.info(f"  Total shots (this circuit): {sum(all_counts.values())}")

    return final_fidelity, final_success_prob, all_counts


# =============================================================================
# Backend Setup
# =============================================================================

def setup_ibm_backend(backend_name: str):
    """Setup IBM quantum backend."""
    if not IBM_RUNTIME_AVAILABLE:
        raise ImportError(
            "IBM Runtime not available. Install qiskit-ibm-runtime."
        )

    service = QiskitRuntimeService()
    backend = service.backend(backend_name)

    logger.info(f"Connected to backend: {backend_name}")
    logger.info(f"Backend status: {backend.status()}")

    return service, backend


def transpile_circuit_for_backend(
    circuit: QuantumCircuit, backend, optimization_level: int = 2
):
    """Transpile circuit allowing full backend topology for connectivity."""

    num_circuit_qubits = circuit.num_qubits
    backend_qubits = backend.configuration().n_qubits

    logger.info(
        f"Circuit requires {num_circuit_qubits} qubits, backend has {backend_qubits}"
    )

    if num_circuit_qubits > backend_qubits:
        raise ValueError(
            f"Circuit needs {num_circuit_qubits} qubits but backend "
            f"only has {backend_qubits}"
        )

    pass_manager = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
    )

    transpiled = pass_manager.run(circuit)

    logger.info(
        f"Transpiled: {num_circuit_qubits} logical → {transpiled.num_qubits} "
        f"physical qubits"
    )
    logger.info(
        f"Circuit depth: {circuit.depth()} → {transpiled.depth()}"
    )

    return transpiled


# =============================================================================
# Main Experiment Function
# =============================================================================

def run_complete_purification_experiment(
    config: PurificationConfig, service=None, backend=None
) -> Dict:
    """
    Run the complete batch purification experiment with Monte Carlo
    over Pauli noise realizations.

    For each noise realization r = 1..num_noise_realizations:
        - Build a fresh noisy circuit with a different base_noise_seed
        - Run it with execute_with_retry (for SWAP-success post-selection)
        - Aggregate all measurement counts across realizations

    Final fidelity and success probability are computed once from the
    aggregated counts, approximating the channel-averaged behavior.
    """
    config.validate()

    logger.info(
        f"Starting batch purification experiment: {config.synthesize_run_id()}"
    )
    logger.info(
        f"Configuration: M={config.M}, N={config.N}, p={config.p}, "
        f"noise={config.noise_type}, "
        f"num_noise_realizations={config.num_noise_realizations}"
    )

    # Step 0: Setup backend if needed
    if service is None or backend is None:
        service, backend = setup_ibm_backend(config.backend_name)

    # Global aggregation of counts across all noise realizations
    global_counts: Dict[str, int] = {}

    measured_circuit: Optional[QuantumCircuit] = None

    for r in range(config.num_noise_realizations):
        base_noise_seed = 100000 * r  # Large stride to avoid overlaps
        logger.info(
            f"--- Noise realization {r+1}/{config.num_noise_realizations} "
            f"(base_noise_seed={base_noise_seed}) ---"
        )

        # Step 1: Create batch purification circuit for this realization
        purification_circuit = create_batch_purification_circuit(
            config,
            base_noise_seed=base_noise_seed,
        )

        # Step 2: Add measurements
        measured_circuit = add_measurements(purification_circuit, config)

        # Step 3: Execute with retry (for this fixed noise pattern)
        fidelity_r, success_prob_r, counts_r = execute_with_retry(
            measured_circuit, config, service, backend
        )

        logger.info(
            f"Realization {r+1}: fidelity={fidelity_r:.4f}, "
            f"success_prob={success_prob_r:.4f}, "
            f"shots={sum(counts_r.values())}"
        )

        # Step 4: Aggregate counts across realizations
        for outcome, count in counts_r.items():
            global_counts[outcome] = global_counts.get(outcome, 0) + count

    # Step 5: Analyze aggregated counts (channel-averaged estimate)
    final_fidelity, final_success_prob = analyze_results(global_counts, config)

    logger.info("=== Aggregated results over all noise realizations ===")
    logger.info(f"  Total shots: {sum(global_counts.values())}")
    logger.info(f"  Final fidelity: {final_fidelity:.4f}")
    logger.info(f"  Final success probability: {final_success_prob:.4f}")

    # Package results
    # measured_circuit is guaranteed not None because num_noise_realizations>0
    results = {
        "run_id": config.synthesize_run_id(),
        "M": config.M,
        "N": config.N,
        "p": config.p,
        "noise_type": config.noise_type,
        "max_purification_level": int(np.log2(config.N)),
        "final_fidelity": final_fidelity,
        "swap_success_probability": final_success_prob,
        "total_shots": sum(global_counts.values()),
        "backend_name": config.backend_name,
        "circuit_depth": measured_circuit.depth() if measured_circuit is not None else -1,
        "circuit_qubits": measured_circuit.num_qubits if measured_circuit is not None else -1,
        "num_noise_realizations": config.num_noise_realizations,
        "error_message": "",  # Explicitly empty on success for CSV consistency
    }

    logger.info(
        f"Experiment complete (channel-averaged): fidelity={final_fidelity:.4f}, "
        f"success_prob={final_success_prob:.4f}"
    )
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
    "apply_dephasing_noise",
    "apply_clifford_twirled_dephasing",
    "apply_noise",
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
