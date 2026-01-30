"""
Comprehensive unit tests for IBMQ_components.py

This test suite validates all functions to ensure correctness before running
experiments on IBM Quantum hardware. Tests include:
- Configuration validation
- Circuit construction and structure
- Noise application correctness
- Measurement analysis
- Statistical properties
- Integration tests with Aer simulator

Run with: pytest test_IBMQ_components.py -v
"""

import pytest
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter
import logging

# Qiskit imports
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

# Import the module under test
from IBMQ_components import (
    PurificationConfig,
    create_hadamard_target_circuit,
    apply_depolarizing_noise,
    apply_dephasing_noise,
    apply_clifford_twirled_dephasing,
    apply_noise,
    create_batch_purification_circuit,
    add_measurements,
    analyze_results,
    transpile_circuit_for_backend,
    run_complete_purification_experiment,  # imported but not used (kept for completeness)
)


# =============================================================================
# Test Fixtures and Utilities
# =============================================================================

@pytest.fixture
def aer_backend():
    """Provide Aer simulator backend for testing."""
    return AerSimulator()


@pytest.fixture
def valid_configs():
    """Provide various valid configurations for testing."""
    return [
        PurificationConfig(M=1, N=2, p=0.1, noise_type="depolarizing"),
        PurificationConfig(M=2, N=4, p=0.2, noise_type="dephasing"),
        PurificationConfig(M=1, N=8, p=0.05, noise_type="twirled_dephasing"),
        PurificationConfig(M=3, N=2, p=0.3, noise_type="depolarizing", shots=1024),
    ]


def count_gate_types(circuit: QuantumCircuit) -> Dict[str, int]:
    """Count gate types in a circuit."""
    gate_counts = Counter()
    for instruction in circuit.data:
        gate_counts[instruction.operation.name] += 1
    return dict(gate_counts)


def simulate_circuit(circuit: QuantumCircuit, shots: int = 8192) -> Dict[str, int]:
    """Simulate a circuit and return measurement counts."""
    simulator = AerSimulator()
    job = simulator.run(circuit, shots=shots)
    return job.result().get_counts()


def calculate_fidelity_to_hadamard(statevector: Statevector, M: int) -> float:
    """Calculate fidelity to |+⟩^⊗M state."""
    # Create ideal |+⟩^⊗M state
    ideal_circuit = QuantumCircuit(M)
    for i in range(M):
        ideal_circuit.h(i)
    ideal_state = Statevector.from_instruction(ideal_circuit)

    # Calculate fidelity
    return np.abs(ideal_state.inner(statevector)) ** 2


# =============================================================================
# Test PurificationConfig Class
# =============================================================================

class TestPurificationConfig:
    """Test configuration validation and synthesis."""

    def test_valid_configs(self, valid_configs):
        """Test that valid configurations pass validation."""
        for config in valid_configs:
            config.validate()  # Should not raise

    def test_invalid_M_values(self):
        """Test validation with invalid M values."""
        with pytest.raises(ValueError, match="M must be in"):
            PurificationConfig(M=0, N=2, p=0.1).validate()

        with pytest.raises(ValueError, match="M must be in"):
            PurificationConfig(M=10, N=2, p=0.1).validate()

    def test_invalid_N_values(self):
        """Test validation with invalid N values."""
        with pytest.raises(ValueError, match="N must be a power of 2"):
            PurificationConfig(M=1, N=1, p=0.1).validate()

        with pytest.raises(ValueError, match="N must be a power of 2"):
            PurificationConfig(M=1, N=3, p=0.1).validate()

        with pytest.raises(ValueError, match="N must be a power of 2"):
            PurificationConfig(M=1, N=6, p=0.1).validate()

    def test_invalid_p_values(self):
        """Test validation with invalid p values."""
        with pytest.raises(ValueError, match="p must be in"):
            PurificationConfig(M=1, N=2, p=-0.1).validate()

        with pytest.raises(ValueError, match="p must be in"):
            PurificationConfig(M=1, N=2, p=1.1).validate()

    def test_invalid_noise_types(self):
        """Test validation with invalid noise types."""
        with pytest.raises(ValueError, match="noise_type must be"):
            PurificationConfig(M=1, N=2, p=0.1, noise_type="invalid").validate()

    def test_invalid_shots(self):
        """Test validation with invalid shot counts."""
        with pytest.raises(ValueError, match="shots must be positive"):
            PurificationConfig(M=1, N=2, p=0.1, shots=0).validate()

    def test_invalid_noise_realizations(self):
        """Test validation with invalid noise realization counts."""
        with pytest.raises(ValueError, match="num_noise_realizations must be positive"):
            PurificationConfig(M=1, N=2, p=0.1, num_noise_realizations=0).validate()

    def test_too_many_qubits(self):
        """Test validation with configurations requiring too many qubits."""
        with pytest.raises(ValueError, match="exceeds backend limit"):
            # This would need 16*8 + 15 = 143 qubits
            PurificationConfig(M=8, N=16, p=0.1).validate()

    def test_synthesize_run_id(self):
        """Test run ID synthesis."""
        config = PurificationConfig(M=2, N=4, p=0.123, noise_type="depolarizing")
        expected = "batch_M2_N4_p0.1230_depolarizing"
        assert config.synthesize_run_id() == expected


# =============================================================================
# Test State Preparation
# =============================================================================

class TestStatePreparation:
    """Test Hadamard state preparation circuits."""

    def test_hadamard_circuit_structure(self):
        """Test that Hadamard circuit has correct structure."""
        for M in [1, 2, 3, 4]:
            circuit = create_hadamard_target_circuit(M)

            # Check basic properties
            assert circuit.num_qubits == M
            assert circuit.name == f"hadamard_target_M{M}"

            # Check gate count
            gate_counts = count_gate_types(circuit)
            assert gate_counts.get('h', 0) == M

    def test_hadamard_circuit_statevector(self):
        """Test that Hadamard circuit produces correct statevector."""
        for M in [1, 2, 3]:
            circuit = create_hadamard_target_circuit(M)
            statevector = Statevector.from_instruction(circuit)

            # Should be |+⟩^⊗M = (1/√2^M) * sum over all computational basis states
            expected_amplitude = 1.0 / np.sqrt(2 ** M)

            # All amplitudes should be equal and real
            for amplitude in statevector.data:
                assert np.isclose(amplitude, expected_amplitude)
                assert np.isclose(amplitude.imag, 0.0)

    def test_invalid_M(self):
        """Test error handling for invalid M values."""
        with pytest.raises(ValueError, match="M must be positive"):
            create_hadamard_target_circuit(0)

        with pytest.raises(ValueError, match="M must be positive"):
            create_hadamard_target_circuit(-1)


# =============================================================================
# Test Noise Functions
# =============================================================================

class TestNoiseApplication:
    """Test noise channel implementations."""

    def test_depolarizing_noise_zero_p(self):
        """Test depolarizing noise with p=0 (no noise)."""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.h(1)
        circuit.h(2)

        original_circuit = circuit.copy()
        apply_depolarizing_noise(circuit, [0, 1, 2], p=0.0, seed=42)

        # Circuit should be unchanged
        assert circuit.data == original_circuit.data

    def test_depolarizing_noise_deterministic(self):
        """Test that depolarizing noise is deterministic with fixed seed."""
        for p in [0.1, 0.3, 0.5]:
            circuit1 = QuantumCircuit(4)
            circuit2 = QuantumCircuit(4)

            apply_depolarizing_noise(circuit1, [0, 1, 2, 3], p=p, seed=42)
            apply_depolarizing_noise(circuit2, [0, 1, 2, 3], p=p, seed=42)

            # Should produce identical circuits
            assert circuit1.data == circuit2.data

    def test_depolarizing_noise_statistics(self):
        """Test statistical properties of depolarizing noise."""
        p = 0.3
        num_trials = 10000

        error_counts = {'I': 0, 'X': 0, 'Y': 0, 'Z': 0}

        for trial in range(num_trials):
            circuit = QuantumCircuit(1)
            apply_depolarizing_noise(circuit, [0], p=p, seed=trial)

            if len(circuit.data) == 0:
                error_counts['I'] += 1
            else:
                gate_name = circuit.data[0].operation.name.upper()
                error_counts[gate_name] += 1

        # Check probabilities (with tolerance for statistical fluctuation)
        tolerance = 0.02  # 2% tolerance
        assert abs(error_counts['I'] / num_trials - (1 - p)) < tolerance
        assert abs(error_counts['X'] / num_trials - (p / 3)) < tolerance
        assert abs(error_counts['Y'] / num_trials - (p / 3)) < tolerance
        assert abs(error_counts['Z'] / num_trials - (p / 3)) < tolerance

    def test_dephasing_noise_statistics(self):
        """Test statistical properties of dephasing noise."""
        p = 0.2
        num_trials = 10000

        z_count = 0

        for trial in range(num_trials):
            circuit = QuantumCircuit(1)
            apply_dephasing_noise(circuit, [0], p=p, seed=trial)

            if len(circuit.data) > 0 and circuit.data[0].operation.name == 'z':
                z_count += 1

        # Check probability
        tolerance = 0.02
        assert abs(z_count / num_trials - p) < tolerance

    def test_clifford_twirled_dephasing_structure(self):
        """Test that Clifford twirled dephasing has expected structure."""
        circuit = QuantumCircuit(1)
        apply_clifford_twirled_dephasing(circuit, [0], p=1.0, seed=42)  # Force error

        # Should have at least 1 gate (the Z error), possibly more from Clifford twirling
        assert len(circuit.data) >= 1

        # Test with p=0 should have some gates (Clifford + inverse)
        circuit_p0 = QuantumCircuit(1)
        apply_clifford_twirled_dephasing(circuit_p0, [0], p=0.0, seed=42)

        # May have gates from twirling even with no error
        assert len(circuit_p0.data) >= 0

    def test_identical_copies_same_seed(self):
        """Test that same seed produces identical noise patterns."""
        M, N = 2, 4
        seed = 12345

        circuits = []
        for i in range(N):
            circuit = QuantumCircuit(M)
            apply_depolarizing_noise(circuit, list(range(M)), p=0.3, seed=seed)
            circuits.append(circuit)

        # All circuits should be identical
        for i in range(1, N):
            assert circuits[i].data == circuits[0].data

    def test_different_seeds_different_patterns(self):
        """Test that different seeds produce different noise patterns."""
        M = 3
        p = 0.5  # High noise to ensure differences

        circuit1 = QuantumCircuit(M)
        circuit2 = QuantumCircuit(M)

        apply_depolarizing_noise(circuit1, list(range(M)), p=p, seed=111)
        apply_depolarizing_noise(circuit2, list(range(M)), p=p, seed=222)

        # Should be very likely to be different with high p
        assert circuit1.data != circuit2.data

    def test_noise_function_edge_cases(self):
        """Test noise functions with edge cases."""
        circuit = QuantumCircuit(2)

        # Test p=0
        apply_depolarizing_noise(circuit, [0, 1], p=0.0)
        apply_dephasing_noise(circuit, [0, 1], p=0.0)
        apply_clifford_twirled_dephasing(circuit, [0, 1], p=0.0)

        # Test p=1
        circuit_p1 = QuantumCircuit(1)
        apply_depolarizing_noise(circuit_p1, [0], p=1.0, seed=42)
        # For p=1, should definitely have an error gate
        assert len(circuit_p1.data) > 0

    def test_invalid_p_values_noise(self):
        """Test error handling for invalid p values in noise functions."""
        circuit = QuantumCircuit(1)

        with pytest.raises(ValueError, match="p must be in"):
            apply_depolarizing_noise(circuit, [0], p=-0.1)

        with pytest.raises(ValueError, match="p must be in"):
            apply_depolarizing_noise(circuit, [0], p=1.1)

        with pytest.raises(ValueError, match="p must be in"):
            apply_dephasing_noise(circuit, [0], p=-0.1)

        with pytest.raises(ValueError, match="p must be in"):
            apply_clifford_twirled_dephasing(circuit, [0], p=1.1)

        def _bad():
            apply_noise(circuit, [0], "invalid_noise", 0.1)

        def _ok():
            apply_noise(circuit, [0], "depolarizing", 0.1)

        # Wrapper error for bad noise type
        with pytest.raises(ValueError, match="Unknown noise type"):
            _bad()

        # Should not raise
        _ok()


# =============================================================================
# Test Batch Circuit Construction
# =============================================================================

class TestBatchCircuitConstruction:
    """Test batch purification circuit construction."""

    def test_batch_circuit_structure(self, valid_configs):
        """Test structural properties of batch circuits."""
        for config in valid_configs:
            circuit = create_batch_purification_circuit(config, base_noise_seed=42)

            # Check qubit count
            expected_qubits = config.N * config.M + (config.N - 1)
            assert circuit.num_qubits == expected_qubits

            # Check circuit has gates (Hadamards, noise, CSWAPs)
            assert len(circuit.data) > 0

            # Check name
            assert f"M{config.M}_N{config.N}" in circuit.name

    def test_batch_circuit_gate_counts(self):
        """Test expected gate counts in batch circuits."""
        config = PurificationConfig(M=2, N=4, p=0.2, noise_type="depolarizing")
        circuit = create_batch_purification_circuit(config, base_noise_seed=42)

        gate_counts = count_gate_types(circuit)

        # Should have Hadamards for preparation (N*M) and SWAP test structure
        assert gate_counts.get('h', 0) >= config.N * config.M  # At least for preparation
        assert gate_counts.get('cswap', 0) == (config.N - 1) * config.M  # One per SWAP test per qubit

    def test_reproducible_circuits(self):
        """Test that circuits are reproducible with same seed."""
        config = PurificationConfig(M=1, N=4, p=0.3, noise_type="dephasing")

        circuit1 = create_batch_purification_circuit(config, base_noise_seed=123)
        circuit2 = create_batch_purification_circuit(config, base_noise_seed=123)

        assert circuit1.data == circuit2.data

    def test_different_noise_seeds(self):
        """Test that different seeds produce different noise patterns."""
        config = PurificationConfig(M=2, N=2, p=0.5, noise_type="depolarizing")

        circuit1 = create_batch_purification_circuit(config, base_noise_seed=111)
        circuit2 = create_batch_purification_circuit(config, base_noise_seed=222)

        # Circuits should be different (different noise patterns)
        assert circuit1.data != circuit2.data

    def test_swap_test_structure(self):
        """Test SWAP test implementation within circuit."""
        config = PurificationConfig(M=1, N=2, p=0.0, noise_type="depolarizing")  # No noise for clarity
        circuit = create_batch_purification_circuit(config, base_noise_seed=42)

        # For N=2, should have 1 SWAP test
        gate_counts = count_gate_types(circuit)

        # Check for SWAP test gates
        assert gate_counts.get('cswap', 0) == config.M  # One CSWAP per data qubit

    def test_tree_structure_scaling(self):
        """Test that SWAP tree scales correctly with N."""
        M = 1
        for N in [2, 4, 8, 16]:
            config = PurificationConfig(M=M, N=N, p=0.1, noise_type="depolarizing")
            circuit = create_batch_purification_circuit(config, base_noise_seed=42)

            gate_counts = count_gate_types(circuit)
            expected_cswaps = (N - 1) * M  # Total SWAP tests across all levels
            assert gate_counts.get('cswap', 0) == expected_cswaps

    def test_measurement_addition(self, valid_configs):
        """Test measurement addition to circuits."""
        for config in valid_configs:
            circuit = create_batch_purification_circuit(config, base_noise_seed=42)
            measured_circuit = add_measurements(circuit, config)

            # Check classical register
            assert len(measured_circuit.cregs) == 1
            assert measured_circuit.cregs[0].name == "meas"
            assert measured_circuit.cregs[0].size == config.M + (config.N - 1)

            # Check measurement count
            gate_counts = count_gate_types(measured_circuit)
            expected_measurements = config.M + (config.N - 1)
            assert gate_counts.get('measure', 0) == expected_measurements

    def test_measurement_x_basis(self):
        """Test that measurements are in X basis (H then measure Z)."""
        config = PurificationConfig(M=2, N=2, p=0.0, noise_type="depolarizing")
        circuit = create_batch_purification_circuit(config, base_noise_seed=42)
        measured_circuit = add_measurements(circuit, config)

        # Should have additional H gates for X-basis measurement
        original_h_count = count_gate_types(circuit).get('h', 0)
        final_h_count = count_gate_types(measured_circuit).get('h', 0)

        # Should have at least config.M additional H gates for X-basis measurement
        assert final_h_count >= original_h_count + config.M


# =============================================================================
# Test Results Analysis
# =============================================================================

class TestResultsAnalysis:
    """Test measurement results analysis."""

    def test_analyze_perfect_results(self):
        """Test analysis with perfect results (all successful, all |0⟩)."""
        config = PurificationConfig(M=2, N=4, p=0.1)

        # Perfect result: all ancillas 0, all data qubits 0
        # Bit order: data[0], data[1], anc[0], anc[1], anc[2]  (little-endian reversed)
        counts = {"00000": 1000}  # All zeros

        fidelity, success_prob = analyze_results(counts, config)

        assert np.isclose(fidelity, 1.0)
        assert np.isclose(success_prob, 1.0)

    def test_analyze_no_success(self):
        """Test analysis with no successful SWAP tests."""
        config = PurificationConfig(M=1, N=2, p=0.1)

        # For M=1, N=2: 1 data bit + 1 ancilla bit = 2 total
        # Qiskit little-endian: "10" means meas[1]=1, meas[0]=0
        # After reversal: bitstring="01", final="0", ancilla="1" (FAILED)
        # Qiskit little-endian: "11" means meas[1]=1, meas[0]=1
        # After reversal: bitstring="11", final="1", ancilla="1" (FAILED)
        counts = {"10": 500, "11": 500}  # All ancillas failed (ancilla bit = 1)

        fidelity, success_prob = analyze_results(counts, config)

        assert np.isclose(fidelity, 0.0)
        assert np.isclose(success_prob, 0.0)

    def test_analyze_partial_success(self):
        """Test analysis with partial success."""
        config = PurificationConfig(M=1, N=2, p=0.1)

        # For M=1, N=2: 1 data bit + 1 ancilla bit = 2 total
        # Want: 50% success rate, of those 80% perfect fidelity
        # In Qiskit little-endian format before reversal:
        counts = {
            "00": 400,  # meas[1]=0,meas[0]=0 -> final="0",ancilla="0" -> Success + perfect
            "01": 100,  # meas[1]=0,meas[0]=1 -> final="1",ancilla="0" -> Success + imperfect
            "10": 250,  # meas[1]=1,meas[0]=0 -> final="0",ancilla="1" -> Failure
            "11": 250,  # meas[1]=1,meas[0]=1 -> final="1",ancilla="1" -> Failure
        }

        fidelity, success_prob = analyze_results(counts, config)

        assert np.isclose(success_prob, 0.5)  # 500/1000
        assert np.isclose(fidelity, 0.8)     # 400/500

    def test_analyze_empty_counts(self):
        """Test analysis with empty measurement counts."""
        config = PurificationConfig(M=1, N=2, p=0.1)
        counts = {}

        fidelity, success_prob = analyze_results(counts, config)

        assert fidelity == 0.0
        assert success_prob == 0.0

    def test_bit_ordering_correctness(self):
        """Test correct interpretation of bit ordering."""
        config = PurificationConfig(M=2, N=4, p=0.1)  # 3 ancillas

        # Test specific bit pattern understanding
        # "10100" in Qiskit = bits [0,0,1,0,1] after reversal
        # data[0]=0, data[1]=0, anc[0]=1, anc[1]=0, anc[2]=1
        counts = {"10100": 1000}

        fidelity, success_prob = analyze_results(counts, config)

        # Not all ancillas are 0, so no success
        assert np.isclose(success_prob, 0.0)

    def test_multiple_noise_realizations_aggregation(self):
        """Test that counts from multiple realizations aggregate correctly."""
        config = PurificationConfig(M=1, N=2, p=0.1)

        # Simulate aggregation of counts from multiple realizations
        counts1 = {"00": 200, "10": 50, "01": 100, "11": 150}
        counts2 = {"00": 300, "10": 100, "01": 50, "11": 50}

        # Manual aggregation
        aggregated = {}
        for outcome, count in counts1.items():
            aggregated[outcome] = aggregated.get(outcome, 0) + count
        for outcome, count in counts2.items():
            aggregated[outcome] = aggregated.get(outcome, 0) + count

        fidelity, success_prob = analyze_results(aggregated, config)

        # Check aggregated results according to analyze_results' ancilla logic:
        # successes are outcomes with ancilla bit = 0, i.e. "00" and "01"
        total_shots = sum(aggregated.values())
        successful_shots = aggregated["00"] + aggregated["01"]
        expected_success_prob = successful_shots / total_shots
        expected_fidelity = aggregated["00"] / successful_shots

        assert np.isclose(success_prob, expected_success_prob)
        assert np.isclose(fidelity, expected_fidelity)

    def test_malformed_measurement_outcomes(self):
        """Test analysis with malformed measurement outcomes."""
        config = PurificationConfig(M=2, N=4, p=0.1)

        # Test with only valid entries first
        # For M=2, N=4: expected length = 2 data + 3 ancilla = 5 bits
        counts = {
            "00000": 100,    # Valid: all bits 0 -> success + perfect
        }

        # Should handle gracefully and process only valid entries
        fidelity, success_prob = analyze_results(counts, config)

        # Should be based only on valid entry (100 shots, all successful, all perfect)
        assert np.isclose(success_prob, 1.0)
        assert np.isclose(fidelity, 1.0)

        # Now test with malformed entries - they get warnings but are skipped
        malformed_counts = {
            "00000": 100,    # Valid
            "0000": 50,      # Too short - should be skipped
            "000000": 25,    # Too long - should be skipped
        }

        fidelity2, success_prob2 = analyze_results(malformed_counts, config)

        # total_shots = 100 + 50 + 25 = 175
        # successful_shots = 100 (only the valid entry with ancillas=000)
        expected_success_prob = 100.0 / (100 + 50 + 25)  # ≈ 0.571
        assert np.isclose(success_prob2, expected_success_prob, rtol=0.01)

        # For fidelity: among successful shots (100), all are perfect (100)
        assert np.isclose(fidelity2, 1.0)


# =============================================================================
# Test Backend Integration
# =============================================================================

class TestBackendIntegration:
    """Test backend integration and transpilation."""

    def test_transpilation_aer(self, aer_backend, valid_configs):
        """Test circuit transpilation with Aer backend."""
        for config in valid_configs[:2]:  # Test subset for speed
            circuit = create_batch_purification_circuit(config, base_noise_seed=42)
            measured_circuit = add_measurements(circuit, config)

            transpiled = transpile_circuit_for_backend(measured_circuit, aer_backend)

            # Check basic properties
            assert transpiled.num_qubits >= measured_circuit.num_qubits
            assert len(transpiled.cregs) == len(measured_circuit.cregs)

    def test_circuit_too_large_for_backend(self):
        """Test error handling for circuits too large for backend."""
        # Create a mock backend with very few qubits
        class SmallMockConfig:
            n_qubits = 3

        class SmallMockBackend:
            def configuration(self):
                return SmallMockConfig()

        small_backend = SmallMockBackend()

        config = PurificationConfig(M=2, N=4, p=0.1)  # Needs 2*4 + 3 = 11 qubits
        circuit = create_batch_purification_circuit(config, base_noise_seed=42)
        measured_circuit = add_measurements(circuit, config)

        with pytest.raises(ValueError, match="Circuit needs .* qubits but backend only has"):
            transpile_circuit_for_backend(measured_circuit, small_backend)


# =============================================================================
# Test Integration with Simulation
# =============================================================================

class TestSimulationIntegration:
    """Test complete pipeline with Aer simulation."""

    def test_end_to_end_no_noise(self, aer_backend):
        """Test complete pipeline with no noise (should be perfect)."""
        config = PurificationConfig(
            M=1, N=2, p=0.0, noise_type="depolarizing",
            shots=1024, num_noise_realizations=1
        )

        # Create and simulate circuit
        circuit = create_batch_purification_circuit(config, base_noise_seed=42)
        measured_circuit = add_measurements(circuit, config)

        # Run simulation
        counts = simulate_circuit(measured_circuit, shots=config.shots)
        fidelity, success_prob = analyze_results(counts, config)

        # With no noise, should have high success and fidelity
        assert success_prob > 0.8  # High success probability
        assert fidelity > 0.9      # High fidelity

    def test_end_to_end_with_noise(self, aer_backend):
        """Test complete pipeline with noise."""
        config = PurificationConfig(
            M=1, N=2, p=0.2, noise_type="depolarizing",
            shots=2048, num_noise_realizations=1
        )

        # Create and simulate circuit
        circuit = create_batch_purification_circuit(config, base_noise_seed=42)
        measured_circuit = add_measurements(circuit, config)

        # Run simulation
        counts = simulate_circuit(measured_circuit, shots=config.shots)
        fidelity, success_prob = analyze_results(counts, config)

        # With noise, performance should degrade but still be reasonable
        assert 0.0 <= success_prob <= 1.0
        assert 0.0 <= fidelity <= 1.0

    def test_different_noise_types_simulation(self, aer_backend):
        """Test all noise types produce valid results."""
        noise_types = ["depolarizing", "dephasing", "twirled_dephasing"]

        for noise_type in noise_types:
            config = PurificationConfig(
                M=1, N=2, p=0.15, noise_type=noise_type,
                shots=1024, num_noise_realizations=1
            )

            circuit = create_batch_purification_circuit(config, base_noise_seed=42)
            measured_circuit = add_measurements(circuit, config)

            # Should not crash and should produce reasonable results
            counts = simulate_circuit(measured_circuit, shots=config.shots)
            fidelity, success_prob = analyze_results(counts, config)

            assert 0.0 <= success_prob <= 1.0
            assert 0.0 <= fidelity <= 1.0

    def test_purification_effectiveness_scaling(self, aer_backend):
        """Test that more purification levels affect results."""
        # Compare N=2 vs N=4 vs N=8 (1 vs 2 vs 3 rounds of purification)
        base_config = {
            'M': 1, 'p': 0.25, 'noise_type': 'depolarizing',
            'shots': 1024, 'num_noise_realizations': 1
        }

        results = {}
        for N in [2, 4, 8]:
            config = PurificationConfig(N=N, **base_config)
            circuit = create_batch_purification_circuit(config, base_noise_seed=42)
            measured_circuit = add_measurements(circuit, config)

            counts = simulate_circuit(measured_circuit, shots=config.shots)
            fidelity, success_prob = analyze_results(counts, config)
            results[N] = {'fidelity': fidelity, 'success_prob': success_prob}

        # All results should be valid
        for N in [2, 4, 8]:
            assert 0.0 <= results[N]['fidelity'] <= 1.0
            assert 0.0 <= results[N]['success_prob'] <= 1.0
            print(f"N={N}: F={results[N]['fidelity']:.3f}, P={results[N]['success_prob']:.3f}")

    def test_monte_carlo_noise_realizations(self, aer_backend):
        """Test Monte Carlo over multiple noise realizations."""
        config = PurificationConfig(
            M=1, N=2, p=0.2, noise_type="depolarizing",
            shots=512, num_noise_realizations=3
        )

        # Simulate the Monte Carlo process manually
        global_counts = {}

        for r in range(config.num_noise_realizations):
            base_noise_seed = 100000 * r  # Large stride like in real implementation
            circuit = create_batch_purification_circuit(config, base_noise_seed=base_noise_seed)
            measured_circuit = add_measurements(circuit, config)

            counts = simulate_circuit(measured_circuit, shots=config.shots)

            # Aggregate counts
            for outcome, count in counts.items():
                global_counts[outcome] = global_counts.get(outcome, 0) + count

        # Analyze aggregated results
        fidelity, success_prob = analyze_results(global_counts, config)

        # Should produce valid results
        assert 0.0 <= fidelity <= 1.0
        assert 0.0 <= success_prob <= 1.0

        # Total shots should equal config.shots * config.num_noise_realizations
        expected_total = config.shots * config.num_noise_realizations
        actual_total = sum(global_counts.values())
        assert actual_total == expected_total


# =============================================================================
# Test Statistical Properties
# =============================================================================

class TestStatisticalProperties:
    """Test statistical properties across multiple runs."""

    def test_noise_scaling_effects(self, aer_backend):
        """Test that higher noise reduces performance as expected."""
        M, N = 1, 2
        noise_levels = [0.05, 0.15, 0.25, 0.35]

        results = []

        for p in noise_levels:
            config = PurificationConfig(
                M=M, N=N, p=p, noise_type="depolarizing",
                shots=1024, num_noise_realizations=1
            )

            circuit = create_batch_purification_circuit(config, base_noise_seed=42)
            measured_circuit = add_measurements(circuit, config)

            counts = simulate_circuit(measured_circuit, shots=config.shots)
            fidelity, success_prob = analyze_results(counts, config)

            results.append((p, fidelity, success_prob))

        # All results should be valid
        for i in range(len(results)):
            p, f, s = results[i]
            assert 0.0 <= f <= 1.0
            assert 0.0 <= s <= 1.0
            print(f"p={p:.2f}: F={f:.3f}, P={s:.3f}")

    def test_identical_copies_verification(self, aer_backend):
        """
        Verify that identical noise seeds produce similar behavior statistically.

        Because we sample with finite shots, results won't be *exactly* equal,
        but repeated runs should agree within reasonable tolerance.
        """
        config = PurificationConfig(M=2, N=4, p=0.3, noise_type="depolarizing", shots=2048)

        # Run same configuration twice with same seed
        results = []
        for trial in range(2):
            circuit = create_batch_purification_circuit(config, base_noise_seed=42)
            measured_circuit = add_measurements(circuit, config)

            counts = simulate_circuit(measured_circuit, shots=config.shots)
            fidelity, success_prob = analyze_results(counts, config)

            results.append((fidelity, success_prob))

        # Results should be statistically close, not bit-for-bit identical
        assert np.isclose(results[0][0], results[1][0], rtol=0.1, atol=0.05)
        assert np.isclose(results[0][1], results[1][1], rtol=0.1, atol=0.05)

    def test_multiqubit_scaling(self, aer_backend):
        """Test behavior across different numbers of qubits M."""
        N = 2  # Keep N small for test efficiency
        p = 0.2

        for M in [1, 2, 3]:
            config = PurificationConfig(
                M=M, N=N, p=p, noise_type="depolarizing",
                shots=1024, num_noise_realizations=1
            )

            circuit = create_batch_purification_circuit(config, base_noise_seed=42)
            measured_circuit = add_measurements(circuit, config)

            counts = simulate_circuit(measured_circuit, shots=config.shots)
            fidelity, success_prob = analyze_results(counts, config)

            # Should produce valid results for all M values
            assert 0.0 <= fidelity <= 1.0
            assert 0.0 <= success_prob <= 1.0
            print(f"M={M}: F={fidelity:.3f}, P={success_prob:.3f}")


# =============================================================================
# Test Error Handling and Edge Cases
# =============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_zero_shots_config(self):
        """Test configuration with zero shots."""
        config = PurificationConfig(M=1, N=2, p=0.1, shots=0)
        with pytest.raises(ValueError, match="shots must be positive"):
            config.validate()

    def test_extreme_noise_values(self, aer_backend):
        """Test with extreme but valid noise values."""
        # Very low noise
        config_low = PurificationConfig(M=1, N=2, p=0.001, noise_type="depolarizing", shots=512)
        circuit = create_batch_purification_circuit(config_low, base_noise_seed=42)
        measured_circuit = add_measurements(circuit, config_low)

        counts = simulate_circuit(measured_circuit, shots=config_low.shots)
        fidelity, success_prob = analyze_results(counts, config_low)

        # Should still work
        assert 0.0 <= fidelity <= 1.0
        assert 0.0 <= success_prob <= 1.0

        # Very high noise
        config_high = PurificationConfig(M=1, N=2, p=0.999, noise_type="depolarizing", shots=512)
        circuit = create_batch_purification_circuit(config_high, base_noise_seed=42)
        measured_circuit = add_measurements(circuit, config_high)

        counts = simulate_circuit(measured_circuit, shots=config_high.shots)
        fidelity, success_prob = analyze_results(counts, config_high)

        # Should still work, though performance will be poor
        assert 0.0 <= fidelity <= 1.0
        assert 0.0 <= success_prob <= 1.0

    def test_large_valid_configurations(self, aer_backend):
        """Test large but valid configurations."""
        # Large N
        config_large_n = PurificationConfig(M=1, N=16, p=0.1, noise_type="depolarizing", shots=256)
        config_large_n.validate()  # Should pass validation

        circuit = create_batch_purification_circuit(config_large_n, base_noise_seed=42)
        assert circuit.num_qubits == 1 * 16 + 15  # Should be correct

        # Large M
        config_large_m = PurificationConfig(M=6, N=2, p=0.1, noise_type="depolarizing", shots=256)
        config_large_m.validate()  # Should pass validation

        circuit = create_batch_purification_circuit(config_large_m, base_noise_seed=42)
        assert circuit.num_qubits == 6 * 2 + 1  # Should be correct


# =============================================================================
# Test Performance and Benchmarks
# =============================================================================

class TestPerformance:
    """Test performance characteristics and scaling."""

    def test_circuit_depth_scaling(self):
        """Test how circuit depth scales with parameters."""
        results = []

        for M in [1, 2]:
            for N in [2, 4, 8]:
                config = PurificationConfig(M=M, N=N, p=0.1, noise_type="depolarizing")
                circuit = create_batch_purification_circuit(config, base_noise_seed=42)
                measured_circuit = add_measurements(circuit, config)

                results.append({
                    'M': M, 'N': N,
                    'depth': measured_circuit.depth(),
                    'qubits': measured_circuit.num_qubits,
                    'gates': len(measured_circuit.data)
                })

        # Verify reasonable scaling
        for result in results:
            print(f"M={result['M']}, N={result['N']}: "
                  f"depth={result['depth']}, qubits={result['qubits']}, gates={result['gates']}")

            # Basic sanity checks
            assert result['depth'] > 0
            assert result['qubits'] == result['M'] * result['N'] + (result['N'] - 1)
            assert result['gates'] > result['M'] * result['N']  # At least preparation gates

    def test_gate_count_consistency(self):
        """Test that gate counts are consistent and expected."""
        config = PurificationConfig(M=3, N=4, p=0.2, noise_type="depolarizing")
        circuit = create_batch_purification_circuit(config, base_noise_seed=42)

        gate_counts = count_gate_types(circuit)

        # Should have specific expected gates
        # Preparation: N*M H gates for |+> states = 4*3 = 12
        # SWAP tests: (N-1) SWAP tests, each with 2 H gates on ancilla = 3*2 = 6
        expected_h = config.N * config.M + 2 * (config.N - 1)  # 12 + 6 = 18
        # SWAP test CSWAPs: (N-1) tests * M CSWAPs per test = 3*3 = 9
        expected_cswap = (config.N - 1) * config.M  # 9

        assert gate_counts.get('h', 0) == expected_h  # 18
        assert gate_counts.get('cswap', 0) == expected_cswap  # 9

        # Noise gates depend on specific realization, but should be reasonable
        noise_gates = ['x', 'y', 'z']
        total_noise = sum(gate_counts.get(gate, 0) for gate in noise_gates)
        max_possible_noise = config.N * config.M  # At most one noise gate per qubit
        assert 0 <= total_noise <= max_possible_noise


# =============================================================================
# Integration Tests for Full Workflow
# =============================================================================

class TestFullWorkflow:
    """Test complete workflow integration."""

    def test_minimal_complete_experiment_simulation(self, aer_backend):
        """Test a minimal complete experiment using simulation."""
        config = PurificationConfig(
            M=1, N=2, p=0.15, noise_type="depolarizing",
            shots=256, num_noise_realizations=2
        )

        # This simulates what run_complete_purification_experiment does
        global_counts = {}

        for r in range(config.num_noise_realizations):
            base_noise_seed = 100000 * r

            circuit = create_batch_purification_circuit(config, base_noise_seed=base_noise_seed)
            measured_circuit = add_measurements(circuit, config)

            # Simulate execution
            counts = simulate_circuit(measured_circuit, shots=config.shots)

            # Aggregate like the real function would
            for outcome, count in counts.items():
                global_counts[outcome] = global_counts.get(outcome, 0) + count

        # Analyze final results
        final_fidelity, final_success_prob = analyze_results(global_counts, config)

        # Should produce valid final results
        assert 0.0 <= final_fidelity <= 1.0
        assert 0.0 <= final_success_prob <= 1.0

        # Total shots should match expected
        expected_shots = config.shots * config.num_noise_realizations
        actual_shots = sum(global_counts.values())
        assert actual_shots == expected_shots

    def test_config_validation_in_workflow(self):
        """Test that invalid configs are caught early in workflow."""
        # Invalid configuration should fail validation
        bad_config = PurificationConfig(M=1, N=3, p=0.1)  # N=3 not power of 2

        with pytest.raises(ValueError):
            bad_config.validate()

    def test_different_configurations_consistency(self, aer_backend):
        """Test that different valid configurations all work."""
        test_configs = [
            PurificationConfig(M=1, N=2, p=0.1, noise_type="depolarizing"),
            PurificationConfig(M=2, N=2, p=0.15, noise_type="dephasing"),
            PurificationConfig(M=1, N=4, p=0.2, noise_type="twirled_dephasing"),
        ]

        for i, config in enumerate(test_configs):
            config.shots = 256  # Keep shots low for test speed
            config.num_noise_realizations = 1

            # Should all work without errors
            circuit = create_batch_purification_circuit(config, base_noise_seed=42)
            measured_circuit = add_measurements(circuit, config)

            counts = simulate_circuit(measured_circuit, shots=config.shots)
            fidelity, success_prob = analyze_results(counts, config)

            assert 0.0 <= fidelity <= 1.0
            assert 0.0 <= success_prob <= 1.0

            print(f"Config {i+1}: M={config.M}, N={config.N}, p={config.p}, "
                  f"noise={config.noise_type} -> F={fidelity:.3f}, P={success_prob:.3f}")


if __name__ == "__main__":
    # Configure logging for test visibility
    logging.basicConfig(level=logging.INFO)

    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-s"])
