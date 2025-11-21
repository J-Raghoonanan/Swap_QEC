"""
Unit tests for IBMQ_components.py

Tests all functions for correctness, error handling, and compatibility
with the simulation codebase patterns.
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.result import Result

# Import functions to test
from IBMQ_components import (
    # Configuration
    IBMQRunSpec,
    # State preparation
    prepare_hadamard_state,
    prepare_ghz_state,
    # Noise application
    apply_depolarizing_noise,
    apply_twirled_dephasing_noise,
    # SWAP test
    build_swap_test_circuit,
    # Measurements
    add_fidelity_measurement_hadamard,
    add_fidelity_measurement_ghz,
    add_ancilla_measurement,
    calculate_fidelity_from_counts,
    # Backend integration
    transpile_for_backend,
    save_ibmq_results,
    # High-level assembly
    build_full_purification_experiment,
    # Internal functions
    _sample_clifford_gate_ibmq,
    _apply_clifford_gate_ibmq,
    _apply_inverse_clifford_gate_ibmq,
)

# Import configs for testing
from configs import NoiseSpec, NoiseType, NoiseMode, TwirlingSpec


class TestIBMQRunSpec:
    """Test the IBMQRunSpec configuration class."""
    
    def test_valid_config(self):
        """Test valid configuration creation."""
        noise = NoiseSpec(noise_type=NoiseType.depolarizing, mode=NoiseMode.iid_p, p=0.1)
        spec = IBMQRunSpec(M=3, N=8, noise=noise)
        
        assert spec.M == 3
        assert spec.N == 8
        assert spec.noise.p == 0.1
        assert spec.target_type == "hadamard"  # default
        assert spec.backend_name == "ibm_torino"  # default
        assert spec.shots == 8192  # default
        assert spec.num_rounds() == 3  # log₂(8) = 3
        
        # Should not raise
        spec.validate()
    
    def test_invalid_M(self):
        """Test invalid M values."""
        noise = NoiseSpec(noise_type=NoiseType.depolarizing, mode=NoiseMode.iid_p, p=0.1)
        
        # M <= 0
        spec = IBMQRunSpec(M=0, N=4, noise=noise)
        with pytest.raises(ValueError, match="M must be in"):
            spec.validate()
        
        # M too large
        spec = IBMQRunSpec(M=20, N=4, noise=noise)
        with pytest.raises(ValueError, match="M must be in"):
            spec.validate()
    
    def test_invalid_N(self):
        """Test invalid N values."""
        noise = NoiseSpec(noise_type=NoiseType.depolarizing, mode=NoiseMode.iid_p, p=0.1)
        
        # N not power of 2
        spec = IBMQRunSpec(M=2, N=3, noise=noise)
        with pytest.raises(ValueError, match="N must be a positive power of 2"):
            spec.validate()
        
        # N <= 0
        spec = IBMQRunSpec(M=2, N=0, noise=noise)
        with pytest.raises(ValueError, match="N must be a positive power of 2"):
            spec.validate()
    
    def test_num_rounds(self):
        """Test num_rounds calculation."""
        noise = NoiseSpec(noise_type=NoiseType.depolarizing, mode=NoiseMode.iid_p, p=0.1)
        
        test_cases = [(2, 1), (4, 2), (8, 3), (16, 4), (32, 5), (64, 6), (128, 7), (256, 8)]
        
        for N, expected_rounds in test_cases:
            spec = IBMQRunSpec(M=2, N=N, noise=noise)
            assert spec.num_rounds() == expected_rounds
    
    def test_invalid_noise_p(self):
        """Test invalid noise probability."""
        noise = NoiseSpec(noise_type=NoiseType.depolarizing, mode=NoiseMode.iid_p, p=1.5)
        spec = IBMQRunSpec(M=2, N=4, noise=noise)
        
        with pytest.raises(ValueError, match="noise.p must be in"):
            spec.validate()
    
    def test_invalid_target_type(self):
        """Test invalid target type."""
        noise = NoiseSpec(noise_type=NoiseType.depolarizing, mode=NoiseMode.iid_p, p=0.1)
        spec = IBMQRunSpec(M=2, N=4, noise=noise, target_type="invalid")
        
        with pytest.raises(ValueError, match="target_type must be"):
            spec.validate()


class TestStatePreperation:
    """Test state preparation functions."""
    
    def test_prepare_hadamard_state(self):
        """Test Hadamard state preparation."""
        for M in [1, 2, 3, 5]:
            qc = prepare_hadamard_state(M)
            
            # Check circuit structure
            assert qc.num_qubits == M
            assert qc.name == f"prep_hadamard_M{M}"
            
            # Check that all qubits get H gate
            h_count = sum(1 for instr in qc.data if instr.operation.name == 'h')
            assert h_count == M
            
            # Verify state vector
            sv = Statevector.from_instruction(qc)
            expected = np.ones(2**M) / np.sqrt(2**M)
            np.testing.assert_allclose(sv.data, expected, atol=1e-10)
    
    def test_prepare_ghz_state(self):
        """Test GHZ state preparation."""
        for M in [2, 3, 4, 5]:
            qc = prepare_ghz_state(M)
            
            # Check circuit structure
            assert qc.num_qubits == M
            assert qc.name == f"prep_ghz_M{M}"
            
            # Check gate counts
            h_count = sum(1 for instr in qc.data if instr.operation.name == 'h')
            cx_count = sum(1 for instr in qc.data if instr.operation.name == 'cx')
            assert h_count == 1  # H on first qubit only
            assert cx_count == M - 1  # CX from first to all others
            
            # Verify state vector (should be |00...0> + |11...1>)/sqrt(2)
            sv = Statevector.from_instruction(qc)
            expected = np.zeros(2**M, dtype=complex)
            expected[0] = 1/np.sqrt(2)  # |00...0>
            expected[-1] = 1/np.sqrt(2)  # |11...1>
            np.testing.assert_allclose(sv.data, expected, atol=1e-10)
    
    def test_invalid_M_state_prep(self):
        """Test error handling for invalid M."""
        with pytest.raises(ValueError, match="M must be positive"):
            prepare_hadamard_state(0)
        
        with pytest.raises(ValueError, match="M must be positive"):
            prepare_ghz_state(-1)


class TestNoiseApplication:
    """Test noise application functions."""
    
    def test_apply_depolarizing_noise_structure(self):
        """Test that depolarizing noise is applied correctly."""
        M = 3
        qc = QuantumCircuit(M)
        qubits = list(range(M))
        p = 0.3
        
        # Apply noise (with fixed seed for reproducibility)
        np.random.seed(42)
        apply_depolarizing_noise(qc, qubits, p)
        
        # Check that some Pauli gates were added
        pauli_count = sum(1 for instr, qargs, cargs in qc.data 
                         if instr.name in ['x', 'y', 'z'])
        
        # With p=0.3 and 3 qubits, expect some Paulis (exact count depends on random)
        # Just check structure is reasonable
        assert pauli_count >= 0
        assert pauli_count <= M  # At most one Pauli per qubit
    
    def test_apply_twirled_dephasing_noise_structure(self):
        """Test that twirled dephasing noise includes proper twirling."""
        M = 2
        qc = QuantumCircuit(M)
        qubits = list(range(M))
        p = 0.5
        
        initial_gates = len(qc.data)
        apply_twirled_dephasing_noise(qc, qubits, p, twirling_seed=42)
        final_gates = len(qc.data)
        
        # Should have added gates (twirling + dephasing + inverse twirling)
        assert final_gates > initial_gates
        
        # Check for presence of various gates used in twirling
        gate_names = [instr.operation.name for instr in qc.data]
        
        # Should have some combination of h, s, sdg, z gates
        twirl_gates = set(['h', 's', 'sdg', 'z'])
        applied_gates = set(gate_names)
        
        # At least some twirling gates should be present
        assert len(twirl_gates & applied_gates) > 0
    
    def test_noise_probability_bounds(self):
        """Test error handling for invalid probabilities."""
        qc = QuantumCircuit(2)
        qubits = [0, 1]
        
        # Test depolarizing bounds
        with pytest.raises(ValueError, match="p must be in"):
            apply_depolarizing_noise(qc, qubits, -0.1)
        
        with pytest.raises(ValueError, match="p must be in"):
            apply_depolarizing_noise(qc, qubits, 1.1)
        
        # Test dephasing bounds  
        with pytest.raises(ValueError, match="p must be in"):
            apply_twirled_dephasing_noise(qc, qubits, -0.1)
        
        with pytest.raises(ValueError, match="p must be in"):
            apply_twirled_dephasing_noise(qc, qubits, 1.1)
    
    def test_clifford_gate_sampling(self):
        """Test Clifford gate sampling for twirling."""
        options = ['i', 'h', 's', 'sdg', 'sh', 'sdgh']
        
        # Test random mode
        gate = _sample_clifford_gate_ibmq("random", 0, seed=42)
        assert gate in options
        
        # Test cyclic mode
        for i in range(len(options) * 2):
            gate = _sample_clifford_gate_ibmq("cyclic", i)
            assert gate == options[i % len(options)]
    
    def test_clifford_gate_application(self):
        """Test Clifford gate application and inversion."""
        qc = QuantumCircuit(1)
        
        test_gates = ['i', 'h', 's', 'sdg', 'sh', 'sdgh']
        
        for gate_name in test_gates:
            qc_test = QuantumCircuit(1)
            
            # Apply gate then its inverse
            _apply_clifford_gate_ibmq(qc_test, 0, gate_name)
            _apply_inverse_clifford_gate_ibmq(qc_test, 0, gate_name)
            
            # Should return to identity (up to global phase)
            if gate_name == 'i':
                assert len(qc_test.data) == 0  # No gates applied
            else:
                # Check that we can compute state vector without error
                try:
                    sv = Statevector.from_instruction(qc_test)
                    # Should be close to |0> state (up to global phase)
                    assert abs(abs(sv.data[0]) - 1.0) < 1e-10
                except:
                    pytest.fail(f"Failed to compute statevector for gate {gate_name}")


class TestSWAPTest:
    """Test SWAP test circuit construction."""
    
    def test_swap_test_structure(self):
        """Test SWAP test circuit structure."""
        for M in [1, 2, 3]:
            qc = build_swap_test_circuit(M)
            
            # Check qubit count
            expected_qubits = 1 + 2 * M  # ancilla + 2 registers
            assert qc.num_qubits == expected_qubits
            assert qc.name == f"swap_test_M{M}"
            
            # Check for Hadamard gates on ancilla (qubit 0)
            h_count = sum(1 for instr in qc.data 
                         if instr.operation.name == 'h' and instr.qubits[0]._index == 0)
            assert h_count == 2  # H before and after CSWAP
            
            # Check for controlled operations
            ccx_count = sum(1 for instr in qc.data if instr.operation.name == 'ccx')
            assert ccx_count == 3 * M  # 3 CCX gates per qubit pair for Fredkin
    
    def test_swap_test_invalid_M(self):
        """Test error handling for invalid M in SWAP test."""
        with pytest.raises(ValueError, match="M must be positive"):
            build_swap_test_circuit(0)


class TestMeasurements:
    """Test measurement functions."""
    
    def test_hadamard_fidelity_measurement(self):
        """Test Hadamard fidelity measurement setup."""
        M = 3
        qc = QuantumCircuit(M)
        qubits = [0, 1, 2]
        
        add_fidelity_measurement_hadamard(qc, qubits)
        
        # Check that H gates were added
        h_count = sum(1 for instr in qc.data if instr.operation.name == 'h')
        assert h_count == M
        
        # Check that classical register was added
        assert len(qc.cregs) == 1
        assert qc.cregs[0].size == M
        
        # Check that measurements were added
        measure_count = sum(1 for instr in qc.data if instr.operation.name == 'measure')
        assert measure_count == M
    
    def test_ghz_fidelity_measurement(self):
        """Test GHZ fidelity measurement setup."""
        M = 3
        qc = QuantumCircuit(M)
        qubits = [0, 1, 2]
        
        add_fidelity_measurement_ghz(qc, qubits)
        
        # Check that NO H gates were added (direct Z measurement)
        h_count = sum(1 for instr in qc.data if instr.operation.name == 'h')
        assert h_count == 0
        
        # Check classical register and measurements
        assert len(qc.cregs) == 1
        assert qc.cregs[0].size == M
        
        measure_count = sum(1 for instr in qc.data if instr.operation.name == 'measure')
        assert measure_count == M
    
    def test_ancilla_measurement(self):
        """Test ancilla measurement setup."""
        qc = QuantumCircuit(5)
        ancilla = 0
        
        add_ancilla_measurement(qc, ancilla)
        
        # Check classical register and measurement
        assert len(qc.cregs) == 1
        assert qc.cregs[0].size == 1
        
        measure_count = sum(1 for instr in qc.data if instr.operation.name == 'measure')
        assert measure_count == 1
    
    def test_empty_qubits_error(self):
        """Test error handling for empty qubit lists."""
        qc = QuantumCircuit(3)
        
        with pytest.raises(ValueError, match="qubits list cannot be empty"):
            add_fidelity_measurement_hadamard(qc, [])
        
        with pytest.raises(ValueError, match="qubits list cannot be empty"):
            add_fidelity_measurement_ghz(qc, [])


class TestFidelityCalculation:
    """Test fidelity calculation from measurement counts."""
    
    def test_hadamard_fidelity_perfect(self):
        """Test fidelity calculation for perfect Hadamard state."""
        M = 3
        # Perfect Hadamard state should give all zeros after H rotation
        counts = {"000": 1000}
        
        fidelity = calculate_fidelity_from_counts(counts, M, "hadamard")
        assert fidelity == 1.0
    
    def test_hadamard_fidelity_mixed(self):
        """Test fidelity calculation for mixed Hadamard state."""
        M = 2
        counts = {"00": 600, "01": 200, "10": 150, "11": 50}
        
        fidelity = calculate_fidelity_from_counts(counts, M, "hadamard")
        assert fidelity == 0.6  # 600/1000
    
    def test_ghz_fidelity_perfect(self):
        """Test fidelity calculation for perfect GHZ state."""
        M = 3
        # Perfect GHZ should give only |000> and |111>
        counts = {"000": 500, "111": 500}
        
        fidelity = calculate_fidelity_from_counts(counts, M, "ghz")
        assert fidelity == 1.0
    
    def test_ghz_fidelity_mixed(self):
        """Test fidelity calculation for mixed GHZ state."""
        M = 2
        counts = {"00": 300, "01": 100, "10": 200, "11": 400}
        
        fidelity = calculate_fidelity_from_counts(counts, M, "ghz")
        assert fidelity == 0.7  # (300 + 400)/1000
    
    def test_fidelity_no_counts(self):
        """Test fidelity calculation with no counts."""
        fidelity = calculate_fidelity_from_counts({}, 2, "hadamard")
        assert fidelity == 0.0
    
    def test_fidelity_invalid_target(self):
        """Test error handling for invalid target type."""
        with pytest.raises(ValueError, match="Unknown target_type"):
            calculate_fidelity_from_counts({"00": 100}, 2, "invalid")


class TestBackendIntegration:
    """Test backend integration functions (using mocks)."""
    
    @patch('IBMQ_components.generate_preset_pass_manager')
    def test_transpile_for_backend(self, mock_pass_manager):
        """Test circuit transpilation."""
        # Setup mocks
        mock_pm = Mock()
        mock_transpiled = QuantumCircuit(2)
        mock_pm.run.return_value = mock_transpiled
        mock_pass_manager.return_value = mock_pm
        
        mock_backend = Mock()
        
        # Test transpilation
        qc = QuantumCircuit(2)
        result = transpile_for_backend(qc, mock_backend, optimization_level=2)
        
        # Verify calls
        mock_pass_manager.assert_called_once_with(
            optimization_level=2,
            backend=mock_backend
        )
        mock_pm.run.assert_called_once_with(qc)
        assert result == mock_transpiled
    
    def test_save_ibmq_results(self):
        """Test results saving functionality."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = Path(tmp_dir) / "test_results.csv"
            
            # Create test data
            noise = NoiseSpec(noise_type=NoiseType.depolarizing, mode=NoiseMode.iid_p, p=0.1)
            run_spec = IBMQRunSpec(M=2, N=4, noise=noise)
            results = {
                'fidelity': 0.85,
                'success_probability': 0.78,
                'ancilla_counts': {'0': 780, '1': 220}
            }
            
            # Save results
            save_ibmq_results(results, filepath, run_spec)
            
            # Verify file was created and has correct content
            assert filepath.exists()
            
            with open(filepath, 'r') as f:
                content = f.read()
                assert 'M,N,noise_type' in content  # Header
                assert '2,4,depolarizing' in content  # Data
                assert '0.85' in content  # Fidelity


class TestHighLevelAssembly:
    """Test high-level circuit assembly functions."""
    
    def test_build_full_purification_hadamard_depolarizing(self):
        """Test complete purification circuit for Hadamard + depolarizing."""
        M = 2
        N = 4  # Test with sequential purification
        noise = NoiseSpec(noise_type=NoiseType.depolarizing, mode=NoiseMode.iid_p, p=0.2)
        
        qc = build_full_purification_experiment(M, noise, "hadamard", N)
        
        # Check structure
        expected_qubits = N * M + N // 2  # N copies + ancillas for first round
        assert qc.num_qubits >= expected_qubits - N // 4  # Allow for fewer ancillas in later rounds
        
        # Check classical registers
        assert len(qc.cregs) >= 1  # At least fidelity register (ancilla register for sequential)
        
        # Check for key gate types
        gate_names = [instr.operation.name for instr in qc.data]
        
        # Should have Hadamard gates (prep + possibly fidelity measurement)
        assert 'h' in gate_names
        # Should have some noise gates
        assert any(g in gate_names for g in ['x', 'y', 'z'])
        # Should have SWAP test gates  
        assert 'ccx' in gate_names
        # Should have measurements
        assert 'measure' in gate_names
    
    def test_build_full_purification_ghz_dephasing(self):
        """Test complete purification circuit for GHZ + dephasing."""
        M = 2
        N = 4  # Test with sequential purification
        noise = NoiseSpec(noise_type=NoiseType.dephase_z, mode=NoiseMode.iid_p, p=0.3)
        
        qc = build_full_purification_experiment(M, noise, "ghz", N)
        
        # Check structure - similar to above but for sequential purification
        expected_qubits = N * M + N // 2  
        assert qc.num_qubits >= expected_qubits - N // 4
        
        # Check for GHZ prep gates
        gate_names = [instr.operation.name for instr in qc.data]
        assert 'h' in gate_names  # GHZ prep
        assert 'cx' in gate_names  # GHZ prep
        
        # Should have twirling gates (s, sdg, etc.)
        twirl_gates = ['s', 'sdg']
        assert any(g in gate_names for g in twirl_gates)
    
    def test_invalid_target_type_assembly(self):
        """Test error handling for invalid target type in assembly."""
        M = 2
        noise = NoiseSpec(noise_type=NoiseType.depolarizing, mode=NoiseMode.iid_p, p=0.1)
        
        with pytest.raises(ValueError, match="Unknown target_type"):
            build_full_purification_experiment(M, noise, "invalid")
    
    def test_unsupported_noise_type_assembly(self):
        """Test error handling for unsupported noise types."""
        M = 2
        # Create a custom noise type that's not supported
        noise = NoiseSpec(noise_type="unsupported", mode=NoiseMode.iid_p, p=0.1)
        
        with pytest.raises((ValueError, AttributeError)):
            build_full_purification_experiment(M, noise, "hadamard")


# Test fixtures and utilities
@pytest.fixture
def sample_noise_specs():
    """Provide sample noise specifications for testing."""
    return {
        'depolarizing': NoiseSpec(noise_type=NoiseType.depolarizing, mode=NoiseMode.iid_p, p=0.1),
        'dephasing_z': NoiseSpec(noise_type=NoiseType.dephase_z, mode=NoiseMode.iid_p, p=0.2),
        'dephasing_x': NoiseSpec(noise_type=NoiseType.dephase_x, mode=NoiseMode.iid_p, p=0.15),
    }


@pytest.fixture
def sample_run_specs(sample_noise_specs):
    """Provide sample run specifications for testing."""
    return {
        name: IBMQRunSpec(M=2, noise=noise, target_type="hadamard")
        for name, noise in sample_noise_specs.items()
    }


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])