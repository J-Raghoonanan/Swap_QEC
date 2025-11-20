"""
Comprehensive unit tests for IBMQ_components.py

This test suite validates all functions in the IBMQ components library to ensure
correct behavior before running experiments on quantum hardware.

Run with: python -m pytest test_ibmq_components.py -v
Or: python test_ibmq_components.py
"""

import unittest
import numpy as np
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json


# Add the source directory to path for imports
sys.path.append('src/simulation')
from IBMQ_components import *
COMPONENTS_AVAILABLE = True

# Import Qiskit for testing if available
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    QISKIT_TEST_AVAILABLE = True
except ImportError:
    QISKIT_TEST_AVAILABLE = False


class TestStatePreparation(unittest.TestCase):
    """Test state preparation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("IBMQ components not available")
    
    def test_create_hadamard_state_circuit_basic(self):
        """Test basic Hadamard state creation."""
        M = 2
        qc, sv = create_hadamard_state_circuit(M)
        
        # Check circuit structure
        self.assertEqual(qc.num_qubits, M)
        self.assertIn('hadamard', qc.name.lower())
        
        # Check that we have M Hadamard gates
        h_gates = sum(1 for instruction in qc.data if instruction.operation.name == 'h')
        self.assertEqual(h_gates, M)
        
        # Check statevector properties
        self.assertEqual(sv.dim, 2**M)
        self.assertAlmostEqual(float(np.abs(sv.data).sum()), 2**(M/2), places=10)
    
    def test_create_hadamard_state_circuit_edge_cases(self):
        """Test Hadamard state creation edge cases."""
        # Single qubit
        qc1, sv1 = create_hadamard_state_circuit(1)
        self.assertEqual(qc1.num_qubits, 1)
        expected_sv = np.array([1, 1]) / np.sqrt(2)
        np.testing.assert_array_almost_equal(sv1.data, expected_sv)
        
        # Larger system
        qc3, sv3 = create_hadamard_state_circuit(3)
        self.assertEqual(qc3.num_qubits, 3)
        self.assertEqual(sv3.dim, 8)
        
        # Check all amplitudes are equal for |+>^⊗M
        expected_amplitude = 1.0 / (2**(3/2))
        for amplitude in sv3.data:
            self.assertAlmostEqual(abs(amplitude), expected_amplitude, places=10)
    
    def test_create_ghz_state_circuit_basic(self):
        """Test basic GHZ state creation."""
        M = 3
        qc, sv = create_ghz_state_circuit(M)
        
        # Check circuit structure
        self.assertEqual(qc.num_qubits, M)
        self.assertIn('ghz', qc.name.lower())
        
        # Check gates: 1 H + (M-1) CNOTs
        h_gates = sum(1 for instruction in qc.data if instruction.operation.name == 'h')
        cx_gates = sum(1 for instruction in qc.data if instruction.operation.name == 'cx')
        self.assertEqual(h_gates, 1)
        self.assertEqual(cx_gates, M-1)
        
        # Check GHZ statevector properties
        sv_data = sv.data
        self.assertEqual(sv.dim, 2**M)
        
        # GHZ state: (|000> + |111>)/√2
        expected_nonzero_indices = [0, 2**M - 1]  # |000...> and |111...>
        for i, amplitude in enumerate(sv_data):
            if i in expected_nonzero_indices:
                self.assertAlmostEqual(abs(amplitude), 1/np.sqrt(2), places=10)
            else:
                self.assertAlmostEqual(abs(amplitude), 0.0, places=10)
    
    def test_create_ghz_state_invalid_input(self):
        """Test GHZ state creation with invalid inputs."""
        with self.assertRaises(ValueError):
            create_ghz_state_circuit(0)  # M must be >= 1
    
    def test_create_target_state_circuit_all_types(self):
        """Test target state creation for all supported types."""
        M = 2
        
        # Test hadamard type
        qc_h, sv_h = create_target_state_circuit(M, 'hadamard')
        self.assertEqual(qc_h.num_qubits, M)
        self.assertIn('hadamard', qc_h.name.lower())
        
        # Test GHZ type
        qc_ghz, sv_ghz = create_target_state_circuit(M, 'ghz')
        self.assertEqual(qc_ghz.num_qubits, M)
        self.assertIn('ghz', qc_ghz.name.lower())
        
        # Test random type (should default to hadamard with warning)
        with patch('IBMQ_components.logger') as mock_logger:
            qc_rand, sv_rand = create_target_state_circuit(M, 'random', seed=42)
            mock_logger.warning.assert_called_once()
            self.assertEqual(qc_rand.num_qubits, M)
    
    def test_create_target_state_circuit_invalid_type(self):
        """Test target state creation with invalid state type."""
        with self.assertRaises(ValueError):
            create_target_state_circuit(2, 'invalid_state_type')


class TestNoiseApplication(unittest.TestCase):
    """Test noise application functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("IBMQ components not available")
        
        # Create a simple test circuit
        self.test_qc = QuantumCircuit(2)
        self.test_qc.h(0)
        self.test_qc.h(1)
    
    def test_apply_depolarizing_noise_no_noise(self):
        """Test depolarizing noise with p=0 (no noise)."""
        qc = self.test_qc.copy()
        original_length = len(qc)
        
        # With p=0, no noise should be added
        apply_depolarizing_noise_stochastic(qc, 0, 0.0, 42)
        self.assertEqual(len(qc), original_length)
    
    def test_apply_depolarizing_noise_with_seed(self):
        """Test depolarizing noise with fixed seed for reproducibility."""
        qc1 = self.test_qc.copy()
        qc2 = self.test_qc.copy()
        
        # Same seed should give same noise
        apply_depolarizing_noise_stochastic(qc1, 0, 0.5, 42)
        apply_depolarizing_noise_stochastic(qc2, 0, 0.5, 42)
        
        # Check that circuits are identical
        self.assertEqual(len(qc1), len(qc2))
        for i, (instr1, instr2) in enumerate(zip(qc1.data, qc2.data)):
            self.assertEqual(instr1.operation.name, instr2.operation.name)
    
    def test_apply_z_dephasing_noise(self):
        """Test Z-dephasing noise application."""
        qc = self.test_qc.copy()
        original_length = len(qc)
        
        # With p=1, should always add Z gate
        apply_z_dephasing_noise(qc, 0, 1.0, 42)
        
        # Should have one more gate
        self.assertEqual(len(qc), original_length + 1)
        # Last added gate should be Z
        self.assertEqual(qc.data[-1].operation.name, 'z')
        
        # With p=0, no noise should be added
        qc_no_noise = self.test_qc.copy()
        apply_z_dephasing_noise(qc_no_noise, 0, 0.0, 42)
        self.assertEqual(len(qc_no_noise), original_length)
    
    def test_apply_x_dephasing_noise(self):
        """Test X-dephasing noise application."""
        qc = self.test_qc.copy()
        original_length = len(qc)
        
        # With p=1, should always add X gate
        apply_x_dephasing_noise(qc, 0, 1.0, 42)
        
        # Should have one more gate
        self.assertEqual(len(qc), original_length + 1)
        # Last added gate should be X
        self.assertEqual(qc.data[-1].operation.name, 'x')
    
    def test_clifford_gates_application(self):
        """Test all Clifford gate applications."""
        test_gates = ['i', 'h', 's', 'sdg', 'sh', 'sdgh']
        
        for gate_name in test_gates:
            qc = QuantumCircuit(1)
            original_length = len(qc)
            
            apply_single_qubit_clifford(qc, 0, gate_name)
            
            if gate_name == 'i':
                # Identity should not add gates
                self.assertEqual(len(qc), original_length)
            else:
                # Other gates should add at least one gate
                self.assertGreaterEqual(len(qc), original_length + 1)
    
    def test_inverse_clifford_gates(self):
        """Test inverse Clifford gate applications."""
        test_gates = ['i', 'h', 's', 'sdg', 'sh', 'sdgh']
        
        for gate_name in test_gates:
            qc = QuantumCircuit(1)
            
            # Apply gate then its inverse
            apply_single_qubit_clifford(qc, 0, gate_name)
            apply_inverse_clifford(qc, 0, gate_name)
            
            # For testable cases, check if we get back to identity
            # (This is a simplified test; full verification would need statevector comparison)
            if gate_name == 'h':
                # H * H = I, so we should have 2 H gates
                h_count = sum(1 for instr in qc.data if instr.operation.name == 'h')
                self.assertEqual(h_count, 2)
    
    def test_invalid_clifford_gates(self):
        """Test error handling for invalid Clifford gates."""
        qc = QuantumCircuit(1)
        
        with self.assertRaises(ValueError):
            apply_single_qubit_clifford(qc, 0, 'invalid_gate')
        
        with self.assertRaises(ValueError):
            apply_inverse_clifford(qc, 0, 'invalid_gate')
    
    def test_apply_noise_to_circuit_basic(self):
        """Test applying noise to circuit without twirling."""
        qc = self.test_qc.copy()
        original_length = len(qc)
        
        # Test depolarizing noise
        noisy_qc = apply_noise_to_circuit(qc, 'depolarizing', 0.1, apply_twirling=False)
        
        self.assertIsInstance(noisy_qc, QuantumCircuit)
        self.assertEqual(noisy_qc.num_qubits, qc.num_qubits)
        self.assertIn('noisy_depolarizing', noisy_qc.name)
    
    def test_apply_noise_to_circuit_with_twirling(self):
        """Test applying noise with Clifford twirling."""
        qc = self.test_qc.copy()
        
        # Test with dephasing noise (should trigger twirling)
        noisy_qc = apply_noise_to_circuit(qc, 'dephase_z', 0.1, 
                                         apply_twirling=True, twirl_seed=42)
        
        self.assertIsInstance(noisy_qc, QuantumCircuit)
        # Should have more gates due to twirling
        self.assertGreater(len(noisy_qc), len(qc))
    
    def test_apply_noise_invalid_type(self):
        """Test error handling for invalid noise types."""
        qc = self.test_qc.copy()
        
        with self.assertRaises(ValueError):
            apply_noise_to_circuit(qc, 'invalid_noise_type', 0.1)


class TestSWAPTestCircuits(unittest.TestCase):
    """Test SWAP test circuit construction."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("IBMQ components not available")
    
    def test_build_swap_test_circuit_basic(self):
        """Test basic SWAP test circuit construction."""
        M = 2
        qc = build_swap_test_circuit(M, measure_ancilla=True)
        
        # Check qubit count: 1 ancilla + 2M data qubits
        self.assertEqual(qc.num_qubits, 1 + 2*M)
        
        # Check classical bits for measurement
        self.assertEqual(qc.num_clbits, 1)
        
        # Check circuit name
        self.assertIn('swap_test', qc.name.lower())
        self.assertIn(f'M{M}', qc.name)
        
        # Check for required gates
        gate_names = [instr.operation.name for instr in qc.data]
        self.assertIn('h', gate_names)  # Should have Hadamard gates
        self.assertIn('cswap', gate_names)  # Should have CSWAP gates
        self.assertIn('measure', gate_names)  # Should have measurement
        
        # Count CSWAP gates (should be M)
        cswap_count = sum(1 for name in gate_names if name == 'cswap')
        self.assertEqual(cswap_count, M)
    
    def test_build_swap_test_circuit_no_measurement(self):
        """Test SWAP test circuit without measurement."""
        M = 1
        qc = build_swap_test_circuit(M, measure_ancilla=False)
        
        # Should have no classical bits
        self.assertEqual(qc.num_clbits, 0)
        
        # Should not have measurement gates
        gate_names = [instr.operation.name for instr in qc.data]
        self.assertNotIn('measure', gate_names)
    
    def test_create_swap_purification_circuit(self):
        """Test complete SWAP purification circuit creation."""
        M = 2
        
        # Create test state preparation circuits
        prep_A = QuantumCircuit(M)
        prep_A.h(0)
        prep_B = QuantumCircuit(M) 
        prep_B.h(1)
        
        qc = create_swap_purification_circuit(prep_A, prep_B, measure_ancilla=True)
        
        # Check structure
        self.assertEqual(qc.num_qubits, 1 + 2*M)
        self.assertEqual(qc.num_clbits, 1)
        self.assertIn('swap_purification', qc.name.lower())
        
        # Should contain preparation gates plus SWAP test
        gate_names = [instr.operation.name for instr in qc.data]
        self.assertIn('h', gate_names)
        self.assertIn('cswap', gate_names)
        self.assertIn('measure', gate_names)
    
    def test_create_swap_purification_circuit_mismatched_qubits(self):
        """Test error handling for mismatched qubit numbers."""
        prep_A = QuantumCircuit(2)
        prep_B = QuantumCircuit(3)  # Different number of qubits
        
        with self.assertRaises(AssertionError):
            create_swap_purification_circuit(prep_A, prep_B)
    
    def test_create_fidelity_measurement_circuit(self):
        """Test fidelity measurement circuit creation."""
        M = 1
        target_prep = QuantumCircuit(M)
        target_prep.h(0)
        
        noisy_prep = QuantumCircuit(M)
        noisy_prep.h(0)
        noisy_prep.z(0)  # Add some noise
        
        qc = create_fidelity_measurement_circuit(target_prep, noisy_prep)
        
        # Should be equivalent to SWAP purification circuit
        self.assertEqual(qc.num_qubits, 1 + 2*M)
        self.assertEqual(qc.num_clbits, 1)


class TestAmplitudeAmplification(unittest.TestCase):
    """Test amplitude amplification helper functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("IBMQ components not available")
    
    def test_calculate_grover_iterations_basic(self):
        """Test basic Grover iteration calculation."""
        # Test case where no amplification needed
        k = calculate_grover_iterations(success_prob=0.99, target_success=0.95)
        self.assertEqual(k, 0)
        
        # Test case where amplification needed
        k = calculate_grover_iterations(success_prob=0.25, target_success=0.9)
        self.assertGreater(k, 0)
        self.assertLessEqual(k, 32)  # Should respect max_iters
    
    def test_calculate_grover_iterations_edge_cases(self):
        """Test Grover iteration calculation edge cases."""
        # Zero success probability
        k = calculate_grover_iterations(success_prob=0.0, target_success=0.9)
        self.assertEqual(k, 0)
        
        # Perfect success probability  
        k = calculate_grover_iterations(success_prob=1.0, target_success=0.9)
        self.assertEqual(k, 0)
        
        # Very low success probability
        k = calculate_grover_iterations(success_prob=0.001, target_success=0.9, max_iters=10)
        self.assertEqual(k, 10)  # Should hit max_iters limit
    
    def test_build_amplitude_amplification_circuit(self):
        """Test amplitude amplification circuit construction."""
        M = 1
        base_circuit = build_swap_test_circuit(M)
        
        # Test with 0 iterations (should return copy)
        aa_circuit = build_amplitude_amplification_circuit(base_circuit, 0)
        self.assertEqual(aa_circuit.num_qubits, base_circuit.num_qubits)
        self.assertEqual(aa_circuit.name, base_circuit.name)
        
        # Test with positive iterations (currently returns copy with warning)
        with patch('IBMQ_components.logger') as mock_logger:
            aa_circuit = build_amplitude_amplification_circuit(base_circuit, 3)
            mock_logger.warning.assert_called_once()
    
    def test_create_repeated_swap_circuit(self):
        """Test repeated SWAP circuit creation."""
        M = 1
        num_repeats = 3
        
        prep_A = QuantumCircuit(M)
        prep_A.h(0)
        prep_B = QuantumCircuit(M)
        prep_B.x(0)
        
        qc = create_repeated_swap_circuit(prep_A, prep_B, num_repeats)
        
        # Check structure
        expected_qubits = num_repeats * (1 + 2*M)  # Each repeat: 1 ancilla + 2M data
        self.assertEqual(qc.num_qubits, expected_qubits)
        self.assertEqual(qc.num_clbits, num_repeats)  # One measurement per repeat
        
        # Check name
        self.assertIn('repeated_swap', qc.name.lower())
        self.assertIn(f'x{num_repeats}', qc.name)


class TestBackendManagement(unittest.TestCase):
    """Test backend management functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("IBMQ components not available")
    
    def test_setup_quantum_backend_aer_simulator(self):
        """Test setting up local AER simulator."""
        service, backend = setup_quantum_backend('aer_simulator')
        
        if QISKIT_TEST_AVAILABLE:
            self.assertIsNone(service)
            self.assertIsInstance(backend, AerSimulator)
        else:
            # If Qiskit not available, should return None
            self.assertIsNone(service)
            self.assertIsNone(backend)
    
    @patch('IBMQ_components.QiskitRuntimeService')
    def test_setup_quantum_backend_ibm_hardware(self, mock_service_class):
        """Test setting up IBM hardware backend."""
        # Mock the service and backend
        mock_service = Mock()
        mock_backend = Mock()
        mock_service.backend.return_value = mock_backend
        mock_service_class.return_value = mock_service
        
        service, backend = setup_quantum_backend('ibm_brisbane')
        
        self.assertEqual(service, mock_service)
        self.assertEqual(backend, mock_backend)
        mock_service.backend.assert_called_once_with('ibm_brisbane')
    
    @patch('IBMQ_components.QiskitRuntimeService')
    def test_setup_quantum_backend_connection_error(self, mock_service_class):
        """Test handling of connection errors."""
        # Mock connection failure
        mock_service_class.side_effect = Exception("Connection failed")
        
        with self.assertRaises(Exception):
            setup_quantum_backend('ibm_brisbane')
    
    def test_execute_circuits_mock_mode(self):
        """Test circuit execution in mock mode."""
        # Create test circuit
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        
        # Execute with None backend (should trigger mock mode)
        counts_list = execute_circuits_with_backend([qc], None, None, shots=1000)
        
        self.assertEqual(len(counts_list), 1)
        counts = counts_list[0]
        self.assertIsInstance(counts, dict)
        self.assertIn('0', counts)
        self.assertIn('1', counts)
        
        # Check total counts approximately equal to shots
        total_counts = sum(counts.values())
        self.assertGreaterEqual(total_counts, 900)  # Allow some tolerance


class TestMeasurementAnalysis(unittest.TestCase):
    """Test measurement and analysis functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("IBMQ components not available")
    
    def test_analyze_swap_test_results_single_test(self):
        """Test analysis of single SWAP test results."""
        # Test successful case
        counts = {'0': 700, '1': 300}
        success, prob, analysis = analyze_swap_test_results(counts, num_repeats=1)
        
        self.assertTrue(success)
        self.assertAlmostEqual(prob, 0.7, places=3)
        self.assertEqual(analysis['total_shots'], 1000)
        self.assertEqual(analysis['success_outcomes'], 700)
        
        # Test failure case
        counts_fail = {'1': 1000}  # No successful outcomes
        success, prob, analysis = analyze_swap_test_results(counts_fail, num_repeats=1)
        
        self.assertFalse(success)
        self.assertEqual(prob, 0.0)
    
    def test_analyze_swap_test_results_multiple_tests(self):
        """Test analysis of multiple SWAP test results."""
        # Test with multiple ancilla measurements
        counts = {'00': 100, '01': 200, '10': 300, '11': 400}  # 2 ancillas
        success, prob, analysis = analyze_swap_test_results(counts, num_repeats=2)
        
        # Any outcome with at least one '0' should count as success
        expected_success = 100 + 200 + 300  # '00', '01', '10'
        self.assertEqual(analysis['success_outcomes'], expected_success)
        self.assertAlmostEqual(prob, 0.6, places=3)
    
    def test_analyze_swap_test_results_empty_data(self):
        """Test analysis with empty measurement data."""
        success, prob, analysis = analyze_swap_test_results({}, num_repeats=1)
        
        self.assertFalse(success)
        self.assertEqual(prob, 0.0)
        self.assertIn('error', analysis)
    
    def test_measure_swap_success_probability(self):
        """Test SWAP success probability measurement."""
        # Create a simple SWAP test circuit
        qc = build_swap_test_circuit(1, measure_ancilla=True)
        
        # Mock the execution
        with patch('IBMQ_components.execute_circuits_with_backend') as mock_execute:
            mock_execute.return_value = [{'0': 600, '1': 400}]
            
            prob, counts = measure_swap_success_probability(qc, None, None, shots=1000)
            
            self.assertAlmostEqual(prob, 0.6, places=3)
            self.assertEqual(counts, {'0': 600, '1': 400})
    
    def test_measure_fidelity_with_swap_test(self):
        """Test fidelity measurement via SWAP test."""
        # Create test preparation circuits
        target_prep = QuantumCircuit(1)
        target_prep.h(0)
        
        noisy_prep = QuantumCircuit(1)
        noisy_prep.h(0)
        
        # Mock the SWAP test execution
        with patch('IBMQ_components.measure_swap_success_probability') as mock_measure:
            # P_success = 0.75 should give fidelity = 2*0.75 - 1 = 0.5
            mock_measure.return_value = (0.75, {'0': 750, '1': 250})
            
            fidelity, counts = measure_fidelity_with_swap_test(
                target_prep, noisy_prep, None, None, shots=1000
            )
            
            expected_fidelity = 2 * 0.75 - 1
            self.assertAlmostEqual(fidelity, expected_fidelity, places=3)
    
    def test_measure_fidelity_boundary_cases(self):
        """Test fidelity measurement boundary cases."""
        target_prep = QuantumCircuit(1)
        noisy_prep = QuantumCircuit(1)
        
        with patch('IBMQ_components.measure_swap_success_probability') as mock_measure:
            # Perfect fidelity case: P_success = 1.0
            mock_measure.return_value = (1.0, {'0': 1000})
            fidelity, _ = measure_fidelity_with_swap_test(target_prep, noisy_prep, None, None)
            self.assertAlmostEqual(fidelity, 1.0, places=3)
            
            # Worst case: P_success = 0.5  
            mock_measure.return_value = (0.5, {'0': 500, '1': 500})
            fidelity, _ = measure_fidelity_with_swap_test(target_prep, noisy_prep, None, None)
            self.assertAlmostEqual(fidelity, 0.0, places=3)
            
            # Edge case: P_success = 0.0 (should clip to 0.0)
            mock_measure.return_value = (0.0, {'1': 1000})
            fidelity, _ = measure_fidelity_with_swap_test(target_prep, noisy_prep, None, None)
            self.assertAlmostEqual(fidelity, 0.0, places=3)


class TestRecursivePurification(unittest.TestCase):
    """Test recursive purification functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("IBMQ components not available")
        
        # Create test target preparation circuit
        self.target_prep = QuantumCircuit(1)
        self.target_prep.h(0)
    
    def test_create_noisy_copy_circuit(self):
        """Test noisy copy circuit creation."""
        noisy_copy = create_noisy_copy_circuit(
            self.target_prep, 'depolarizing', 0.1, apply_twirling=True, copy_id=42
        )
        
        self.assertIsInstance(noisy_copy, QuantumCircuit)
        self.assertEqual(noisy_copy.num_qubits, self.target_prep.num_qubits)
        self.assertIn('noisy_copy_42', noisy_copy.name)
        
        # Should have more gates than original due to noise
        self.assertGreaterEqual(len(noisy_copy), len(self.target_prep))
    
    def test_run_single_purification_step_success(self):
        """Test single purification step with mocked success."""
        with patch('IBMQ_components.execute_circuits_with_backend') as mock_execute:
            # Mock successful SWAP test
            mock_execute.return_value = [{'0': 800, '1': 200}]  # 80% success
            
            success, results = run_single_purification_step(
                self.target_prep, 'depolarizing', 0.1, True, None, None,
                shots=1000, max_attempts=3
            )
            
            self.assertTrue(success)
            self.assertEqual(results['attempts'], 1)
            self.assertEqual(results['measurements_used'], 1000)
            self.assertAlmostEqual(results['final_success_prob'], 0.8, places=3)
    
    def test_run_single_purification_step_failure(self):
        """Test single purification step with mocked failures."""
        with patch('IBMQ_components.execute_circuits_with_backend') as mock_execute:
            # Mock failed SWAP tests (only measure |1⟩)
            mock_execute.return_value = [{'1': 1000}]  # 0% success
            
            success, results = run_single_purification_step(
                self.target_prep, 'depolarizing', 0.1, True, None, None,
                shots=1000, max_attempts=3
            )
            
            self.assertFalse(success)
            self.assertEqual(results['attempts'], 3)  # Should try all attempts
            self.assertEqual(results['measurements_used'], 3000)  # 3 attempts × 1000 shots
            self.assertEqual(results['final_success_prob'], 0.0)
    
    def test_simulate_recursive_purification_basic(self):
        """Test basic recursive purification simulation."""
        with patch('IBMQ_components.run_single_purification_step') as mock_step:
            # Mock successful purification steps
            mock_step.return_value = (True, {
                'attempts': 1,
                'measurements_used': 1000,
                'final_success_prob': 0.8,
                'error': None
            })
            
            results = simulate_recursive_purification(
                self.target_prep, 'depolarizing', 0.1, num_levels=1,
                backend=None, service=None, apply_twirling=True, shots_per_step=1000
            )
            
            self.assertTrue(results['final_success'])
            self.assertEqual(results['num_levels'], 1)
            self.assertEqual(len(results['levels']), 2)  # Level 0 + Level 1
            self.assertGreater(results['total_measurements'], 0)
    
    def test_simulate_recursive_purification_failure(self):
        """Test recursive purification with step failures."""
        with patch('IBMQ_components.run_single_purification_step') as mock_step:
            # Mock failed purification steps
            mock_step.return_value = (False, {
                'attempts': 5,
                'measurements_used': 5000,
                'final_success_prob': 0.0,
                'error': 'All attempts failed'
            })
            
            results = simulate_recursive_purification(
                self.target_prep, 'depolarizing', 0.5, num_levels=1,
                backend=None, service=None, apply_twirling=True, shots_per_step=1000
            )
            
            self.assertFalse(results['final_success'])
            self.assertIn('error', results)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("IBMQ components not available")
    
    def test_estimate_circuit_resources_basic(self):
        """Test basic circuit resource estimation."""
        M = 2
        resources = estimate_circuit_resources(M, 'depolarizing', apply_twirling=False)
        
        # Check required keys
        expected_keys = [
            'total_qubits', 'qubits_per_register', 'swap_test_gates',
            'total_gates_per_round', 'circuit_depth_estimate'
        ]
        for key in expected_keys:
            self.assertIn(key, resources)
        
        # Check values
        self.assertEqual(resources['total_qubits'], 1 + 2*M)
        self.assertEqual(resources['qubits_per_register'], M)
        self.assertGreater(resources['swap_test_gates'], 0)
        self.assertGreater(resources['total_gates_per_round'], resources['swap_test_gates'])
    
    def test_estimate_circuit_resources_with_twirling(self):
        """Test resource estimation with Clifford twirling."""
        M = 2
        resources_no_twirl = estimate_circuit_resources(M, 'dephase_z', apply_twirling=False)
        resources_twirl = estimate_circuit_resources(M, 'dephase_z', apply_twirling=True)
        
        # Twirling should increase gate count
        self.assertGreater(
            resources_twirl['total_gates_per_round'],
            resources_no_twirl['total_gates_per_round']
        )
    
    def test_estimate_circuit_resources_different_noise_types(self):
        """Test resource estimation for different noise types."""
        M = 2
        
        for noise_type in ['depolarizing', 'dephase_z', 'dephase_x']:
            resources = estimate_circuit_resources(M, noise_type, apply_twirling=False)
            
            # Basic sanity checks
            self.assertEqual(resources['total_qubits'], 1 + 2*M)
            self.assertGreater(resources['total_gates_per_round'], 0)


class TestComponentIntegration(unittest.TestCase):
    """Integration tests combining multiple components."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not COMPONENTS_AVAILABLE:
            self.skipTest("IBMQ components not available")
    
    def test_full_workflow_integration(self):
        """Test complete workflow integration."""
        M = 1
        
        # 1. Create target state
        target_circuit, target_sv = create_target_state_circuit(M, 'hadamard')
        self.assertEqual(target_circuit.num_qubits, M)
        
        # 2. Create noisy copies
        copy_A = create_noisy_copy_circuit(target_circuit, 'depolarizing', 0.1, True, 1)
        copy_B = create_noisy_copy_circuit(target_circuit, 'depolarizing', 0.1, True, 2)
        
        # 3. Create SWAP purification circuit
        swap_circuit = create_swap_purification_circuit(copy_A, copy_B, measure_ancilla=True)
        self.assertEqual(swap_circuit.num_qubits, 1 + 2*M)
        self.assertEqual(swap_circuit.num_clbits, 1)
        
        # 4. Estimate resources
        resources = estimate_circuit_resources(M, 'depolarizing', apply_twirling=True)
        self.assertGreater(resources['total_qubits'], 0)
        
        # 5. Mock execution and analysis
        with patch('IBMQ_components.execute_circuits_with_backend') as mock_execute:
            mock_execute.return_value = [{'0': 600, '1': 400}]
            
            success, prob, analysis = analyze_swap_test_results(
                mock_execute.return_value[0], num_repeats=1
            )
            
            self.assertTrue(success)
            self.assertAlmostEqual(prob, 0.6, places=3)
    
    def test_comprehensive_test_function(self):
        """Test the built-in comprehensive test function."""
        if not QISKIT_TEST_AVAILABLE:
            self.skipTest("Full Qiskit not available for comprehensive test")
        
        test_results = run_comprehensive_test(M=1, noise_type='depolarizing', p=0.1)
        
        self.assertIn('overall_success', test_results)
        self.assertIn('tests', test_results)
        
        # Check that key tests are present
        expected_tests = [
            'state_prep', 'noise_application', 'swap_circuit', 
            'purification_circuit', 'resource_estimation'
        ]
        
        for test_name in expected_tests:
            self.assertIn(test_name, test_results['tests'])


def run_all_tests():
    """Run all tests and provide summary."""
    if not COMPONENTS_AVAILABLE:
        print("❌ Cannot run tests: IBMQ components not available")
        print("   Make sure IBMQ_components.py is in src/simulation/")
        return False
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.wasSuccessful():
        print(f"✅ All tests passed!")
        print(f"🚀 IBMQ components are ready for experiments!")
    else:
        print(f"❌ Some tests failed!")
        
        if result.failures:
            print(f"\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
        
        if result.errors:
            print(f"\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split('Error:')[-1].strip()}")
    
    print(f"{'='*70}")
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run tests when script is executed directly
    success = run_all_tests()
    sys.exit(0 if success else 1)