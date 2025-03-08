import numpy as np
import unittest
import cirq
import re

#########
# test ansatz.py
#########
from ..ansatz import QasmStateAnsatzXZ, stateAnsatzXZ,timeLikeAnsatzXZ,UnitaryAnsatz

class TestQasmStateAnsatzXZ(unittest.TestCase):
    """Tests for the QasmStateAnsatzXZ function."""

    def setUp(self):
        # Create sets of parameters
        """Set up test parameters."""
        self.param_set_list = [0.1, 1, 0.2, 3, 0.4, 5, 0.6, 7]
        self.param_set_array = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        # Create list of two test qubit indices
        self.qubits = [0,1]
        
    def test_parameters(self):
        """Test functionality with list and np.ndarray parameters."""
        result_list = QasmStateAnsatzXZ(self.param_set_list, self.qubits)
        result_array = QasmStateAnsatzXZ(self.param_set_array, self.qubits)
        # Check that the result is a string
        self.assertIsInstance(result_list, str)
        self.assertIsInstance(result_array, str)
        
        # Check that all parameters are included in the result
        for i, param in enumerate(self.param_set_list):
            self.assertIn(f"pi*{param}", result_list)
        for i, param in enumerate(self.param_set_array):
            self.assertIn(f"pi*{param}", result_array)
        
        # Check that both qubits are referenced in the result
        for qubit in self.qubits:
            self.assertIn(f"q[{qubit}]", result_list)
            self.assertIn(f"q[{qubit}]", result_array)
        
        # Check that the CX gates are correctly generated
        self.assertEqual(result_list.count("cx"), 2)
        self.assertEqual(result_array.count("cx"), 2)
        self.assertEqual(result_list.count(f"cx q[{self.qubits[0]}],q[{self.qubits[1]}]"), 2)
        self.assertEqual(result_array.count(f"cx q[{self.qubits[0]}],q[{self.qubits[1]}]"), 2)
        
    def test_different_qubit_indices(self):
        """Test with different qubit indices."""
        qubits = [3, 7]
        qubits_tuple = (3,7)
        qubits_array = np.array([3,7])
        result = QasmStateAnsatzXZ(self.param_set_array, qubits)
        result_tuple = QasmStateAnsatzXZ(self.param_set_array, qubits_tuple)
        result_array = QasmStateAnsatzXZ(self.param_set_array, qubits_tuple)
        
        # Check that both qubits are referenced in the result
        for qubit in qubits:
            self.assertIn(f"q[{qubit}]", result)
            self.assertIn(f"q[{qubit}]", result_tuple)
            self.assertIn(f"q[{qubit}]", result_array)
        
        # Check that the CX gates are correctly generated
        self.assertEqual(result.count(f"cx q[{qubits[0]}],q[{qubits[1]}]"), 2)
        self.assertEqual(result_tuple.count(f"cx q[{qubits_tuple[0]}],q[{qubits_tuple[1]}]"), 2)
        self.assertEqual(result_array.count(f"cx q[{qubits_array[0]}],q[{qubits_array[1]}]"), 2)
    
    def test_gate_order(self):
        """Test that gates appear in the correct order."""
        params = self.param_set_array
        q0, q1 = self.qubits[0], self.qubits[1]
        result = QasmStateAnsatzXZ(params, self.qubits)
        
        # Remove whitespace and comments for easier pattern matching
        clean_result = re.sub(r'//.*?\n', '', result)
        clean_result = re.sub(r'\s+', '', clean_result)
        
        # Check for the correct sequence of gates
        expected_pattern = (
            f"rz\\(pi\\*{params[0]}\\)q\\[{q0}\\];"
            f"rx\\(pi\\*{params[1]}\\)q\\[{q0}\\];"
            f"rz\\(pi\\*{params[2]}\\)q\\[{q1}\\];"
            f"rx\\(pi\\*{params[3]}\\)q\\[{q1}\\];"
            f"cxq\\[{q0}\\],q\\[{q1}\\];"
            f"rz\\(pi\\*{params[4]}\\)q\\[{q0}\\];"
            f"rx\\(pi\\*{params[5]}\\)q\\[{q0}\\];"
            f"rz\\(pi\\*{params[6]}\\)q\\[{q1}\\];"
            f"rx\\(pi\\*{params[7]}\\)q\\[{q1}\\];"
            f"cxq\\[{q0}\\],q\\[{q1}\\];"
        )
        
        self.assertRegex(clean_result, expected_pattern)
    
    def test_invalid_params_length(self):
        """Test with invalid number of parameters."""
        # Test with fewer than 8 parameters
        params = [0, 1, 2]
       
        with self.assertRaises(ValueError):
            QasmStateAnsatzXZ(params, self.qubits)

        
        # Test with more than 8 parameters
        params = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        with self.assertRaises(ValueError):
            QasmStateAnsatzXZ(params, self.qubits)
    
    def test_invalid_qubits_length(self):
        """Test with invalid number of qubit indices."""
        
        # Test with only one qubit
        qubits = [0]
        with self.assertRaises(ValueError):
            QasmStateAnsatzXZ(self.param_set_array, qubits)
        
        # Test with more than two qubits
        qubits = [0, 1, 2]
        with self.assertRaises(ValueError):
            QasmStateAnsatzXZ(self.param_set_array, qubits)
        
    def test_non_integer_qubits(self):
        """Test with non-integer qubit indices."""
        
        # Test with float values for qubits
        qubits = [0.5, 1.5]
        
        with self.assertRaises(TypeError):
            QasmStateAnsatzXZ(self.param_set_array, qubits)
        
        # Test with string values for qubits
        qubits = ["0", "1"]
        
        with self.assertRaises(TypeError):
            QasmStateAnsatzXZ(self.param_set_array, qubits)

    def test_negative_qubit_indices(self):
        """Test with negative qubit indices."""
        qubits = [-1, -2]
        
        with self.assertRaises(ValueError):
            QasmStateAnsatzXZ(self.param_set_array, qubits)

    def test_mixed_valid_invalid_qubits(self):
        """Test with a mix of valid and invalid qubit indices."""
        
        # One valid integer and one float
        qubits = [0, 1.5]
        
        with self.assertRaises(TypeError):
            QasmStateAnsatzXZ(self.param_set_array, qubits)
        
        # One valid non-negative and one negative
        qubits = [0, -1]
        
        with self.assertRaises(ValueError):
            QasmStateAnsatzXZ(self.param_set_list, qubits)

class TestAnsatzXZ(unittest.TestCase):
    """Tests for the stateAnsatzXZ and timeLikeAnsatzXZ classes."""
    
    def setUp(self):
        """Set up test parameters."""
        # Create a set of valid parameters
        self.valid_params = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        # Create two test qubits
        self.q0, self.q1 = cirq.LineQubit.range(2)
        # Create a valid ansatz for stateAnsatzXZ
        self.ansatz_space = stateAnsatzXZ(self.valid_params)
        # Create a valid ansatz for timeLikeAnsatzXZ
        self.ansatz_time = timeLikeAnsatzXZ(self.valid_params)
        
    def test_init_valid_params(self):
        """
        Test initialization with valid parameters.
        - checks length of params
        - checks that parameters outputted are the same as those inputted
        """
        ansatz_space = stateAnsatzXZ(self.valid_params)
        ansatz_time = timeLikeAnsatzXZ(self.valid_params)
        np.testing.assert_array_equal(ansatz_space.params, self.valid_params)
        np.testing.assert_array_equal(ansatz_time.params, self.valid_params)
        
    def test_init_invalid_params(self):
        """Test initialization with invalid number of parameters."""
        invalid_params = np.array([0.1, 0.2, 0.3])
        with self.assertRaises(ValueError):
            stateAnsatzXZ(invalid_params)
        with self.assertRaises(ValueError):
            timeLikeAnsatzXZ(invalid_params)
        
    def test_decomposition(self):
        """
        Test decomposition of the gate.
        - check number of gates
        - check types of gates
        """
        qubits = [self.q0, self.q1]
        gates_space = list(self.ansatz_space._decompose_(qubits))
        gates_time = list(self.ansatz_time._decompose_(qubits))
        
        # Spacelike: There should be 10 gates: 8 rotations + 2 CNOTs
        self.assertEqual(len(gates_space), 10)
        
        # Timelike: There should be 11 gates: 8 rotations + 3 CNOTs
        self.assertEqual(len(gates_time), 11)
        
        # Check specific types of gates and their parameters
        #spacelike
        self.assertIsInstance(gates_space[0].gate, cirq.Rz)
        self.assertEqual(vars(gates_space[0].gate)['_rads'],self.valid_params[0])
        self.assertIsInstance(gates_space[1].gate, cirq.Rx)
        self.assertEqual(vars(gates_space[1].gate)['_rads'],self.valid_params[1])
        #timelike
        self.assertIsInstance(gates_time[0].gate, cirq.Rz)
        self.assertEqual(vars(gates_time[0].gate)['_rads'],self.valid_params[0])
        self.assertIsInstance(gates_time[1].gate, cirq.Rx)
        self.assertEqual(vars(gates_time[1].gate)['_rads'],self.valid_params[1])
        
        #check CNOT gates are in correct positions
        #spacelike
        self.assertIsInstance(gates_space[4].gate, type(cirq.CNOT))
        self.assertIsInstance(gates_space[9].gate, type(cirq.CNOT))
        #timelike
        self.assertIsInstance(gates_time[4].gate, type(cirq.CNOT))
        self.assertIsInstance(gates_time[9].gate, type(cirq.CNOT))
        self.assertIsInstance(gates_time[10].gate, type(cirq.CNOT))

        
    def test_decompose_invalid_qubits(self):
        """Test decomposition with invalid number of qubits."""
        qubits = [self.q0]  # Only one qubit
        with self.assertRaises(ValueError):
            list(self.ansatz_space._decompose_(qubits))
        with self.assertRaises(ValueError):
            list(self.ansatz_time._decompose_(qubits))
            
    def test_gate_in_circuit(self):
        """Test using the gate in a cirq Circuit."""
        qubits = [self.q0, self.q1]
        circuit = [cirq.Circuit(),cirq.Circuit()]
        #spacelike
        circuit[0].append(self.ansatz_space.on(*qubits))
        
        # Check that the circuit has the right number of moments
        # Decomposed gate should have 6 moments
        decomposed_circuit_space = cirq.Circuit(self.ansatz_space._decompose_(qubits))
        self.assertEqual(len(decomposed_circuit_space), 6)
        #timelike
        circuit[1].append(self.ansatz_time.on(*qubits))
        
        # Check that the circuit has the right number of moments
        # Decomposed gate should have 7 moments
        decomposed_circuit_time = cirq.Circuit(self.ansatz_time._decompose_(qubits))
        self.assertEqual(len(decomposed_circuit_time), 7)
        
    def test_parameters_change_unitary(self):
        """Test that changing parameters changes the resulting unitary."""
        params1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        params2 = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        
        # Get the unitaries
        unitaries1 = [cirq.unitary(stateAnsatzXZ(params1)),cirq.unitary(timeLikeAnsatzXZ(params1))]
        unitaries2 = [cirq.unitary(stateAnsatzXZ(params2)),cirq.unitary(timeLikeAnsatzXZ(params2))]
        
        # The unitaries should be different
        self.assertFalse(np.allclose(unitaries1[0], unitaries2[0]))
        self.assertFalse(np.allclose(unitaries1[1], unitaries2[1]))     

class TestUnitaryAnsatz(unittest.TestCase):
    """Tests for the UnitaryAnsatz class."""
    
    def setUp(self):
        """Set up test parameters."""
        # Create a valid unitary matrix
        ## Using Hadamard gate tensor product with itself
        h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        self.unitary_matrix = np.kron(h, h)
        
        # Create a non-unitary matrix
        self.non_unitary_matrix = np.array([
            [1, 1, 1, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Create wrong-sized matrix
        self.wrong_size_matrix = np.eye(3)
        
        # Create two test qubits
        self.q0, self.q1 = cirq.LineQubit.range(2)
        
        # Create a valid ansatz
        self.ansatz = UnitaryAnsatz(self.unitary_matrix)
        
    def test_init_valid_unitary(self):
        """Test initialization with a valid unitary matrix."""
        ansatz = UnitaryAnsatz(self.unitary_matrix)
        np.testing.assert_array_equal(ansatz.U, self.unitary_matrix)
        
    def test_init_non_unitary_matrix(self):
        """Test initialization with a non-unitary matrix."""
        with self.assertRaises(ValueError):
            UnitaryAnsatz(self.non_unitary_matrix)
            
    def test_init_wrong_size_matrix(self):
        """Test initialization with a wrong-sized matrix."""
        with self.assertRaises(ValueError):
            UnitaryAnsatz(self.wrong_size_matrix)
        