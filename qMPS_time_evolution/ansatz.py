import cirq
import numpy as np

class stateAnsatzXZ(cirq.Gate):
    """
    8-parameter shallow factorization of a 2 qubit unitary using XZ gates:
        - uses single-qubit rotations around X and Z axes and 2 CNOT gates.
        - space-like gate ansatz
    =============
    Attributes:
        params (np.ndarray): Array of 8 parameters which parametrize the unitary, requires exactly 8 parameters.
    =============
    Raises:
        ValueError: If params does not contain exactly 8 values.
        ValueError: If number of qubits gate is applied to isn't 2
    """
    def __init__(self, params):
        if len(params) != 8:
             raise ValueError(f"Expected array of 8 parameters, got {len(params)} parameters")
        self.params = params

    def _decompose_(self, qubits):
        if len(qubits) != 2:
            raise ValueError(f"Expected 2 qubits, got {len(qubits)}")
        return [
			cirq.rz(self.params[0]).on(qubits[0]),
			cirq.rx(self.params[1]).on(qubits[0]),
			cirq.rz(self.params[2]).on(qubits[1]), 
			cirq.rx(self.params[3]).on(qubits[1]), 
			cirq.CNOT(*qubits), 
			cirq.rz(self.params[4]).on(qubits[0]), 
			cirq.rx(self.params[5]).on(qubits[0]), 
			cirq.rz(self.params[6]).on(qubits[1]), 
			cirq.rx(self.params[7]).on(qubits[1]), 
			cirq.CNOT(*qubits),
		]

    def num_qubits(self):
        return 2

    def _circuit_diagram_info_(self, args):
    	return ['XZ₁','XZ₂']

class timeLikeAnsatzXZ(cirq.Gate):
    """
    8-parameter shallow factorization of a 2 qubit unitary using XZ gates:
        - uses single-qubit rotations around X and Z axes and 2 CNOT gates.
        - space-like gate ansatz
    =============
    Attributes:
        params (np.ndarray): Array of 8 parameters which parametrize the unitary, requires exactly 8 parameters.
    =============
    Raises:
        ValueError: If params does not contain exactly 8 values.
        ValueError: If number of qubits is applied to isn't 2
    """
    def __init__(self, params):
        if len(params) != 8:
            raise ValueError(f"Expected array of 8 parameters, got {len(params)} parameters")
        self.params = params

    def _decompose_(self, qubits):
        if len(qubits) != 2:
            raise ValueError(f"Expected 2 qubits, got {len(qubits)}")
        return [
            cirq.rz(self.params[0]).on(qubits[0]),
            cirq.rx(self.params[1]).on(qubits[0]),
            cirq.rz(self.params[2]).on(qubits[1]), 
            cirq.rx(self.params[3]).on(qubits[1]), 
            cirq.CNOT(qubits[0],qubits[1]), 
            cirq.rz(self.params[4]).on(qubits[0]), 
            cirq.rx(self.params[5]).on(qubits[0]), 
            cirq.rz(self.params[6]).on(qubits[1]), 
            cirq.rx(self.params[7]).on(qubits[1]), 
            cirq.CNOT(qubits[1],qubits[0]),
            cirq.CNOT(qubits[0],qubits[1]),
        ]

    def num_qubits(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return ['TXZ₁','TXZ₂']

class UnitaryAnsatz(cirq.Gate):
    """
    Two-qubit gate directly defined by a 4x4 unitary matrix.
    =============
    Attributes:
        U (np.ndarray): 4x4 unitary matrix defining the gate operation.
    =============
    Inputs:
        U (np.ndarray): 4x4 unitary matrix defining the gate operation.
    =============
    Raises:
        ValueError: If U is not a unitary matrix or has wrong dimensions.
    """

    def __init__(self, U):
        '''
        Initialise the matrix and check it is a 4x4 unitary
        '''
        U = np.asarray(U)

        # Check matrix dimensions
        if U.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix, got shape {U.shape}")
        
        # Check unitarity
        if not np.allclose(np.eye(4), U @ U.conj().T):
            raise ValueError("Input matrix must be unitary (UU† = I)")
        
        self.U = U

    def num_qubits(self) -> int:
        return 2

    def _unitary_(self):
        return self.U

    def _circuit_diagram_info_(self, args):
        return ['U₁','U₂']
    
