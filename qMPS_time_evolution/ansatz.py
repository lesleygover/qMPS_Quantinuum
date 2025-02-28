import cirq

class stateAnsatzXZ(cirq.Gate):
	"""
    8-parameter shallow factorization of a 2 qubit unitary using XZ gates
    =============
    Inputs:
        params (np.array): parameters which parametrize the unitary
    """
	def __init__(self, params):
		self.params = params

	def _decompose_(self, qubits):
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
		return ['U1','U2']

class timeLikeAnsatzXZ(cirq.Gate):
    """
    8-parameter shallow factorization of a 2 qubit unitary using XZ gates
    =============
    Inputs:
        params (np.array): parameters which parametrize the unitary
    """
    def __init__(self, params):
        self.params = params

    def _decompose_(self, qubits):
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
        return ['U1','U2']

class UnitaryAnsatz(cirq.Gate):
    """Gate defined using a unitary matrix
    =============
    Inputs:
        U (np.array): unitary which defines the 2 qubit unitary gate
    """

    def __init__(self, U):
        self.U = U

    def num_qubits(self) -> int:
        return 2

    def _unitary_(self):
        return self.U

    def _circuit_diagram_info_(self, args):
        return ['U','U']
    
