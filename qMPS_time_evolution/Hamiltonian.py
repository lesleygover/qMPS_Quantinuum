from functools import reduce
from itertools import product
from numpy import zeros, kron, trace, eye, allclose, array
import numpy as np
from scipy.linalg import expm
import cirq


Sx = array([[0, 1],[1, 0]])
Sy = 1j*array([[0, 1],[-1, 0]])
Sz = array([[1, 0],[0, -1]])
I = eye(2)+0j
S = {'I': I, 'X': Sx, 'Y': Sy, 'Z': Sz}

class PauliMeasure(cirq.Gate):
    """PauliMeasure:apply appropriate transformation to 2 qubits
       st. measuring qb[0] in z basis measures string"""
    def __init__(self, string):
        if string=='II':
            raise Exception('don\'t measure the identity')
        assert len(string)==2
        self.string = string

    def _decompose_(self, qubits):

        def single_qubits(string, qubit):
            assert len(string)==1
            if string =='X':
                yield cirq.H(qubit)
            elif string == 'Y':
                yield cirq.inverse(cirq.S(qubit))
                yield cirq.H(qubit)

        i, j = self.string
        if i=='I':
            yield cirq.SWAP(qubits[0], qubits[1])
            j, i = i, j
        yield single_qubits(i, qubits[0])
        yield single_qubits(j, qubits[1])
        if i!='I' and j!='I':
            yield cirq.CNOT(qubits[1], qubits[0])

    def num_qubits(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return list(self.string)

class Hamiltonian:
    """
    Hamiltonian: string of terms in local hamiltonian.
    Just do quadratic spin 1/2
    ex. tfim = Hamiltonian({'ZZ': -1, 'X': λ}) = Hamiltonian({'ZZ': 1, 'IX': λ/2, 'XI': λ/2})
    for parity invariant specify can single site terms ('X')
    otherwise 'IX' 'YI' etc.
    """

    def __init__(self, strings=None):
        self.strings = strings
        if strings is not None:
            for key, val in {key:val for key, val in self.strings.items()}.items():
                if len(key)==1:
                    self.strings['I'+key] = val/2
                    self.strings[key+'I'] = val/2
                    self.strings.pop(key)

    def to_matrix(self):
        assert self.strings is not None
        h_i = zeros((4, 4))+0j
        for js, J in self.strings.items():
            h_i += J*reduce(kron, [S[j] for j in js])
        self._matrix = h_i
        return h_i

    def from_matrix(self, mat):
        xyz = list(S.keys())
        strings = list(product(xyz, xyz))
        self.strings = {a+b:trace(kron(a, b)@mat) for a, b in strings}
        del self.strings['II']
        return self

    def measure_energy(self, circuit, qubits, reps=300000):
        assert self.strings is not None
        ev = 0
        for string, g in self.strings.items():
            c = circuit.copy()
            c.append(PauliMeasure(string)(*qubits))
            c.append(cirq.measure(qubits[0], key=string))

            sim = cirq.Simulator()
            meas = sim.run(c, repetitions=reps).measurements[string]
            ev += g*array(list(map(lambda x: 1-2*int(x), meas))).mean()
        return ev

    def calculate_energy(self, circuit, loc=0):
        c = circuit.copy()
        sim = cirq.Simulator()
        ψ = sim.simulate(c).final_state
        H = self.to_matrix()

        I = eye(2)
        H = reduce(kron, [I]*loc+[H]+[I]*(len(c.all_qubits())-loc-2))
        return np.real(ψ.conj().T@H@ψ)

def evolution_op(g,t):
    n = 2
    H = Hamiltonian(({'ZZ': -1, 'X': g})).to_matrix()
    W = expm(-2j*t*H).reshape([2]*(2*n))
    return W

def evolution_circuit_op(g,t):
    n=2
    H = Hamiltonian(({'ZZ': -1, 'X': g})).to_matrix()
    W = expm(-2j*t*H)
    return W

class Wgate0202(cirq.Gate):
    """ 
    W time evolution operator for TFIM with g = 0.2, t = 0.2
        - broken down into Rz, Ry and 2 CNOT gates
        - more efficient to run on device
    
    """
    def __init__(self):
        super(Wgate0202, self)

    def _decompose_(self, qubits):
        return [
                cirq.rz(-np.pi/2).on(qubits[0]),
                cirq.ry(3.0993167).on(qubits[0]),
                cirq.rz(3.1415367).on(qubits[1]),
                cirq.ry(1.5694742).on(qubits[1]),
                cirq.rz(1.6130723).on(qubits[1]),
                cirq.CNOT(qubits[0],qubits[1]),
                cirq.rz(np.pi/2).on(qubits[0]),
                cirq.ry(-3.141137).on(qubits[0]),
                cirq.rz(-1.5698475).on(qubits[1]),
                cirq.ry(1.5711972).on(qubits[1]),
                cirq.rz(0.77125282).on(qubits[1]),
                cirq.CNOT(qubits[0],qubits[1]),
                cirq.rz(-np.pi/2).on(qubits[0]),
                cirq.ry(0.042275973).on(qubits[0]),
                cirq.rz(-np.pi/2).on(qubits[0]),
                cirq.rz(-3.1103202).on(qubits[1]),
                cirq.ry(0.042296666).on(qubits[1]),
                cirq.rz(-1.6020968).on(qubits[1]),
            ]

    def num_qubits(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return ['W','W']