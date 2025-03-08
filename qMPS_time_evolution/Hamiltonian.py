from functools import reduce
from itertools import product
from numpy import zeros, kron, trace, eye, array
import numpy as np
from scipy.linalg import expm
import cirq


Sx = array([[0, 1],[1, 0]])
Sy = 1j*array([[0, 1],[-1, 0]])
Sz = array([[1, 0],[0, -1]])
I = eye(2)+0j
S = {'I': I, 'X': Sx, 'Y': Sy, 'Z': Sz}

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

    def calculate_energy(self, circuit, loc=0):
        c = circuit.copy()
        sim = cirq.Simulator()
        ψ = sim.simulate(c).final_state
        H = self.to_matrix()

        I = eye(2)
        H = reduce(kron, [I]*loc+[H]+[I]*(len(c.all_qubits())-loc-2))
        return np.real(ψ.conj().T@H@ψ)

def evolution_op(g,dt):
    '''
    TFIM evolution operator for use in transfer matrix simulation
    ========
    Inputs:
        g (float): coupling strength
        dt (float): length of timestep
    ========
    Outputs:
        W (np.array): the evolution operator in shape (2,2,2,2)
    '''
    n = 2
    H = Hamiltonian(({'ZZ': -1, 'X': g})).to_matrix()
    W = expm(-2j*dt*H).reshape([2]*(2*n))
    return W

def evolution_circuit_op(g,dt):
    '''
    TFIM evolution operator for use in circuit simulation
    ========
    Inputs:
        g (float): coupling strength
        dt (float): length of timestep
    ========
    Outputs:
        W (np.array): the evolution operator in shape (4,4)
    '''
    n=2
    H = Hamiltonian(({'ZZ': -1, 'X': g})).to_matrix()
    W = expm(-2j*dt*H)
    return W

def evolOpQASM(qubits):
    '''
    Produces openqasm string for evolution operator gates
    =============
    Inputs: 
        qubits (list): list number which two qubits the gates should act on
    =============
    Outputs: 
        evolOp_qasm (str): openqasm string of evolution operator
    '''
    evolOp_qasm = f"""
// evolution operator: TFIM Hamiltonian g = 0.2, dt = 0.2

rz(pi*-0.5) q[{qubits[0]}];
ry(pi*0.986543146) q[{qubits[0]}];
rz(pi*0.9999821894) q[{qubits[1]}];
ry(pi*0.499579154) q[{qubits[1]}];
rz(pi*0.5134568602) q[{qubits[1]}];
cx q[{qubits[0]}],q[{qubits[1]}];
rz(pi*0.5) q[{qubits[0]}];
rz(pi*-0.4996979791) q[{qubits[1]}];
ry(pi*-0.999854961) q[{qubits[0]}];
ry(pi*0.5001276019) q[{qubits[1]}];
rz(pi*0.2454973974) q[{qubits[1]}];
cx q[{qubits[0]}],q[{qubits[1]}];
rz(pi*-0.5) q[{qubits[0]}];
rz(pi*-0.9900456689) q[{qubits[1]}];
ry(pi*0.0134568602) q[{qubits[0]}];
ry(pi*0.0134634469) q[{qubits[1]}];
rz(pi*-0.5) q[{qubits[0]}];
rz(pi*-0.5099632501) q[{qubits[1]}];
"""
    return evolOp_qasm

def evolOp2ndOrderQASM(qubits):
    '''
    Produces openqasm string for 2nd order evolution operator gates
    =============
    Inputs: 
        qubits (list): list number which two qubits the gates should act on
    =============
    Outputs: 
        evolOp2ndOrderQASM (str): openqasm string of evolution operator
    '''
    evolOp2ndOrder_qasm = f"""
// 2nd order evolution operator: TFIM Hamiltonian g = 0.2, dt = 0.5

rz(pi*-0.5) q[{qubits[0]}];
ry(pi*-0.0081276962) q[{qubits[0]}];
rz(pi*0.987084481) q[{qubits[1]}];
ry(pi*0.1788599049) q[{qubits[1]}];
rz(pi*0.5152588931) q[{qubits[1]}];
cx q[{qubits[0]}],q[{qubits[1]}];
rx(pi*0.999965997) q[{qubits[0]}];
rz(pi*0.5) q[{qubits[0]}];
rz(pi*-0.6332094846) q[{qubits[1]}];
ry(pi*0.5175620031) q[{qubits[1]}];
rz(pi*0.5822619028) q[{qubits[1]}];
cx q[{qubits[0]}],q[{qubits[1]}];
rz(pi*0.5) q[{qubits[0]}];
ry(pi*0.9918723038) q[{qubits[0]}];
rz(pi*0.5) q[{qubits[0]}];
rz(pi*-0.5096005063) q[{qubits[1]}];
ry(pi*0.3213702675) q[{qubits[1]}];
rz(pi*-0.9948896231) q[{qubits[1]}];
"""
    return evolOp2ndOrder_qasm

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