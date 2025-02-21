import pandas as pd
import numpy as np


def evolutionCircuit(param_set1,param_set2):
    '''
    Returns a string of QASM code, with the evolution circuit for g=0.2 in TFIM Hamiltonian and dt=0.2
    Inputs:
        param_set1: (np.array) An array of the first set of parameters
        param_set2: (np.array) An array of the second set of parameters
    '''
    params1 = []
    for array1 in param_set1:
        newArray1 = array1/np.pi
        params1.append(newArray1)

    params2 = []
    for array2 in param_set2:
        newArray2 = array2/np.pi
        params2.append(newArray2)
        
    openqasm = f""" 
OPENQASM 2.0;
include "qelib1.inc";

// evolution circuit with 2 iterations of the transfer matrix

// Qubits: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
qreg q[6];
creg m0[5];  // Measurement: phi
creg m1[5];  // Measurement: psi

// Gate: U applied to qubits 0,1
rz(pi*{params1[0]}) q[0];
rx(pi*{params1[1]}) q[0];
rz(pi*{params1[2]}) q[1];
rx(pi*{params1[3]}) q[1];
cx q[0],q[1];
rz(pi*{params1[4]}) q[0];
rx(pi*{params1[5]}) q[0];
rz(pi*{params1[6]}) q[1];
rx(pi*{params1[7]}) q[1];
cx q[0],q[1];

// Gate: U applied to qubits 5,4
rz(pi*{params1[0]}) q[5];
rx(pi*{params1[1]}) q[5];
rz(pi*{params1[2]}) q[4];
rx(pi*{params1[3]}) q[4];
cx q[5],q[4];
rz(pi*{params1[4]}) q[5];
rx(pi*{params1[5]}) q[5];
rz(pi*{params1[6]}) q[4];
rx(pi*{params1[7]}) q[4];
cx q[5],q[4];

// SWAP test 0
cx q[0],q[5];
h q[0];
measure q[5] -> m0[0];
measure q[0] -> m1[0];
reset q[5];
reset q[0];

// Gate: U' applied to qubits 4,5
rz(pi*{params2[0]}) q[4];
rx(pi*{params2[1]}) q[4];
rz(pi*{params2[2]}) q[5];
rx(pi*{params2[3]}) q[5];
cx q[4],q[5];
rz(pi*{params2[4]}) q[4];
rx(pi*{params2[5]}) q[4];
rz(pi*{params2[6]}) q[5];
rx(pi*{params2[7]}) q[5];
cx q[4],q[5];

// Gate: U applied to qubits 1,0
rz(pi*{params1[0]}) q[1];
rx(pi*{params1[1]}) q[1];
rz(pi*{params1[2]}) q[0];
rx(pi*{params1[3]}) q[0];
cx q[1],q[0];
rz(pi*{params1[4]}) q[1];
rx(pi*{params1[5]}) q[1];
rz(pi*{params1[6]}) q[0];
rx(pi*{params1[7]}) q[0];
cx q[1],q[0];

// Gate: U' applied to qubits 5,3
rz(pi*{params2[0]}) q[5];
rx(pi*{params2[1]}) q[5];
rz(pi*{params2[2]}) q[3];
rx(pi*{params2[3]}) q[3];
cx q[5],q[3];
rz(pi*{params2[4]}) q[5];
rx(pi*{params2[5]}) q[5];
rz(pi*{params2[6]}) q[3];
rx(pi*{params2[7]}) q[3];
cx q[5],q[3];

// Gate: U applied to qubits 0,2
rz(pi*{params1[0]}) q[0];
rx(pi*{params1[1]}) q[0];
rz(pi*{params1[2]}) q[2];
rx(pi*{params1[3]}) q[2];
cx q[0],q[2];
rz(pi*{params1[4]}) q[0];
rx(pi*{params1[5]}) q[0];
rz(pi*{params1[6]}) q[2];
rx(pi*{params1[7]}) q[2];
cx q[0],q[2];

// evolution operator: TFIM Hamiltonian g = 0.2, dt = 0.2

rz(pi*-0.5) q[1];
ry(pi*0.986543146) q[1];
rz(pi*0.9999821894) q[0];
ry(pi*0.499579154) q[0];
rz(pi*0.5134568602) q[0];
cx q[1],q[0];
rz(pi*0.5) q[1];
rz(pi*-0.4996979791) q[0];
ry(pi*-0.999854961) q[1];
ry(pi*0.5001276019) q[0];
rz(pi*0.2454973974) q[0];
cx q[1],q[0];
rz(pi*-0.5) q[1];
rz(pi*-0.9900456689) q[0];
ry(pi*0.0134568602) q[1];
ry(pi*0.0134634469) q[0];
rz(pi*-0.5) q[1];
rz(pi*-0.5099632501) q[0];

// SWAP tests 1,2
cx q[1],q[4];
cx q[0],q[5];
h q[1];
measure q[4] -> m0[1];
h q[0];
measure q[5] -> m0[2];
measure q[1] -> m1[1];
measure q[0] -> m1[2];
reset q[5];
reset q[4];
reset q[1];
reset q[0];

// Gate: U' applied to qubits 3,4
rz(pi*{params2[0]}) q[3];
rx(pi*{params2[1]}) q[3];
rz(pi*{params2[2]}) q[4];
rx(pi*{params2[3]}) q[4];
cx q[3],q[4];
rz(pi*{params2[4]}) q[3];
rx(pi*{params2[5]}) q[3];
rz(pi*{params2[6]}) q[4];
rx(pi*{params2[7]}) q[4];
cx q[3],q[4];

// Gate: U applied to qubits 2,1
rz(pi*{params1[0]}) q[2];
rx(pi*{params1[1]}) q[2];
rz(pi*{params1[2]}) q[1];
rx(pi*{params1[3]}) q[1];
cx q[2],q[1];
rz(pi*{params1[4]}) q[2];
rx(pi*{params1[5]}) q[2];
rz(pi*{params1[6]}) q[1];
rx(pi*{params1[7]}) q[1];
cx q[2],q[1];

// Gate: U' applied to qubits 4,5
rz(pi*{params2[0]}) q[4];
rx(pi*{params2[1]}) q[4];
rz(pi*{params2[2]}) q[5];
rx(pi*{params2[3]}) q[5];
cx q[4],q[5];
rz(pi*{params2[4]}) q[4];
rx(pi*{params2[5]}) q[4];
rz(pi*{params2[6]}) q[5];
rx(pi*{params2[7]}) q[5];
cx q[4],q[5];

// Gate: U applied to qubits 1,0
rz(pi*{params1[0]}) q[1];
rx(pi*{params1[1]}) q[1];
rz(pi*{params1[2]}) q[0];
rx(pi*{params1[3]}) q[0];
cx q[1],q[0];
rz(pi*{params1[4]}) q[1];
rx(pi*{params1[5]}) q[1];
rz(pi*{params1[6]}) q[0];
rx(pi*{params1[7]}) q[0];
cx q[1],q[0];

// evolution operator: TFIM Hamiltonian, g = 0.2, dt = 0.2

rz(pi*-0.5) q[2];
ry(pi*0.986543146) q[2];
rz(pi*0.9999821894) q[1];
ry(pi*0.499579154) q[1];
rz(pi*0.5134568602) q[1];
cx q[2],q[1];
rz(pi*0.5) q[2];
rz(pi*-0.4996979791) q[1];
ry(pi*-0.999854961) q[2];
ry(pi*0.5001276019) q[1];
rz(pi*0.2454973974) q[1];
cx q[2],q[1];
rz(pi*-0.5) q[2];
rz(pi*-0.9900456689) q[1];
ry(pi*0.0134568602) q[2];
ry(pi*0.0134634469) q[1];
rz(pi*-0.5) q[2];
rz(pi*-0.5099632501) q[1];

// SWAP tests 3,4
cx q[1],q[4];
cx q[2],q[3];
h q[1];
measure q[4] -> m0[4];
h q[2];
measure q[3] -> m0[3];
measure q[1] -> m1[4];
measure q[2] -> m1[3];
"""
    return openqasm

def doubledEvolutionCircuit(param_set1,param_set2):
    '''
    Returns a string of QASM code, with the evolution circuit for g=0.2 in TFIM Hamiltonian and dt=0.2
    Inputs:
        param_set1: (np.array) An array of the first set of parameters
        param_set2: (np.array) An array of the second set of parameters
    '''
    params1 = []
    for array1 in param_set1:
        newArray1 = array1/np.pi
        params1.append(newArray1)

    params2 = []
    for array2 in param_set2:
        newArray2 = array2/np.pi
        params2.append(newArray2)
        
    openqasm = f""" 
OPENQASM 2.0;
include "qelib1.inc";

// evolution circuit with 2 iterations of the transfer matrix

// Qubits: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6,0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0)]
qreg q[12];
creg m0[5];  // Measurement: phi
creg m1[5];  // Measurement: psi
creg m2[5];  // Measurement: phi2
creg m3[5];  // Measurement: psi2

// Gate: U applied to qubits 0,1
rz(pi*{params1[0]}) q[0];
rx(pi*{params1[1]}) q[0];
rz(pi*{params1[2]}) q[1];
rx(pi*{params1[3]}) q[1];
cx q[0],q[1];
rz(pi*{params1[4]}) q[0];
rx(pi*{params1[5]}) q[0];
rz(pi*{params1[6]}) q[1];
rx(pi*{params1[7]}) q[1];
cx q[0],q[1];

// Gate: U applied to qubits 5,4
rz(pi*{params1[0]}) q[5];
rx(pi*{params1[1]}) q[5];
rz(pi*{params1[2]}) q[4];
rx(pi*{params1[3]}) q[4];
cx q[5],q[4];
rz(pi*{params1[4]}) q[5];
rx(pi*{params1[5]}) q[5];
rz(pi*{params1[6]}) q[4];
rx(pi*{params1[7]}) q[4];
cx q[5],q[4];

// SWAP test 0
cx q[0],q[5];
h q[0];
measure q[5] -> m0[0];
measure q[0] -> m1[0];
reset q[5];
reset q[0];

// Gate: U' applied to qubits 4,5
rz(pi*{params2[0]}) q[4];
rx(pi*{params2[1]}) q[4];
rz(pi*{params2[2]}) q[5];
rx(pi*{params2[3]}) q[5];
cx q[4],q[5];
rz(pi*{params2[4]}) q[4];
rx(pi*{params2[5]}) q[4];
rz(pi*{params2[6]}) q[5];
rx(pi*{params2[7]}) q[5];
cx q[4],q[5];

// Gate: U applied to qubits 1,0
rz(pi*{params1[0]}) q[1];
rx(pi*{params1[1]}) q[1];
rz(pi*{params1[2]}) q[0];
rx(pi*{params1[3]}) q[0];
cx q[1],q[0];
rz(pi*{params1[4]}) q[1];
rx(pi*{params1[5]}) q[1];
rz(pi*{params1[6]}) q[0];
rx(pi*{params1[7]}) q[0];
cx q[1],q[0];

// Gate: U' applied to qubits 5,3
rz(pi*{params2[0]}) q[5];
rx(pi*{params2[1]}) q[5];
rz(pi*{params2[2]}) q[3];
rx(pi*{params2[3]}) q[3];
cx q[5],q[3];
rz(pi*{params2[4]}) q[5];
rx(pi*{params2[5]}) q[5];
rz(pi*{params2[6]}) q[3];
rx(pi*{params2[7]}) q[3];
cx q[5],q[3];

// Gate: U applied to qubits 0,2
rz(pi*{params1[0]}) q[0];
rx(pi*{params1[1]}) q[0];
rz(pi*{params1[2]}) q[2];
rx(pi*{params1[3]}) q[2];
cx q[0],q[2];
rz(pi*{params1[4]}) q[0];
rx(pi*{params1[5]}) q[0];
rz(pi*{params1[6]}) q[2];
rx(pi*{params1[7]}) q[2];
cx q[0],q[2];

// evolution operator: TFIM Hamiltonian g = 0.2, dt = 0.2

rz(pi*-0.5) q[1];
ry(pi*0.986543146) q[1];
rz(pi*0.9999821894) q[0];
ry(pi*0.499579154) q[0];
rz(pi*0.5134568602) q[0];
cx q[1],q[0];
rz(pi*0.5) q[1];
rz(pi*-0.4996979791) q[0];
ry(pi*-0.999854961) q[1];
ry(pi*0.5001276019) q[0];
rz(pi*0.2454973974) q[0];
cx q[1],q[0];
rz(pi*-0.5) q[1];
rz(pi*-0.9900456689) q[0];
ry(pi*0.0134568602) q[1];
ry(pi*0.0134634469) q[0];
rz(pi*-0.5) q[1];
rz(pi*-0.5099632501) q[0];

// SWAP tests 1,2
cx q[1],q[4];
cx q[0],q[5];
h q[1];
measure q[4] -> m0[1];
h q[0];
measure q[5] -> m0[2];
measure q[1] -> m1[1];
measure q[0] -> m1[2];
reset q[5];
reset q[4];
reset q[1];
reset q[0];

// Gate: U' applied to qubits 3,4
rz(pi*{params2[0]}) q[3];
rx(pi*{params2[1]}) q[3];
rz(pi*{params2[2]}) q[4];
rx(pi*{params2[3]}) q[4];
cx q[3],q[4];
rz(pi*{params2[4]}) q[3];
rx(pi*{params2[5]}) q[3];
rz(pi*{params2[6]}) q[4];
rx(pi*{params2[7]}) q[4];
cx q[3],q[4];

// Gate: U applied to qubits 2,1
rz(pi*{params1[0]}) q[2];
rx(pi*{params1[1]}) q[2];
rz(pi*{params1[2]}) q[1];
rx(pi*{params1[3]}) q[1];
cx q[2],q[1];
rz(pi*{params1[4]}) q[2];
rx(pi*{params1[5]}) q[2];
rz(pi*{params1[6]}) q[1];
rx(pi*{params1[7]}) q[1];
cx q[2],q[1];

// Gate: U' applied to qubits 4,5
rz(pi*{params2[0]}) q[4];
rx(pi*{params2[1]}) q[4];
rz(pi*{params2[2]}) q[5];
rx(pi*{params2[3]}) q[5];
cx q[4],q[5];
rz(pi*{params2[4]}) q[4];
rx(pi*{params2[5]}) q[4];
rz(pi*{params2[6]}) q[5];
rx(pi*{params2[7]}) q[5];
cx q[4],q[5];

// Gate: U applied to qubits 1,0
rz(pi*{params1[0]}) q[1];
rx(pi*{params1[1]}) q[1];
rz(pi*{params1[2]}) q[0];
rx(pi*{params1[3]}) q[0];
cx q[1],q[0];
rz(pi*{params1[4]}) q[1];
rx(pi*{params1[5]}) q[1];
rz(pi*{params1[6]}) q[0];
rx(pi*{params1[7]}) q[0];
cx q[1],q[0];

// evolution operator: TFIM Hamiltonian, g = 0.2, dt = 0.2

rz(pi*-0.5) q[2];
ry(pi*0.986543146) q[2];
rz(pi*0.9999821894) q[1];
ry(pi*0.499579154) q[1];
rz(pi*0.5134568602) q[1];
cx q[2],q[1];
rz(pi*0.5) q[2];
rz(pi*-0.4996979791) q[1];
ry(pi*-0.999854961) q[2];
ry(pi*0.5001276019) q[1];
rz(pi*0.2454973974) q[1];
cx q[2],q[1];
rz(pi*-0.5) q[2];
rz(pi*-0.9900456689) q[1];
ry(pi*0.0134568602) q[2];
ry(pi*0.0134634469) q[1];
rz(pi*-0.5) q[2];
rz(pi*-0.5099632501) q[1];

// SWAP tests 3,4
cx q[1],q[4];
cx q[2],q[3];
h q[1];
measure q[4] -> m0[4];
h q[2];
measure q[3] -> m0[3];
measure q[1] -> m1[4];
measure q[2] -> m1[3];


// 2nd copy of evolution circuit with 2 iterations of the transfer matrix


// Gate: U applied to qubits 0,1
rz(pi*{params1[0]}) q[6];
rx(pi*{params1[1]}) q[6];
rz(pi*{params1[2]}) q[7];
rx(pi*{params1[3]}) q[7];
cx q[6],q[7];
rz(pi*{params1[4]}) q[6];
rx(pi*{params1[5]}) q[6];
rz(pi*{params1[6]}) q[7];
rx(pi*{params1[7]}) q[7];
cx q[6],q[7];

// Gate: U applied to qubits 5,4
rz(pi*{params1[0]}) q[11];
rx(pi*{params1[1]}) q[11];
rz(pi*{params1[2]}) q[10];
rx(pi*{params1[3]}) q[10];
cx q[11],q[10];
rz(pi*{params1[4]}) q[11];
rx(pi*{params1[5]}) q[11];
rz(pi*{params1[6]}) q[10];
rx(pi*{params1[7]}) q[10];
cx q[11],q[10];

// SWAP test 0
cx q[6],q[11];
h q[6];
measure q[11] -> m2[0];
measure q[6] -> m3[0];
reset q[11];
reset q[6];

// Gate: U' applied to qubits 4,5
rz(pi*{params2[0]}) q[10];
rx(pi*{params2[1]}) q[10];
rz(pi*{params2[2]}) q[11];
rx(pi*{params2[3]}) q[11];
cx q[10],q[11];
rz(pi*{params2[4]}) q[10];
rx(pi*{params2[5]}) q[10];
rz(pi*{params2[6]}) q[11];
rx(pi*{params2[7]}) q[11];
cx q[10],q[11];

// Gate: U applied to qubits 1,0
rz(pi*{params1[0]}) q[7];
rx(pi*{params1[1]}) q[7];
rz(pi*{params1[2]}) q[6];
rx(pi*{params1[3]}) q[6];
cx q[1],q[0];
rz(pi*{params1[4]}) q[7];
rx(pi*{params1[5]}) q[7];
rz(pi*{params1[6]}) q[6];
rx(pi*{params1[7]}) q[6];
cx q[7],q[6];

// Gate: U' applied to qubits 5,3
rz(pi*{params2[0]}) q[11];
rx(pi*{params2[1]}) q[11];
rz(pi*{params2[2]}) q[9];
rx(pi*{params2[3]}) q[9];
cx q[11],q[9];
rz(pi*{params2[4]}) q[11];
rx(pi*{params2[5]}) q[11];
rz(pi*{params2[6]}) q[9];
rx(pi*{params2[7]}) q[9];
cx q[11],q[9];

// Gate: U applied to qubits 0,2
rz(pi*{params1[0]}) q[6];
rx(pi*{params1[1]}) q[6];
rz(pi*{params1[2]}) q[8];
rx(pi*{params1[3]}) q[8];
cx q[6],q[8];
rz(pi*{params1[4]}) q[6];
rx(pi*{params1[5]}) q[6];
rz(pi*{params1[6]}) q[8];
rx(pi*{params1[7]}) q[8];
cx q[6],q[8];

// evolution operator: TFIM Hamiltonian g = 0.2, dt = 0.2

rz(pi*-0.5) q[7];
ry(pi*0.986543146) q[7];
rz(pi*0.9999821894) q[6];
ry(pi*0.499579154) q[6];
rz(pi*0.5134568602) q[6];
cx q[7],q[6];
rz(pi*0.5) q[7];
rz(pi*-0.4996979791) q[6];
ry(pi*-0.999854961) q[7];
ry(pi*0.5001276019) q[6];
rz(pi*0.2454973974) q[6];
cx q[7],q[6];
rz(pi*-0.5) q[7];
rz(pi*-0.9900456689) q[6];
ry(pi*0.0134568602) q[7];
ry(pi*0.0134634469) q[6];
rz(pi*-0.5) q[7];
rz(pi*-0.5099632501) q[6];

// SWAP tests 1,2
cx q[7],q[10];
cx q[6],q[11];
h q[7];
measure q[10] -> m2[1];
h q[6];
measure q[11] -> m2[2];
measure q[7] -> m3[1];
measure q[6] -> m3[2];
reset q[11];
reset q[10];
reset q[7];
reset q[6];

// Gate: U' applied to qubits 3,4
rz(pi*{params2[0]}) q[9];
rx(pi*{params2[1]}) q[9];
rz(pi*{params2[2]}) q[10];
rx(pi*{params2[3]}) q[10];
cx q[9],q[10];
rz(pi*{params2[4]}) q[9];
rx(pi*{params2[5]}) q[9];
rz(pi*{params2[6]}) q[10];
rx(pi*{params2[7]}) q[10];
cx q[9],q[10];

// Gate: U applied to qubits 2,1
rz(pi*{params1[0]}) q[8];
rx(pi*{params1[1]}) q[8];
rz(pi*{params1[2]}) q[7];
rx(pi*{params1[3]}) q[7];
cx q[8],q[7];
rz(pi*{params1[4]}) q[8];
rx(pi*{params1[5]}) q[8];
rz(pi*{params1[6]}) q[7];
rx(pi*{params1[7]}) q[7];
cx q[8],q[7];

// Gate: U' applied to qubits 4,5
rz(pi*{params2[0]}) q[10];
rx(pi*{params2[1]}) q[10];
rz(pi*{params2[2]}) q[11];
rx(pi*{params2[3]}) q[11];
cx q[10],q[11];
rz(pi*{params2[4]}) q[10];
rx(pi*{params2[5]}) q[10];
rz(pi*{params2[6]}) q[11];
rx(pi*{params2[7]}) q[11];
cx q[10],q[11];

// Gate: U applied to qubits 1,0
rz(pi*{params1[0]}) q[7];
rx(pi*{params1[1]}) q[7];
rz(pi*{params1[2]}) q[6];
rx(pi*{params1[3]}) q[6];
cx q[7],q[6];
rz(pi*{params1[4]}) q[7];
rx(pi*{params1[5]}) q[7];
rz(pi*{params1[6]}) q[6];
rx(pi*{params1[7]}) q[6];
cx q[7],q[6];

// evolution operator: TFIM Hamiltonian, g = 0.2, dt = 0.2

rz(pi*-0.5) q[8];
ry(pi*0.986543146) q[8];
rz(pi*0.9999821894) q[7];
ry(pi*0.499579154) q[7];
rz(pi*0.5134568602) q[7];
cx q[8],q[7];
rz(pi*0.5) q[8];
rz(pi*-0.4996979791) q[7];
ry(pi*-0.999854961) q[8];
ry(pi*0.5001276019) q[7];
rz(pi*0.2454973974) q[7];
cx q[8],q[7];
rz(pi*-0.5) q[8];
rz(pi*-0.9900456689) q[7];
ry(pi*0.0134568602) q[8];
ry(pi*0.0134634469) q[7];
rz(pi*-0.5) q[8];
rz(pi*-0.5099632501) q[7];

// SWAP tests 3,4
cx q[7],q[10];
cx q[8],q[9];
h q[7];
measure q[10] -> m2[4];
h q[8];
measure q[9] -> m2[3];
measure q[7] -> m3[4];
measure q[8] -> m3[3];
"""
    return openqasm
        

def tripledEvolutionCircuit(param_set1,param_set2):
    '''
    Returns a string of QASM code, with the evolution circuit for g=0.2 in TFIM Hamiltonian and dt=0.2
    Inputs:
        param_set1: (np.array) An array of the first set of parameters
        param_set2: (np.array) An array of the second set of parameters
    '''
    params1 = []
    for array1 in param_set1:
        newArray1 = array1/np.pi
        params1.append(newArray1)

    params2 = []
    for array2 in param_set2:
        newArray2 = array2/np.pi
        params2.append(newArray2)
        
    openqasm = f""" 
OPENQASM 2.0;
include "qelib1.inc";

// evolution circuit with 2 iterations of the transfer matrix

// Qubits: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6,0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12,0), (13, 0), (14, 0), (15, 0), (16, 0), (17, 0)]
qreg q[18];
creg m0[5];  // Measurement: phi
creg m1[5];  // Measurement: psi
creg m2[5];  // Measurement: phi2
creg m3[5];  // Measurement: psi2
creg m4[5];  // Measurement: phi3
creg m5[5];  // Measurement: psi3

// Gate: U applied to qubits 0,1
rz(pi*{params1[0]}) q[0];
rx(pi*{params1[1]}) q[0];
rz(pi*{params1[2]}) q[1];
rx(pi*{params1[3]}) q[1];
cx q[0],q[1];
rz(pi*{params1[4]}) q[0];
rx(pi*{params1[5]}) q[0];
rz(pi*{params1[6]}) q[1];
rx(pi*{params1[7]}) q[1];
cx q[0],q[1];

// Gate: U applied to qubits 5,4
rz(pi*{params1[0]}) q[5];
rx(pi*{params1[1]}) q[5];
rz(pi*{params1[2]}) q[4];
rx(pi*{params1[3]}) q[4];
cx q[5],q[4];
rz(pi*{params1[4]}) q[5];
rx(pi*{params1[5]}) q[5];
rz(pi*{params1[6]}) q[4];
rx(pi*{params1[7]}) q[4];
cx q[5],q[4];

// SWAP test 0
cx q[0],q[5];
h q[0];
measure q[5] -> m0[0];
measure q[0] -> m1[0];
reset q[5];
reset q[0];

// Gate: U' applied to qubits 4,5
rz(pi*{params2[0]}) q[4];
rx(pi*{params2[1]}) q[4];
rz(pi*{params2[2]}) q[5];
rx(pi*{params2[3]}) q[5];
cx q[4],q[5];
rz(pi*{params2[4]}) q[4];
rx(pi*{params2[5]}) q[4];
rz(pi*{params2[6]}) q[5];
rx(pi*{params2[7]}) q[5];
cx q[4],q[5];

// Gate: U applied to qubits 1,0
rz(pi*{params1[0]}) q[1];
rx(pi*{params1[1]}) q[1];
rz(pi*{params1[2]}) q[0];
rx(pi*{params1[3]}) q[0];
cx q[1],q[0];
rz(pi*{params1[4]}) q[1];
rx(pi*{params1[5]}) q[1];
rz(pi*{params1[6]}) q[0];
rx(pi*{params1[7]}) q[0];
cx q[1],q[0];

// Gate: U' applied to qubits 5,3
rz(pi*{params2[0]}) q[5];
rx(pi*{params2[1]}) q[5];
rz(pi*{params2[2]}) q[3];
rx(pi*{params2[3]}) q[3];
cx q[5],q[3];
rz(pi*{params2[4]}) q[5];
rx(pi*{params2[5]}) q[5];
rz(pi*{params2[6]}) q[3];
rx(pi*{params2[7]}) q[3];
cx q[5],q[3];

// Gate: U applied to qubits 0,2
rz(pi*{params1[0]}) q[0];
rx(pi*{params1[1]}) q[0];
rz(pi*{params1[2]}) q[2];
rx(pi*{params1[3]}) q[2];
cx q[0],q[2];
rz(pi*{params1[4]}) q[0];
rx(pi*{params1[5]}) q[0];
rz(pi*{params1[6]}) q[2];
rx(pi*{params1[7]}) q[2];
cx q[0],q[2];

// evolution operator: TFIM Hamiltonian g = 0.2, dt = 0.2

rz(pi*-0.5) q[1];
ry(pi*0.986543146) q[1];
rz(pi*0.9999821894) q[0];
ry(pi*0.499579154) q[0];
rz(pi*0.5134568602) q[0];
cx q[1],q[0];
rz(pi*0.5) q[1];
rz(pi*-0.4996979791) q[0];
ry(pi*-0.999854961) q[1];
ry(pi*0.5001276019) q[0];
rz(pi*0.2454973974) q[0];
cx q[1],q[0];
rz(pi*-0.5) q[1];
rz(pi*-0.9900456689) q[0];
ry(pi*0.0134568602) q[1];
ry(pi*0.0134634469) q[0];
rz(pi*-0.5) q[1];
rz(pi*-0.5099632501) q[0];

// SWAP tests 1,2
cx q[1],q[4];
cx q[0],q[5];
h q[1];
measure q[4] -> m0[1];
h q[0];
measure q[5] -> m0[2];
measure q[1] -> m1[1];
measure q[0] -> m1[2];
reset q[5];
reset q[4];
reset q[1];
reset q[0];

// Gate: U' applied to qubits 3,4
rz(pi*{params2[0]}) q[3];
rx(pi*{params2[1]}) q[3];
rz(pi*{params2[2]}) q[4];
rx(pi*{params2[3]}) q[4];
cx q[3],q[4];
rz(pi*{params2[4]}) q[3];
rx(pi*{params2[5]}) q[3];
rz(pi*{params2[6]}) q[4];
rx(pi*{params2[7]}) q[4];
cx q[3],q[4];

// Gate: U applied to qubits 2,1
rz(pi*{params1[0]}) q[2];
rx(pi*{params1[1]}) q[2];
rz(pi*{params1[2]}) q[1];
rx(pi*{params1[3]}) q[1];
cx q[2],q[1];
rz(pi*{params1[4]}) q[2];
rx(pi*{params1[5]}) q[2];
rz(pi*{params1[6]}) q[1];
rx(pi*{params1[7]}) q[1];
cx q[2],q[1];

// Gate: U' applied to qubits 4,5
rz(pi*{params2[0]}) q[4];
rx(pi*{params2[1]}) q[4];
rz(pi*{params2[2]}) q[5];
rx(pi*{params2[3]}) q[5];
cx q[4],q[5];
rz(pi*{params2[4]}) q[4];
rx(pi*{params2[5]}) q[4];
rz(pi*{params2[6]}) q[5];
rx(pi*{params2[7]}) q[5];
cx q[4],q[5];

// Gate: U applied to qubits 1,0
rz(pi*{params1[0]}) q[1];
rx(pi*{params1[1]}) q[1];
rz(pi*{params1[2]}) q[0];
rx(pi*{params1[3]}) q[0];
cx q[1],q[0];
rz(pi*{params1[4]}) q[1];
rx(pi*{params1[5]}) q[1];
rz(pi*{params1[6]}) q[0];
rx(pi*{params1[7]}) q[0];
cx q[1],q[0];

// evolution operator: TFIM Hamiltonian, g = 0.2, dt = 0.2

rz(pi*-0.5) q[2];
ry(pi*0.986543146) q[2];
rz(pi*0.9999821894) q[1];
ry(pi*0.499579154) q[1];
rz(pi*0.5134568602) q[1];
cx q[2],q[1];
rz(pi*0.5) q[2];
rz(pi*-0.4996979791) q[1];
ry(pi*-0.999854961) q[2];
ry(pi*0.5001276019) q[1];
rz(pi*0.2454973974) q[1];
cx q[2],q[1];
rz(pi*-0.5) q[2];
rz(pi*-0.9900456689) q[1];
ry(pi*0.0134568602) q[2];
ry(pi*0.0134634469) q[1];
rz(pi*-0.5) q[2];
rz(pi*-0.5099632501) q[1];

// SWAP tests 3,4
cx q[1],q[4];
cx q[2],q[3];
h q[1];
measure q[4] -> m0[4];
h q[2];
measure q[3] -> m0[3];
measure q[1] -> m1[4];
measure q[2] -> m1[3];


// 2nd copy of evolution circuit with 2 iterations of the transfer matrix


// Gate: U applied to qubits 0,1
rz(pi*{params1[0]}) q[6];
rx(pi*{params1[1]}) q[6];
rz(pi*{params1[2]}) q[7];
rx(pi*{params1[3]}) q[7];
cx q[6],q[7];
rz(pi*{params1[4]}) q[6];
rx(pi*{params1[5]}) q[6];
rz(pi*{params1[6]}) q[7];
rx(pi*{params1[7]}) q[7];
cx q[6],q[7];

// Gate: U applied to qubits 5,4
rz(pi*{params1[0]}) q[11];
rx(pi*{params1[1]}) q[11];
rz(pi*{params1[2]}) q[10];
rx(pi*{params1[3]}) q[10];
cx q[11],q[10];
rz(pi*{params1[4]}) q[11];
rx(pi*{params1[5]}) q[11];
rz(pi*{params1[6]}) q[10];
rx(pi*{params1[7]}) q[10];
cx q[11],q[10];

// SWAP test 0
cx q[6],q[11];
h q[6];
measure q[11] -> m2[0];
measure q[6] -> m3[0];
reset q[11];
reset q[6];

// Gate: U' applied to qubits 4,5
rz(pi*{params2[0]}) q[10];
rx(pi*{params2[1]}) q[10];
rz(pi*{params2[2]}) q[11];
rx(pi*{params2[3]}) q[11];
cx q[10],q[11];
rz(pi*{params2[4]}) q[10];
rx(pi*{params2[5]}) q[10];
rz(pi*{params2[6]}) q[11];
rx(pi*{params2[7]}) q[11];
cx q[10],q[11];

// Gate: U applied to qubits 1,0
rz(pi*{params1[0]}) q[7];
rx(pi*{params1[1]}) q[7];
rz(pi*{params1[2]}) q[6];
rx(pi*{params1[3]}) q[6];
cx q[1],q[0];
rz(pi*{params1[4]}) q[7];
rx(pi*{params1[5]}) q[7];
rz(pi*{params1[6]}) q[6];
rx(pi*{params1[7]}) q[6];
cx q[7],q[6];

// Gate: U' applied to qubits 5,3
rz(pi*{params2[0]}) q[11];
rx(pi*{params2[1]}) q[11];
rz(pi*{params2[2]}) q[9];
rx(pi*{params2[3]}) q[9];
cx q[11],q[9];
rz(pi*{params2[4]}) q[11];
rx(pi*{params2[5]}) q[11];
rz(pi*{params2[6]}) q[9];
rx(pi*{params2[7]}) q[9];
cx q[11],q[9];

// Gate: U applied to qubits 0,2
rz(pi*{params1[0]}) q[6];
rx(pi*{params1[1]}) q[6];
rz(pi*{params1[2]}) q[8];
rx(pi*{params1[3]}) q[8];
cx q[6],q[8];
rz(pi*{params1[4]}) q[6];
rx(pi*{params1[5]}) q[6];
rz(pi*{params1[6]}) q[8];
rx(pi*{params1[7]}) q[8];
cx q[6],q[8];

// evolution operator: TFIM Hamiltonian g = 0.2, dt = 0.2

rz(pi*-0.5) q[7];
ry(pi*0.986543146) q[7];
rz(pi*0.9999821894) q[6];
ry(pi*0.499579154) q[6];
rz(pi*0.5134568602) q[6];
cx q[7],q[6];
rz(pi*0.5) q[7];
rz(pi*-0.4996979791) q[6];
ry(pi*-0.999854961) q[7];
ry(pi*0.5001276019) q[6];
rz(pi*0.2454973974) q[6];
cx q[7],q[6];
rz(pi*-0.5) q[7];
rz(pi*-0.9900456689) q[6];
ry(pi*0.0134568602) q[7];
ry(pi*0.0134634469) q[6];
rz(pi*-0.5) q[7];
rz(pi*-0.5099632501) q[6];

// SWAP tests 1,2
cx q[7],q[10];
cx q[6],q[11];
h q[7];
measure q[10] -> m2[1];
h q[6];
measure q[11] -> m2[2];
measure q[7] -> m3[1];
measure q[6] -> m3[2];
reset q[11];
reset q[10];
reset q[7];
reset q[6];

// Gate: U' applied to qubits 3,4
rz(pi*{params2[0]}) q[9];
rx(pi*{params2[1]}) q[9];
rz(pi*{params2[2]}) q[10];
rx(pi*{params2[3]}) q[10];
cx q[9],q[10];
rz(pi*{params2[4]}) q[9];
rx(pi*{params2[5]}) q[9];
rz(pi*{params2[6]}) q[10];
rx(pi*{params2[7]}) q[10];
cx q[9],q[10];

// Gate: U applied to qubits 2,1
rz(pi*{params1[0]}) q[8];
rx(pi*{params1[1]}) q[8];
rz(pi*{params1[2]}) q[7];
rx(pi*{params1[3]}) q[7];
cx q[8],q[7];
rz(pi*{params1[4]}) q[8];
rx(pi*{params1[5]}) q[8];
rz(pi*{params1[6]}) q[7];
rx(pi*{params1[7]}) q[7];
cx q[8],q[7];

// Gate: U' applied to qubits 4,5
rz(pi*{params2[0]}) q[10];
rx(pi*{params2[1]}) q[10];
rz(pi*{params2[2]}) q[11];
rx(pi*{params2[3]}) q[11];
cx q[10],q[11];
rz(pi*{params2[4]}) q[10];
rx(pi*{params2[5]}) q[10];
rz(pi*{params2[6]}) q[11];
rx(pi*{params2[7]}) q[11];
cx q[10],q[11];

// Gate: U applied to qubits 1,0
rz(pi*{params1[0]}) q[7];
rx(pi*{params1[1]}) q[7];
rz(pi*{params1[2]}) q[6];
rx(pi*{params1[3]}) q[6];
cx q[7],q[6];
rz(pi*{params1[4]}) q[7];
rx(pi*{params1[5]}) q[7];
rz(pi*{params1[6]}) q[6];
rx(pi*{params1[7]}) q[6];
cx q[7],q[6];

// evolution operator: TFIM Hamiltonian, g = 0.2, dt = 0.2

rz(pi*-0.5) q[8];
ry(pi*0.986543146) q[8];
rz(pi*0.9999821894) q[7];
ry(pi*0.499579154) q[7];
rz(pi*0.5134568602) q[7];
cx q[8],q[7];
rz(pi*0.5) q[8];
rz(pi*-0.4996979791) q[7];
ry(pi*-0.999854961) q[8];
ry(pi*0.5001276019) q[7];
rz(pi*0.2454973974) q[7];
cx q[8],q[7];
rz(pi*-0.5) q[8];
rz(pi*-0.9900456689) q[7];
ry(pi*0.0134568602) q[8];
ry(pi*0.0134634469) q[7];
rz(pi*-0.5) q[8];
rz(pi*-0.5099632501) q[7];

// SWAP tests 3,4
cx q[7],q[10];
cx q[8],q[9];
h q[7];
measure q[10] -> m2[4];
h q[8];
measure q[9] -> m2[3];
measure q[7] -> m3[4];
measure q[8] -> m3[3];


// 3rd copy of evolution circuit with 2 iterations of the transfer matrix


// Gate: U applied to qubits 0,1
rz(pi*{params1[0]}) q[12];
rx(pi*{params1[1]}) q[12];
rz(pi*{params1[2]}) q[13];
rx(pi*{params1[3]}) q[13];
cx q[12],q[13];
rz(pi*{params1[4]}) q[12];
rx(pi*{params1[5]}) q[12];
rz(pi*{params1[6]}) q[13];
rx(pi*{params1[7]}) q[13];
cx q[12],q[13];

// Gate: U applied to qubits 5,4
rz(pi*{params1[0]}) q[17];
rx(pi*{params1[1]}) q[17];
rz(pi*{params1[2]}) q[16];
rx(pi*{params1[3]}) q[16];
cx q[17],q[16];
rz(pi*{params1[4]}) q[17];
rx(pi*{params1[5]}) q[17];
rz(pi*{params1[6]}) q[16];
rx(pi*{params1[7]}) q[16];
cx q[17],q[16];

// SWAP test 0
cx q[12],q[17];
h q[12];
measure q[17] -> m4[0];
measure q[12] -> m5[0];
reset q[17];
reset q[12];

// Gate: U' applied to qubits 4,5
rz(pi*{params2[0]}) q[16];
rx(pi*{params2[1]}) q[16];
rz(pi*{params2[2]}) q[17];
rx(pi*{params2[3]}) q[17];
cx q[16],q[17];
rz(pi*{params2[4]}) q[16];
rx(pi*{params2[5]}) q[16];
rz(pi*{params2[6]}) q[17];
rx(pi*{params2[7]}) q[17];
cx q[16],q[17];

// Gate: U applied to qubits 1,0
rz(pi*{params1[0]}) q[13];
rx(pi*{params1[1]}) q[13];
rz(pi*{params1[2]}) q[12];
rx(pi*{params1[3]}) q[12];
cx q[13],q[12];
rz(pi*{params1[4]}) q[13];
rx(pi*{params1[5]}) q[13];
rz(pi*{params1[6]}) q[12];
rx(pi*{params1[7]}) q[12];
cx q[13],q[12];


// Gate: U' applied to qubits 5,3
rz(pi*{params2[0]}) q[17];
rx(pi*{params2[1]}) q[17];
rz(pi*{params2[2]}) q[15];
rx(pi*{params2[3]}) q[15];
cx q[17],q[15];
rz(pi*{params2[4]}) q[17];
rx(pi*{params2[5]}) q[17];
rz(pi*{params2[6]}) q[15];
rx(pi*{params2[7]}) q[15];
cx q[17],q[15];

// Gate: U applied to qubits 0,2
rz(pi*{params1[0]}) q[12];
rx(pi*{params1[1]}) q[12];
rz(pi*{params1[2]}) q[14];
rx(pi*{params1[3]}) q[14];
cx q[12],q[14];
rz(pi*{params1[4]}) q[12];
rx(pi*{params1[5]}) q[12];
rz(pi*{params1[6]}) q[14];
rx(pi*{params1[7]}) q[14];
cx q[12],q[14];

// evolution operator: TFIM Hamiltonian g = 0.2, dt = 0.2

rz(pi*-0.5) q[13];
ry(pi*0.986543146) q[13];
rz(pi*0.9999821894) q[12];
ry(pi*0.499579154) q[12];
rz(pi*0.5134568602) q[12];
cx q[13],q[12];
rz(pi*0.5) q[13];
rz(pi*-0.4996979791) q[12];
ry(pi*-0.999854961) q[13];
ry(pi*0.5001276019) q[12];
rz(pi*0.2454973974) q[12];
cx q[13],q[12];
rz(pi*-0.5) q[13];
rz(pi*-0.9900456689) q[12];
ry(pi*0.0134568602) q[13];
ry(pi*0.0134634469) q[12];
rz(pi*-0.5) q[13];
rz(pi*-0.5099632501) q[12];

// SWAP tests 1,2
cx q[13],q[16];
cx q[12],q[17];
h q[13];
measure q[16] -> m4[1];
h q[12];
measure q[17] -> m4[2];
measure q[13] -> m5[1];
measure q[12] -> m5[2];
reset q[17];
reset q[16];
reset q[13];
reset q[12];

// Gate: U' applied to qubits 3,4
rz(pi*{params2[0]}) q[15];
rx(pi*{params2[1]}) q[15];
rz(pi*{params2[2]}) q[16];
rx(pi*{params2[3]}) q[16];
cx q[15],q[16];
rz(pi*{params2[4]}) q[15];
rx(pi*{params2[5]}) q[15];
rz(pi*{params2[6]}) q[16];
rx(pi*{params2[7]}) q[16];
cx q[15],q[16];

// Gate: U applied to qubits 2,1
rz(pi*{params1[0]}) q[14];
rx(pi*{params1[1]}) q[14];
rz(pi*{params1[2]}) q[13];
rx(pi*{params1[3]}) q[13];
cx q[14],q[13];
rz(pi*{params1[4]}) q[14];
rx(pi*{params1[5]}) q[14];
rz(pi*{params1[6]}) q[13];
rx(pi*{params1[7]}) q[13];
cx q[14],q[13];

// Gate: U' applied to qubits 4,5
rz(pi*{params2[0]}) q[16];
rx(pi*{params2[1]}) q[16];
rz(pi*{params2[2]}) q[17];
rx(pi*{params2[3]}) q[17];
cx q[16],q[17];
rz(pi*{params2[4]}) q[16];
rx(pi*{params2[5]}) q[16];
rz(pi*{params2[6]}) q[17];
rx(pi*{params2[7]}) q[17];
cx q[16],q[17];

// Gate: U applied to qubits 1,0
rz(pi*{params1[0]}) q[13];
rx(pi*{params1[1]}) q[13];
rz(pi*{params1[2]}) q[12];
rx(pi*{params1[3]}) q[12];
cx q[13],q[12];
rz(pi*{params1[4]}) q[13];
rx(pi*{params1[5]}) q[13];
rz(pi*{params1[6]}) q[12];
rx(pi*{params1[7]}) q[12];
cx q[13],q[12];

// evolution operator: TFIM Hamiltonian, g = 0.2, dt = 0.2

rz(pi*-0.5) q[14];
ry(pi*0.986543146) q[14];
rz(pi*0.9999821894) q[13];
ry(pi*0.499579154) q[13];
rz(pi*0.5134568602) q[13];
cx q[14],q[13];
rz(pi*0.5) q[14];
rz(pi*-0.4996979791) q[13];
ry(pi*-0.999854961) q[14];
ry(pi*0.5001276019) q[13];
rz(pi*0.2454973974) q[13];
cx q[14],q[13];
rz(pi*-0.5) q[14];
rz(pi*-0.9900456689) q[13];
ry(pi*0.0134568602) q[14];
ry(pi*0.0134634469) q[13];
rz(pi*-0.5) q[14];
rz(pi*-0.5099632501) q[13];

// SWAP tests 3,4
cx q[13],q[16];
cx q[14],q[15];
h q[13];
measure q[16] -> m4[4];
h q[14];
measure q[15] -> m4[3];
measure q[13] -> m5[4];
measure q[14] -> m5[3];


"""
    return openqasm

def twoCircuitsEvolutionCircuit(param_set1,param_set2):
    '''
    Returns a string of QASM code, with the evolution circuit for g=0.2 in TFIM Hamiltonian and dt=0.2
    Inputs:
        param_set1: (np.array) An array of the first set of parameters
        param_set2: (np.array) An array of the second set of parameters
    '''
    params1 = []
    for array1 in param_set1:
        newArray1 = array1/np.pi
        params1.append(newArray1)

    params2 = []
    for array2 in param_set2:
        newArray2 = array2/np.pi
        params2.append(newArray2)
        
    openqasm = f""" 
OPENQASM 2.0;
include "qelib1.inc";

// evolution circuit with 2 iterations of the transfer matrix

// Qubits: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6,0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0)]
qreg q[12];
creg m0[5];  // Measurement: phi
creg m1[5];  // Measurement: psi
creg m2[5];  // Measurement: phi2
creg m3[5];  // Measurement: psi2

// first TM

rz(pi*{params1[0]}) q[0];
rx(pi*{params1[1]}) q[0];
rz(pi*{params1[2]}) q[1];
rx(pi*{params1[3]}) q[1];
rz(pi*{params1[0]}) q[6];
rx(pi*{params1[1]}) q[6];
rz(pi*{params1[2]}) q[7];
rx(pi*{params1[3]}) q[7];
rz(pi*{params1[0]}) q[5];
rx(pi*{params1[1]}) q[5];
rz(pi*{params1[2]}) q[4];
rx(pi*{params1[3]}) q[4];
rz(pi*{params1[0]}) q[11];
rx(pi*{params1[1]}) q[11];
rz(pi*{params1[2]}) q[10];
rx(pi*{params1[3]}) q[10];

cx q[0],q[1];
cx q[6],q[7];
cx q[5],q[4];
cx q[11],q[10];

rz(pi*{params1[4]}) q[0];
rx(pi*{params1[5]}) q[0];
rz(pi*{params1[6]}) q[1];
rx(pi*{params1[7]}) q[1];
rz(pi*{params1[4]}) q[6];
rx(pi*{params1[5]}) q[6];
rz(pi*{params1[6]}) q[7];
rx(pi*{params1[7]}) q[7];
rz(pi*{params1[4]}) q[5];
rx(pi*{params1[5]}) q[5];
rz(pi*{params1[6]}) q[4];
rx(pi*{params1[7]}) q[4];
rz(pi*{params1[4]}) q[11];
rx(pi*{params1[5]}) q[11];
rz(pi*{params1[6]}) q[10];
rx(pi*{params1[7]}) q[10];

cx q[11],q[10];
cx q[5],q[4];
cx q[6],q[7];
cx q[0],q[1];


// SWAP tests 0
cx q[0],q[5];
cx q[6],q[11];
h q[6];
h q[0];
measure q[5] -> m0[0];
measure q[0] -> m1[0];
measure q[11] -> m2[0];
measure q[6] -> m3[0];
reset q[11];
reset q[6];
reset q[5];
reset q[0];


// Second TM
rz(pi*{params2[0]}) q[4];
rx(pi*{params2[1]}) q[4];
rz(pi*{params2[2]}) q[5];
rx(pi*{params2[3]}) q[5];
rz(pi*{params2[0]}) q[10];
rx(pi*{params2[1]}) q[10];
rz(pi*{params2[2]}) q[11];
rx(pi*{params2[3]}) q[11];
rz(pi*{params1[0]}) q[1];
rx(pi*{params1[1]}) q[1];
rz(pi*{params1[2]}) q[0];
rx(pi*{params1[3]}) q[0];
rz(pi*{params1[0]}) q[7];
rx(pi*{params1[1]}) q[7];
rz(pi*{params1[2]}) q[6];
rx(pi*{params1[3]}) q[6];
rz(pi*{params2[0]}) q[5];
rx(pi*{params2[1]}) q[5];
rz(pi*{params2[2]}) q[3];
rx(pi*{params2[3]}) q[3];
rz(pi*{params2[0]}) q[11];
rx(pi*{params2[1]}) q[11];
rz(pi*{params2[2]}) q[9];
rx(pi*{params2[3]}) q[9];
rz(pi*{params1[0]}) q[0];
rx(pi*{params1[1]}) q[0];
rz(pi*{params1[2]}) q[2];
rx(pi*{params1[3]}) q[2];
rz(pi*{params1[0]}) q[6];
rx(pi*{params1[1]}) q[6];
rz(pi*{params1[2]}) q[8];
rx(pi*{params1[3]}) q[8];

cx q[4],q[5];
cx q[10],q[11];
cx q[1],q[0];
cx q[1],q[0];
cx q[5],q[3];
cx q[11],q[9];
cx q[0],q[2];
cx q[6],q[8];
rz(pi*{params2[4]}) q[4];
rx(pi*{params2[5]}) q[4];
rz(pi*{params2[6]}) q[5];
rx(pi*{params2[7]}) q[5];
rz(pi*{params2[4]}) q[10];
rx(pi*{params2[5]}) q[10];
rz(pi*{params2[6]}) q[11];
rx(pi*{params2[7]}) q[11];
rz(pi*{params1[4]}) q[1];
rx(pi*{params1[5]}) q[1];
rz(pi*{params1[6]}) q[0];
rx(pi*{params1[7]}) q[0];
rz(pi*{params1[4]}) q[7];
rx(pi*{params1[5]}) q[7];
rz(pi*{params1[6]}) q[6];
rx(pi*{params1[7]}) q[6];
rz(pi*{params2[4]}) q[5];
rx(pi*{params2[5]}) q[5];
rz(pi*{params2[6]}) q[3];
rx(pi*{params2[7]}) q[3];
rz(pi*{params2[4]}) q[11];
rx(pi*{params2[5]}) q[11];
rz(pi*{params2[6]}) q[9];
rx(pi*{params2[7]}) q[9];
rz(pi*{params1[4]}) q[0];
rx(pi*{params1[5]}) q[0];
rz(pi*{params1[6]}) q[2];
rx(pi*{params1[7]}) q[2];
rz(pi*{params1[4]}) q[6];
rx(pi*{params1[5]}) q[6];
rz(pi*{params1[6]}) q[8];
rx(pi*{params1[7]}) q[8];
cx q[6],q[8];
cx q[0],q[2];
cx q[11],q[9];
cx q[5],q[3];
cx q[7],q[6];
cx q[1],q[0];
cx q[10],q[11];
cx q[4],q[5];

// evolution operators: TFIM Hamiltonian g = 0.2, dt = 0.2

rz(pi*-0.5) q[1];
ry(pi*0.986543146) q[1];
rz(pi*0.9999821894) q[0];
ry(pi*0.499579154) q[0];
rz(pi*0.5134568602) q[0];
rz(pi*-0.5) q[7];
ry(pi*0.986543146) q[7];
rz(pi*0.9999821894) q[6];
ry(pi*0.499579154) q[6];
rz(pi*0.5134568602) q[6];
cx q[1],q[0];
cx q[7],q[6];
rz(pi*0.5) q[1];
rz(pi*-0.4996979791) q[0];
ry(pi*-0.999854961) q[1];
ry(pi*0.5001276019) q[0];
rz(pi*0.2454973974) q[0];
rz(pi*0.5) q[7];
rz(pi*-0.4996979791) q[6];
ry(pi*-0.999854961) q[7];
ry(pi*0.5001276019) q[6];
rz(pi*0.2454973974) q[6];
cx q[7],q[6];
cx q[1],q[0];
rz(pi*-0.5) q[1];
rz(pi*-0.9900456689) q[0];
ry(pi*0.0134568602) q[1];
ry(pi*0.0134634469) q[0];
rz(pi*-0.5) q[1];
rz(pi*-0.5099632501) q[0];
rz(pi*-0.5) q[7];
rz(pi*-0.9900456689) q[6];
ry(pi*0.0134568602) q[7];
ry(pi*0.0134634469) q[6];
rz(pi*-0.5) q[7];
rz(pi*-0.5099632501) q[6];

// SWAP tests 1,2
cx q[1],q[4];
cx q[0],q[5];cx q[7],q[10];
cx q[6],q[11];
h q[7];
h q[1];
measure q[10] -> m2[1];
measure q[4] -> m0[1];
h q[0];
h q[6];
measure q[11] -> m2[2];
measure q[7] -> m3[1];
measure q[6] -> m3[2];
measure q[5] -> m0[2];
measure q[1] -> m1[1];
measure q[0] -> m1[2];
reset q[5];
reset q[4];
reset q[1];
reset q[0];
reset q[11];
reset q[10];
reset q[7];
reset q[6];


// 3rd TM
rz(pi*{params2[0]}) q[3];
rx(pi*{params2[1]}) q[3];
rz(pi*{params2[2]}) q[4];
rx(pi*{params2[3]}) q[4];
rz(pi*{params2[0]}) q[9];
rx(pi*{params2[1]}) q[9];
rz(pi*{params2[2]}) q[10];
rx(pi*{params2[3]}) q[10];
rz(pi*{params1[0]}) q[2];
rx(pi*{params1[1]}) q[2];
rz(pi*{params1[2]}) q[1];
rx(pi*{params1[3]}) q[1];
rz(pi*{params1[0]}) q[8];
rx(pi*{params1[1]}) q[8];
rz(pi*{params1[2]}) q[7];
rx(pi*{params1[3]}) q[7];
rz(pi*{params2[0]}) q[4];
rx(pi*{params2[1]}) q[4];
rz(pi*{params2[2]}) q[5];
rx(pi*{params2[3]}) q[5];
rz(pi*{params2[0]}) q[10];
rx(pi*{params2[1]}) q[10];
rz(pi*{params2[2]}) q[11];
rx(pi*{params2[3]}) q[11];
rz(pi*{params1[0]}) q[1];
rx(pi*{params1[1]}) q[1];
rz(pi*{params1[2]}) q[0];
rx(pi*{params1[3]}) q[0];
rz(pi*{params1[0]}) q[7];
rx(pi*{params1[1]}) q[7];
rz(pi*{params1[2]}) q[6];
rx(pi*{params1[3]}) q[6];
cx q[7],q[6];
cx q[1],q[0];
cx q[10],q[11];
cx q[4],q[5];
cx q[8],q[7];
cx q[2],q[1];
cx q[9],q[10];
cx q[3],q[4];
rz(pi*{params2[4]}) q[3];
rx(pi*{params2[5]}) q[3];
rz(pi*{params2[6]}) q[4];
rx(pi*{params2[7]}) q[4];
rz(pi*{params2[4]}) q[9];
rx(pi*{params2[5]}) q[9];
rz(pi*{params2[6]}) q[10];
rx(pi*{params2[7]}) q[10];
rz(pi*{params1[4]}) q[2];
rx(pi*{params1[5]}) q[2];
rz(pi*{params1[6]}) q[1];
rx(pi*{params1[7]}) q[1];
rz(pi*{params1[4]}) q[8];
rx(pi*{params1[5]}) q[8];
rz(pi*{params1[6]}) q[7];
rx(pi*{params1[7]}) q[7];
rz(pi*{params2[4]}) q[4];
rx(pi*{params2[5]}) q[4];
rz(pi*{params2[6]}) q[5];
rx(pi*{params2[7]}) q[5];
rz(pi*{params2[4]}) q[10];
rx(pi*{params2[5]}) q[10];
rz(pi*{params2[6]}) q[11];
rx(pi*{params2[7]}) q[11];
rz(pi*{params1[4]}) q[1];
rx(pi*{params1[5]}) q[1];
rz(pi*{params1[6]}) q[0];
rx(pi*{params1[7]}) q[0];
rz(pi*{params1[4]}) q[7];
rx(pi*{params1[5]}) q[7];
rz(pi*{params1[6]}) q[6];
rx(pi*{params1[7]}) q[6];
cx q[7],q[6];
cx q[1],q[0];
cx q[10],q[11];
cx q[4],q[5];
cx q[8],q[7];
cx q[2],q[1];
cx q[9],q[10];
cx q[3],q[4];

// evolution operator: TFIM Hamiltonian, g = 0.2, dt = 0.2

rz(pi*-0.5) q[2];
ry(pi*0.986543146) q[2];
rz(pi*0.9999821894) q[1];
ry(pi*0.499579154) q[1];
rz(pi*0.5134568602) q[1];
rz(pi*-0.5) q[8];
ry(pi*0.986543146) q[8];
rz(pi*0.9999821894) q[7];
ry(pi*0.499579154) q[7];
rz(pi*0.5134568602) q[7];
cx q[8],q[7];
cx q[2],q[1];
rz(pi*0.5) q[2];
rz(pi*-0.4996979791) q[1];
ry(pi*-0.999854961) q[2];
ry(pi*0.5001276019) q[1];
rz(pi*0.2454973974) q[1];
rz(pi*0.5) q[8];
rz(pi*-0.4996979791) q[7];
ry(pi*-0.999854961) q[8];
ry(pi*0.5001276019) q[7];
rz(pi*0.2454973974) q[7];
cx q[8],q[7];
cx q[2],q[1];
rz(pi*-0.5) q[2];
rz(pi*-0.9900456689) q[1];
ry(pi*0.0134568602) q[2];
ry(pi*0.0134634469) q[1];
rz(pi*-0.5) q[2];
rz(pi*-0.5099632501) q[1];
rz(pi*-0.5) q[8];
rz(pi*-0.9900456689) q[7];
ry(pi*0.0134568602) q[8];
ry(pi*0.0134634469) q[7];
rz(pi*-0.5) q[8];
rz(pi*-0.5099632501) q[7];

// SWAP tests 3,4
cx q[1],q[4];
cx q[2],q[3];
cx q[7],q[10];
cx q[8],q[9];
h q[1];
h q[7];
measure q[10] -> m2[4];
measure q[4] -> m0[4];
h q[2];
h q[8];
measure q[3] -> m0[3];
measure q[1] -> m1[4];
measure q[2] -> m1[3];
measure q[9] -> m2[3];
measure q[7] -> m3[4];
measure q[8] -> m3[3];
"""
    return openqasm