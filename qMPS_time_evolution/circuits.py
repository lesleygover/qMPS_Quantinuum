import numpy as np

def UGate(params,qubits):
    '''
    Produces openqasm string for parametrised unitary gate
    =============
    Inputs: 
        params (np.array): parameters of the unitary
        qubits (list): list number which two qubits the gate should act on
    =============
    Outputs: 
        UGate_qasm (str): openqasm string of parametrised unitary
    '''
    UGate_qasm = f"""
// Gate: state ansatz gate applied to qubits {qubits[0]} {qubits[1]}

rz(pi*{params[0]}) q[{qubits[0]}];
rx(pi*{params[1]}) q[{qubits[0]}];
rz(pi*{params[2]}) q[{qubits[1]}];
rx(pi*{params[3]}) q[{qubits[1]}];
cx q[{qubits[0]}],q[{qubits[1]}];
rz(pi*{params[4]}) q[{qubits[0]}];
rx(pi*{params[5]}) q[{qubits[0]}];
rz(pi*{params[6]}) q[{qubits[1]}];
rx(pi*{params[7]}) q[{qubits[1]}];
cx q[{qubits[0]}],q[{qubits[1]}];
"""
    return UGate_qasm

def swapMeasureQASM(qubits,cbits,swap_test_number):
    '''
    Produces openqasm string for swap test
    =============
    Inputs: 
        qubits (list): list numbering which two qubits the gates should act on
        cbits (list): list numbering which classical bit to apply the gates to
        swap_test_number (int): which number swap test this is
    =============
    Outputs: 
        swap_qasm (str): openqasm string of the swap test
    '''
    swap_qasm = f"""
// SWAP test

cx q[{qubits[0]}],q[{qubits[1]}];
h q[{qubits[0]}];
measure q[{qubits[1]}] -> m{cbits[0]}[{swap_test_number}];
measure q[{qubits[0]}] -> m{cbits[1]}[{swap_test_number}];
reset q[{qubits[1]}];
reset q[{qubits[0]}];
"""
    return swap_qasm

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

def evolutionCircuit(param_set1,param_set2,circuit_type):
    '''
    Returns a string of QASM code, with the evolution circuit for g=0.2 in TFIM Hamiltonian and dt=0.2
    =============
    Inputs:
        param_set1 (np.array): An array of the first set of parameters
        param_set2 (np.array): An array of the second set of parameters
        circuit_type (str): 
            'single': one copy of the circuit per circuit
            'double': two copies of the circuit per circuit
            'triple': three copies of the circuit per circuit
    =============
    Outputs:
        openqasm (str): openqasm string of parametrised circuit
    '''
    params1 = []
    for array1 in param_set1:
        newArray1 = array1/np.pi
        params1.append(newArray1)

    params2 = []
    for array2 in param_set2:
        newArray2 = array2/np.pi
        params2.append(newArray2)

    single_circuit = UGate(params1,[0,1]) + UGate(params1,[5,4]) + swapMeasureQASM([0,5],[0,1],0) + UGate(params2,[4,5])+UGate(params1,[1,0])+UGate(params2,[5,3])+UGate(params1,[0,2])+evolOpQASM([1,0])+swapMeasureQASM([1,4],[0,1],1)+swapMeasureQASM([0,5],[0,1],2)+UGate(params2,[3,4])+UGate(params1,[2,1])+UGate(params2,[4,5])+UGate(params1,[1,0])+evolOpQASM([2,1])+swapMeasureQASM([2,3],[0,1],3)+swapMeasureQASM([1,4],[0,1],4)
    double_circuit = single_circuit+"""
// 2nd copy of evolution circuit with 2 iterations of the transfer matrix
    """+UGate(params1,[6,7]) + UGate(params1,[11,10]) + swapMeasureQASM([6,11],[2,3],0) + UGate(params2,[10,11])+UGate(params1,[7,6])+UGate(params2,[11,9])+UGate(params1,[6,8])+evolOpQASM([7,6])+swapMeasureQASM([7,10],[2,3],1)+swapMeasureQASM([6,11],[2,3],2)+UGate(params2,[9,10])+UGate(params1,[8,7])+UGate(params2,[10,11])+UGate(params1,[7,6])+evolOpQASM([8,7])+swapMeasureQASM([8,9],[2,3],3)+swapMeasureQASM([7,10],[2,3],4)
    triple_circuit = double_circuit+"""
// 3rd copy of evolution circuit with 2 iterations of the transfer matrix
    """+UGate(params1,[12,13]) + UGate(params1,[17,16]) + swapMeasureQASM([12,17],[4,5],0) + UGate(params2,[16,17])+UGate(params1,[13,12])+UGate(params2,[17,15])+UGate(params1,[12,14])+evolOpQASM([13,12])+swapMeasureQASM([13,16],[4,5],1)+swapMeasureQASM([12,17],[4,5],2)+UGate(params2,[15,16])+UGate(params1,[14,13])+UGate(params2,[16,17])+UGate(params1,[14,13])+evolOpQASM([2,1])+swapMeasureQASM([14,15],[4,5],3)+swapMeasureQASM([13,16],[4,5],4)
    
    single_open = """
OPENQASM 2.0;
include "qelib1.inc";

// evolution circuit with 2 iterations of the transfer matrix

// Qubits: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
qreg q[6];
creg m0[5];  // Measurement: phi
creg m1[5];  // Measurement: psi
"""
    double_open = """
OPENQASM 2.0;
include "qelib1.inc";

// 1st copy of evolution circuit with 2 iterations of the transfer matrix

// Qubits: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6,0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0)]
qreg q[12];
creg m0[5];  // Measurement: phi
creg m1[5];  // Measurement: psi
creg m2[5];  // Measurement: phi2
creg m3[5];  // Measurement: psi2
"""
    triple_open = """
OPENQASM 2.0;
include "qelib1.inc";

// 1st copy of evolution circuit with 2 iterations of the transfer matrix

// Qubits: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6,0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12,0), (13, 0), (14, 0), (15, 0), (16, 0), (17, 0)]
qreg q[18];
creg m0[5];  // Measurement: phi
creg m1[5];  // Measurement: psi
creg m2[5];  // Measurement: phi2
creg m3[5];  // Measurement: psi2
creg m4[5];  // Measurement: phi3
creg m5[5];  // Measurement: psi3
"""
    if circuit_type == 'single':
        openqasm = single_open+single_circuit
    elif circuit_type == 'double':
        openqasm = double_open+double_circuit
    elif circuit_type == 'triple':
        openqasm = triple_open+triple_circuit
    else:
        raise Exception("circuit_type must be one of: 'single', 'double', or 'triple'")
    return openqasm

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

def higherTrotterQasm(param_set1,param_set2):
    '''
    Returns a string of QASM code, with the 2nd order evolution circuit for g=0.2 in TFIM Hamiltonian and dt=0.5
    =============
    Inputs:
        param_set1 (np.array): An array of the first set of parameters
        param_set2 (np.array): An array of the second set of parameters
        circuit_type (str): 
            'single': one copy of the circuit per circuit
            'double': two copies of the circuit per circuit
            'triple': three copies of the circuit per circuit
    =============
    Outputs:
        openqasm (str): openqasm string of parametrised circuit
    '''
    params1 = []
    for array1 in param_set1:
        newArray1 = array1/np.pi
        params1.append(newArray1)

    params2 = []
    for array2 in param_set2:
        newArray2 = array2/np.pi
        params2.append(newArray2)

    circuit = UGate(params1,[0,1])+UGate(params1,[7,6])+swapMeasureQASM([0,7],[0,1],0)+UGate(params1,[1,3])+UGate(params2,[6,4])+UGate(params2,[4,7])+UGate(params1,[3,0])+evolOp2ndOrderQASM([6,4])+evolOp2ndOrderQASM([1,3])+swapMeasureQASM([3,4],[0,1],1)+UGate(params2,[7,5])+UGate(params1,[0,2])+UGate(params2,[5,4])+UGate(params1,[2,3])+evolOp2ndOrderQASM([7,5])+evolOp2ndOrderQASM([0,2])+evolOp2ndOrderQASM([1,2])+swapMeasureQASM([2,5],[0,1],2)+swapMeasureQASM([1,6],[0,1],3)+UGate(params2,[4,6])+UGate(params1,[3,1])+UGate(params2,[6,5])+UGate(params1,[1,2])+evolOp2ndOrderQASM([4,6])+evolOp2ndOrderQASM([3,1])+evolOp2ndOrderQASM([0,1])+swapMeasureQASM([3,4],[0,1],4)+swapMeasureQASM([1,6],[0,1],5)+swapMeasureQASM([0,7],[0,1],6)

    openqasm = f"""
OPENQASM 2.0;
include "qelib1.inc";

// Qubits: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)]
qreg q[8];
creg m0[7]; // Measurement: phi
creg m1[7]; // Measurement: psi
""" + circuit
    
    return openqasm