from qtuum.api_wrappers import QuantinuumAPI as QAPI
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
import cirq
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime
from tqdm import tqdm

from SPSA import minimizeSPSA
from classical import expectation, overlap
from Loschmidt import loschmidt_paper
from ansatz import stateAnsatzXZ
from processing import evolSwapTestRatio
from Hamiltonian import Hamiltonian
from simulation import simulate_noiseless,swapTest,calc_and 

g0, g1 = 1.5, 0.2
max_time = 2
ltimes = np.linspace(0.0, max_time, 800)
correct_ls = [loschmidt_paper(t, g0, g1) for t in ltimes]

paramData = np.load('TMparams100000.npy')
x0 = paramData[0]
x1 = paramData[1]
x2 = paramData[2]

def higherTrotterQasm(param_set1,param_set2):
    '''
    Returns a string of QASM code, with the 2nd order evolution circuit for g=0.2 in TFIM Hamiltonian and dt=0.5
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

// Qubits: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)]
qreg q[8];
creg m0[7]; // Measurement: phi
creg m1[7]; // Measurement: psi



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

// Gate: U applied to qubits 7,6
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


// SWAP test 0
cx q[0],q[7];
h q[0];
measure q[7] -> m0[0];
measure q[0] -> m1[0];
reset q[7];
reset q[0];

// Gate: U applied to qubits 1,3
rz(pi*{params1[0]}) q[1];
rx(pi*{params1[1]}) q[1];
rz(pi*{params1[2]}) q[3];
rx(pi*{params1[3]}) q[3];
cx q[1],q[3];
rz(pi*{params1[4]}) q[1];
rx(pi*{params1[5]}) q[1];
rz(pi*{params1[6]}) q[3];
rx(pi*{params1[7]}) q[3];
cx q[1],q[3];

// Gate: U' applied to qubits 6,4 
rz(pi*{params2[0]}) q[6];
rx(pi*{params2[1]}) q[6];
rz(pi*{params2[2]}) q[4];
rx(pi*{params2[3]}) q[4];
cx q[6],q[4];
rz(pi*{params2[4]}) q[6];
rx(pi*{params2[5]}) q[6];
rz(pi*{params2[6]}) q[4];
rx(pi*{params2[7]}) q[4];
cx q[6],q[4];


// Gate: U' applied to qubits 4,7
rz(pi*{params2[0]}) q[4];
rx(pi*{params2[1]}) q[4];
rz(pi*{params2[2]}) q[7];
rx(pi*{params2[3]}) q[7];
cx q[4],q[7];
rz(pi*{params2[4]}) q[4];
rx(pi*{params2[5]}) q[4];
rz(pi*{params2[6]}) q[7];
rx(pi*{params2[7]}) q[7];
cx q[4],q[7];


// Gate: U applied to 3,0
rz(pi*{params1[0]}) q[3];
rx(pi*{params1[1]}) q[3];
rz(pi*{params1[2]}) q[0];
rx(pi*{params1[3]}) q[0];
cx q[3],q[0];
rz(pi*{params1[4]}) q[3];
rx(pi*{params1[5]}) q[3];
rz(pi*{params1[6]}) q[0];
rx(pi*{params1[7]}) q[0];
cx q[3],q[0];

// Gate: Wo g=0.2 dt = 0.5 applied to qubits 6,4
rz(pi*-0.5) q[6];
ry(pi*-0.0081276962) q[6];
rz(pi*0.987084481) q[4];
ry(pi*0.1788599049) q[4];
rz(pi*0.5152588931) q[4];
cx q[6],q[4];
rx(pi*0.999965997) q[6];
rz(pi*0.5) q[6];
rz(pi*-0.6332094846) q[4];
ry(pi*0.5175620031) q[4];
rz(pi*0.5822619028) q[4];
cx q[6],q[4];
rz(pi*0.5) q[6];
ry(pi*0.9918723038) q[6];
rz(pi*0.5) q[6];
rz(pi*-0.5096005063) q[4];
ry(pi*0.3213702675) q[4];
rz(pi*-0.9948896231) q[4];

// Gate: Wo g=0.2 dt=0.5 applied to qubits 1,3
rz(pi*-0.5) q[1];
ry(pi*-0.0081276962) q[1];
rz(pi*0.987084481) q[3];
ry(pi*0.1788599049) q[3];
rz(pi*0.5152588931) q[3];
cx q[1],q[3];
rx(pi*0.999965997) q[1];
rz(pi*0.5) q[1];
rz(pi*-0.6332094846) q[3];
ry(pi*0.5175620031) q[3];
rz(pi*0.5822619028) q[3];
cx q[1],q[3];
rz(pi*0.5) q[1];
ry(pi*0.9918723038) q[1];
rz(pi*0.5) q[1];
rz(pi*-0.5096005063) q[3];
ry(pi*0.3213702675) q[3];
rz(pi*-0.9948896231) q[3];

// SWAP test 1
cx q[3],q[4];
h q[3];
measure q[4] -> m1[1];
measure q[3] -> m0[1];
reset q[4];
reset q[3];


// Gate: U' applied to qubits 7,5
rz(pi*{params2[0]}) q[7];
rx(pi*{params2[1]}) q[7];
rz(pi*{params2[2]}) q[5];
rx(pi*{params2[3]}) q[5];
cx q[7],q[5];
rz(pi*{params2[4]}) q[7];
rx(pi*{params2[5]}) q[7];
rz(pi*{params2[6]}) q[5];
rx(pi*{params2[7]}) q[5];
cx q[7],q[5];

// Gate: U applied to 0,2
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

// Gate: U' applied to qubits 5,4
rz(pi*{params2[0]}) q[5];
rx(pi*{params2[1]}) q[5];
rz(pi*{params2[2]}) q[4];
rx(pi*{params2[3]}) q[4];
cx q[5],q[4];
rz(pi*{params2[4]}) q[5];
rx(pi*{params2[5]}) q[5];
rz(pi*{params2[6]}) q[4];
rx(pi*{params2[7]}) q[4];
cx q[5],q[4];

// Gate: U applied to qubits 2,3
rz(pi*{params1[0]}) q[2];
rx(pi*{params1[1]}) q[2];
rz(pi*{params1[2]}) q[3];
rx(pi*{params1[3]}) q[3];
cx q[2],q[3];
rz(pi*{params1[4]}) q[2];
rx(pi*{params1[5]}) q[2];
rz(pi*{params1[6]}) q[3];
rx(pi*{params1[7]}) q[3];
cx q[2],q[3];

// Gate: Wo applied to qubits 7,5
rz(pi*-0.5) q[7];
ry(pi*-0.0081276962) q[7];
rz(pi*0.987084481) q[5];
ry(pi*0.1788599049) q[5];
rz(pi*0.5152588931) q[5];
cx q[7],q[5];
rx(pi*0.999965997) q[7];
rz(pi*0.5) q[7];
rz(pi*-0.6332094846) q[5];
ry(pi*0.5175620031) q[5];
rz(pi*0.5822619028) q[5];
cx q[7],q[5];
rz(pi*0.5) q[7];
ry(pi*0.9918723038) q[7];
rz(pi*0.5) q[7];
rz(pi*-0.5096005063) q[5];
ry(pi*0.3213702675) q[5];
rz(pi*-0.9948896231) q[5];

// Gate:Wo applied to qubits 0,2
rz(pi*-0.5) q[0];
ry(pi*-0.0081276962) q[0];
rz(pi*0.987084481) q[2];
ry(pi*0.1788599049) q[2];
rz(pi*0.5152588931) q[2];
cx q[0],q[2];
rx(pi*0.999965997) q[0];
rz(pi*0.5) q[0];
rz(pi*-0.6332094846) q[2];
ry(pi*0.5175620031) q[2];
rz(pi*0.5822619028) q[2];
cx q[0],q[2];
rz(pi*0.5) q[0];
ry(pi*0.9918723038) q[0];
rz(pi*0.5) q[0];
rz(pi*-0.5096005063) q[2];
ry(pi*0.3213702675) q[2];
rz(pi*-0.9948896231) q[2];

// Gate: We applied to 1,2
rz(pi*-0.5) q[1];
ry(pi*-0.0081276962) q[1];
rz(pi*0.987084481) q[2];
ry(pi*0.1788599049) q[2];
rz(pi*0.5152588931) q[2];
cx q[1],q[2];
rx(pi*0.999965997) q[1];
rz(pi*0.5) q[1];
rz(pi*-0.6332094846) q[2];
ry(pi*0.5175620031) q[2];
rz(pi*0.5822619028) q[2];
cx q[1],q[2];
rz(pi*0.5) q[1];
ry(pi*0.9918723038) q[1];
rz(pi*0.5) q[1];
rz(pi*-0.5096005063) q[2];
ry(pi*0.3213702675) q[2];
rz(pi*-0.9948896231) q[2];


// SWAP tests 2,3

cx q[2],q[5];
cx q[1],q[6];
h q[2];
h q[1];
measure q[5] -> m0[2];
measure q[6] -> m0[3];
measure q[2] -> m1[2];
measure q[1] -> m1[3];
reset q[5];
reset q[6];
reset q[2];
reset q[1];

// Gate: U' applied to 4,6
rz(pi*{params2[0]}) q[4];
rx(pi*{params2[1]}) q[4];
rz(pi*{params2[2]}) q[6];
rx(pi*{params2[3]}) q[6];
cx q[4],q[6];
rz(pi*{params2[4]}) q[4];
rx(pi*{params2[5]}) q[4];
rz(pi*{params2[6]}) q[6];
rx(pi*{params2[7]}) q[6];
cx q[4],q[6];

// Gate: U applied to 3,1
rz(pi*{params1[0]}) q[3];
rx(pi*{params1[1]}) q[3];
rz(pi*{params1[2]}) q[1];
rx(pi*{params1[3]}) q[1];
cx q[3],q[1];
rz(pi*{params1[4]}) q[3];
rx(pi*{params1[5]}) q[3];
rz(pi*{params1[6]}) q[1];
rx(pi*{params1[7]}) q[1];
cx q[3],q[1];

// Gate: U' applied to 6,5
rz(pi*{params2[0]}) q[6];
rx(pi*{params2[1]}) q[6];
rz(pi*{params2[2]}) q[5];
rx(pi*{params2[3]}) q[5];
cx q[6],q[5];
rz(pi*{params2[4]}) q[6];
rx(pi*{params2[5]}) q[6];
rz(pi*{params2[6]}) q[5];
rx(pi*{params2[7]}) q[5];
cx q[6],q[5];

// Gate: U applied to 1,2
rz(pi*{params1[0]}) q[1];
rx(pi*{params1[1]}) q[1];
rz(pi*{params1[2]}) q[2];
rx(pi*{params1[3]}) q[2];
cx q[1],q[2];
rz(pi*{params1[4]}) q[1];
rx(pi*{params1[5]}) q[1];
rz(pi*{params1[6]}) q[2];
rx(pi*{params1[7]}) q[2];
cx q[1],q[2];

// Gate: Wo applied to 4,6
rz(pi*-0.5) q[4];
ry(pi*-0.0081276962) q[4];
rz(pi*0.987084481) q[6];
ry(pi*0.1788599049) q[6];
rz(pi*0.5152588931) q[6];
cx q[4],q[6];
rx(pi*0.999965997) q[4];
rz(pi*0.5) q[4];
rz(pi*-0.6332094846) q[6];
ry(pi*0.5175620031) q[6];
rz(pi*0.5822619028) q[6];
cx q[4],q[6];
rz(pi*0.5) q[4];
ry(pi*0.9918723038) q[4];
rz(pi*0.5) q[4];
rz(pi*-0.5096005063) q[6];
ry(pi*0.3213702675) q[6];
rz(pi*-0.9948896231) q[6];

// Gate: Wo applied to 3,1
rz(pi*-0.5) q[3];
ry(pi*-0.0081276962) q[3];
rz(pi*0.987084481) q[1];
ry(pi*0.1788599049) q[1];
rz(pi*0.5152588931) q[1];
cx q[3],q[1];
rx(pi*0.999965997) q[3];
rz(pi*0.5) q[3];
rz(pi*-0.6332094846) q[1];
ry(pi*0.5175620031) q[1];
rz(pi*0.5822619028) q[1];
cx q[3],q[1];
rz(pi*0.5) q[3];
ry(pi*0.9918723038) q[3];
rz(pi*0.5) q[3];
rz(pi*-0.5096005063) q[1];
ry(pi*0.3213702675) q[1];
rz(pi*-0.9948896231) q[1];

// Gate: We applied to 0,1
rz(pi*-0.5) q[0];
ry(pi*-0.0081276962) q[0];
rz(pi*0.987084481) q[1];
ry(pi*0.1788599049) q[1];
rz(pi*0.5152588931) q[1];
cx q[0],q[1];
rx(pi*0.999965997) q[0];
rz(pi*0.5) q[0];
rz(pi*-0.6332094846) q[1];
ry(pi*0.5175620031) q[1];
rz(pi*0.5822619028) q[1];
cx q[0],q[1];
rz(pi*0.5) q[0];
ry(pi*0.9918723038) q[0];
rz(pi*0.5) q[0];
rz(pi*-0.5096005063) q[1];
ry(pi*0.3213702675) q[1];
rz(pi*-0.9948896231) q[1];


// SWAP tests 4,5,6
cx q[3],q[4];
cx q[1],q[6];
cx q[0],q[7];
h q[3];
h q[1];
h q[0];
measure q[4] -> m0[4];
measure q[1] -> m1[5];
measure q[3] -> m1[4];
measure q[6] -> m0[5];
measure q[7] -> m0[6];
measure q[0] -> m1[6];
reset q[4];
reset q[3];
reset q[6];
reset q[7];
reset q[1];
reset q[0];
"""
    return openqasm

def evolutionSimulation(params,set_params,shots, machine, path_to_savefile, timestamp):
    '''
    Cost function for iMPS evolution, submits job to Quantinuum processor then evaluates fidelity density
    Inputs: 
        params: (np.array) parameters calculated by optimiser for timestep t+dt
        set_params: (np.array) parameters of current timestep t
        shots: (int) number of shots to evalute the circuit with
        machine: (str)
            for emulator: machine= 'H1-1E'
            for actual hardware: machine = 'H1-1' or 'H1-2'
        circuit_type: (str)
            'single': one copy of the circuit per circuit
            'double': two copies of the circuit per circuit
            'triple': three copies of the circuit per circuit
        path_to_savefile: (str) the name of the file to save the data to incase of failure
        timestamp: (str) timestamp to append to the filename (calculated in the optimise function)
    '''
    qapi = QAPI(machine=machine)
    
    job_id = qapi.submit_job(higherTrotterQasm(set_params,params), 
                             shots=shots, 
                             machine=machine, # emulator = 'H1-1E'
                             name='higherTrotterCircuitSPSA')
    
    n = 0
    i = 0
    while n != 2000:
        results = qapi.retrieve_job(job_id)
        n = len(results['results']['m0'])+len(results['results']['m1'])
        time.sleep(3)
        i += 1
        if i == 20:
            raise Exception('Results not fully populated after 1 minute')
            
        cost = evolSwapTestRatio(results['results'],7)
        
        with open(str(path_to_savefile)+'_'+timestamp+'.json', 'r+') as file:
            opt_tracker = json.load(file)
            opt_tracker['job_id'].append(job_id)
            opt_tracker['measurement_data'].append(results['results'])
            opt_tracker['exp_cost'].append(cost)
            file.seek(0)
            json.dump(opt_tracker, file, indent=4)
    return -cost

def linExtrap(xa,xb,ya,yb,x):
    y = ya + (x-xa)*(yb-ya)/(xb-xa)    
    return y

def optimise(machine,p0,p1,timestep,path_to_savefile,timin1,ti,tiplus1,iterations=6,shots=1000,a=0.02,c=0.35):
    '''
    Optimise just one time step 
    Uses either the linear fit to make an initial guess of the next parameter
    Then uses SPSA to further optimise the parameter to find the next step of the evolution
    Inputs:
        machine: 
            for emulator: machine= 'H1-1E'
            for actual hardware: machine = 'H1-1' or 'H1-2'?
        p0,p1: the parameters for the previous two time steps (used to do the fitting)
        timestep: which timestep is being evolved from
        path_to_savefile: (str) the name of the file to save the data to incase of failure
        iterations: (default=6) the number of iterations of the SPSA algorithm to run
        startpoint: (default=0) the iteration number to start evaluating from
        shots: (default = 1000) the number of shots with which to evaluate the cost function
        a,c: parameters of SPSA: used to adjust the search parameters (use a = 0.02, c = 0.35 for emulator runs)

    Outputs:
        new_param_guess: array of the updated set of parameters
    Also shows a plot of the Loschmidt echo with this time evolution step plotted on it for comparison. Serves as a check the evolution is progresssing correctly
    '''
    qapi = QAPI(machine=machine) # emulator = 'H1-1E',  machine = 'H1-1' or 'H1-2'?
    now = datetime.now()
    timestamp= now.strftime("%Y%m%dT%H%M%S")
    print('Associated timestamp: ' + timestamp)
    
    def callback_fn(xk):
        cost = expectation(set_params,xk,g,dt)
        loschmidt = overlap(xInit,xk)
        params = xk        
        with open(str(path_to_savefile)+'_'+timestamp+'.json', 'r+') as file:
            opt_tracker = json.load(file)
            opt_tracker['actual_cost'].append(expectation(set_params,xk,g,dt))
            opt_tracker['losch_overlap'].append(overlap(xInit,xk))
            opt_tracker['params'].append(list(xk))
            file.seek(0)
            json.dump(opt_tracker, file, indent=4)
    
    xInit = x0 # the parameters at time t = 0 for calculating the Loschmidt echo
    
    
    data = {
        'job_id': [],
        'measurement_data': [],
        'exp_cost': [],
        'actual_cost': [],
        'losch_overlap': [],
        'params': [],
    }
    json_data = json.dumps(data)
    with open(str(path_to_savefile)+'_'+timestamp+'.json', 'w') as file:
        file.write(json_data)
    
    g = 0.2
    dt = 0.5
    
    set_params = p1
    param_guess = linExtrap(timin1,ti,p0,p1,tiplus1)
    #param_guess = p1

    # SPSA optmisation
    opt = minimizeSPSA(evolutionSimulation,x0 = param_guess, args = [set_params, shots, machine, path_to_savefile, timestamp],  niter = iterations, paired=False,a=a, c=c, callback=callback_fn, restart_point=0)
    
    
    new_param_guess = np.array(opt.x)

    tm_losch = [overlap(xInit,params) for params in paramData]
    plt.plot(ltimes,correct_ls, ls = ':', label = 'exact')
    plt.plot([0.2*i for i in range(len(tm_losch))],-np.log(tm_losch), ls = ':', label = 'tm simulation')
    plt.plot([timestep, timestep+0.5],-np.log([overlap(xInit,p1),overlap(xInit,new_param_guess)]), label = 'emulator simulation', marker = 'x')
    plt.xlim(-0.1,2.1)
    plt.legend()
    plt.xlabel('time')
    plt.ylabel(r'-log$|\langle\psi(t)|\psi(0)\rangle|^2$')
    #plt.minorticks_on()
    #plt.grid(True, which='major')
    #plt.grid(True, which='minor', ls = ':')
    plt.show()
    return new_param_guess

def Wo(g,t):
    n = 2
    H = Hamiltonian(({'ZZ': -1, 'X': g})).to_matrix()
    W = expm(-0.5j*t*H)
    return W

def We(g,t):
    n = 2
    H = Hamiltonian(({'ZZ': -1, 'X': g})).to_matrix()
    W = expm(-1j*t*H)
    return W

def W(g,t):
    n=2
    H = Hamiltonian(({'ZZ': -1, 'X': g})).to_matrix()
    W = expm(-2j*t*H)
    return W
class WeGate0205(cirq.Gate):
    """
    even time evolution operator for TFIM with g=0.2, dt = 0.5
    broken down into CNOT, rx,ry,rz
    """
    def __init__(self):
        super(WeGate0205,self)
        
    def _decompose_(self,qubits):
        return [
            cirq.rz(-np.pi/2).on(qubits[0]),
            cirq.ry(-0.02553391054205689).on(qubits[0]),
            cirq.rz(3.101017353952045).on(qubits[1]),
            cirq.ry(0.5619049631707013).on(qubits[1]),
            cirq.rz(1.6187335532090747).on(qubits[1]),
            cirq.CNOT(qubits[0],qubits[1]),
            cirq.rx(3.1414858300003514).on(qubits[0]),
            cirq.rz(np.pi/2).on(qubits[0]),
            cirq.rz(-1.989286264881903).on(qubits[1]),
            cirq.ry(1.6259689865595808).on(qubits[1]),
            cirq.rz(1.8292297162935665).on(qubits[1]),
            cirq.CNOT(qubits[0],qubits[1]),
            cirq.rz(np.pi/2).on(qubits[0]),
            cirq.ry(3.116058743047737).on(qubits[0]),
            cirq.rz(np.pi/2).on(qubits[0]),
            cirq.rz(-1.600957206750993).on(qubits[1]),
            cirq.ry(1.0096144713440747).on(qubits[1]),
            cirq.rz(-3.1255379311553724).on(qubits[1]),
        ]
            
    
    def num_qubits(self):
        return 2
    
    def _circuit_diagram_info_(self,args):
        return ['We', 'We']
    
class WoGate0205(cirq.Gate):
    """
    odd time evolution operator for TFIM with g=0.2, dt = 0.5
    broken down into CNOT, rx,ry,rz
    """
    def __init__(self):
        super(WoGate0205,self)
        
    def _decompose_(self,qubits):
        return [
            cirq.rz(-np.pi/2).on(qubits[0]),
            cirq.ry(-0.02553391054205689).on(qubits[0]),
            cirq.rz(3.101017353952045).on(qubits[1]),
            cirq.ry(0.5619049631707013).on(qubits[1]),
            cirq.rz(1.6187335532090747).on(qubits[1]),
            cirq.CNOT(qubits[0],qubits[1]),
            cirq.rx(3.1414858300003514) .on(qubits[0]),
            cirq.rz(np.pi/2).on(qubits[0]),
            cirq.rz(-1.989286264881903).on(qubits[1]),
            cirq.ry(1.6259689865595808).on(qubits[1]),
            cirq.rz(1.8292297162935665).on(qubits[1]),
            cirq.CNOT(qubits[0],qubits[1]),
            cirq.rz(np.pi/2).on(qubits[0]),
            cirq.ry(3.116058743047737).on(qubits[0]),
            cirq.rz(np.pi/2).on(qubits[0]),
            cirq.rz(-1.600957206750993).on(qubits[1]),
            cirq.ry(1.0096144713440747).on(qubits[1]),
            cirq.rz(-3.1255379311553724).on(qubits[1]),
        ]
    
    def num_qubits(self):
        return 2
    
    def _circuit_diagram_info_(self,args):
        return ['Wo', 'Wo']
    

def secondOrderCircuit(params1,params2,g,t,show=False):
    '''
    2nd order evolution circuit with mid circuit measurement and reset
    '''
    psi = stateAnsatzXZ(params1)
    phi = stateAnsatzXZ(params2)
    
    circuit = cirq.Circuit()
    qubits = [cirq.GridQubit(i,0) for i in range(8)]
    noOfQubits= len(qubits)
    
    circuit.append(psi.on(qubits[0],qubits[1]))
    circuit.append(psi.on(qubits[7],qubits[6]))
    circuit.append(cirq.CNOT(qubits[0],qubits[7]))
    circuit.append(cirq.H(qubits[0]))
    circuit.append(cirq.measure(qubits[0], key='psi'+str(0)))
    circuit.append(cirq.measure(qubits[7], key='phi'+str(0)))
    circuit.append(cirq.reset(qubits[0]))
    circuit.append(cirq.reset(qubits[7]))
    
    circuit.append(psi.on(qubits[1],qubits[3]))
    circuit.append(psi.on(qubits[3],qubits[0]))
    circuit.append(WoGate0205().on(qubits[1],qubits[3]))
    circuit.append(phi.on(qubits[6],qubits[4]))
    circuit.append(phi.on(qubits[4],qubits[7]))
    circuit.append(WoGate0205().on(qubits[6],qubits[4]))
    circuit.append(cirq.CNOT(qubits[3],qubits[4]))
    circuit.append(cirq.H(qubits[3]))
    circuit.append(cirq.measure(qubits[3], key='psi'+str(1)))
    circuit.append(cirq.measure(qubits[4], key='phi'+str(1)))
    circuit.append(cirq.reset(qubits[3]))
    circuit.append(cirq.reset(qubits[4]))
    
    circuit.append(psi.on(qubits[0],qubits[2]))
    circuit.append(psi.on(qubits[2],qubits[3]))
    circuit.append(WoGate0205().on(qubits[0],qubits[2]))
    circuit.append(WeGate0205().on(qubits[1],qubits[2]))
    circuit.append(phi.on(qubits[7],qubits[5]))
    circuit.append(phi.on(qubits[5],qubits[4]))
    circuit.append(WoGate0205().on(qubits[7],qubits[5]))
    circuit.append(cirq.CNOT(qubits[2],qubits[5]))
    circuit.append(cirq.H(qubits[2]))
    circuit.append(cirq.CNOT(qubits[1],qubits[6]))
    circuit.append(cirq.H(qubits[1]))
    circuit.append(cirq.measure(qubits[2], key='psi'+str(2)))
    circuit.append(cirq.measure(qubits[5], key='phi'+str(2)))
    circuit.append(cirq.reset(qubits[2]))
    circuit.append(cirq.reset(qubits[5]))
    circuit.append(cirq.measure(qubits[1], key='psi'+str(3)))
    circuit.append(cirq.measure(qubits[6], key='phi'+str(3)))
    circuit.append(cirq.reset(qubits[1]))
    circuit.append(cirq.reset(qubits[6]))
    
    circuit.append(psi.on(qubits[3],qubits[1]))
    circuit.append(psi.on(qubits[1],qubits[2]))
    circuit.append(WoGate0205().on(qubits[3],qubits[1]))
    circuit.append(WeGate0205().on(qubits[0],qubits[1]))
    circuit.append(phi.on(qubits[4],qubits[6]))
    circuit.append(phi.on(qubits[6],qubits[5]))
    circuit.append(WoGate0205().on(qubits[4],qubits[6]))
    circuit.append(cirq.CNOT(qubits[3],qubits[4]))
    circuit.append(cirq.H(qubits[3]))
    circuit.append(cirq.CNOT(qubits[1],qubits[6]))
    circuit.append(cirq.H(qubits[1]))
    circuit.append(cirq.CNOT(qubits[0],qubits[7]))
    circuit.append(cirq.H(qubits[0]))
    circuit.append(cirq.measure(qubits[3], key='psi'+str(4)))
    circuit.append(cirq.measure(qubits[4], key='phi'+str(4)))
    circuit.append(cirq.reset(qubits[3]))
    circuit.append(cirq.reset(qubits[4]))
    circuit.append(cirq.measure(qubits[1], key='psi'+str(5)))
    circuit.append(cirq.measure(qubits[6], key='phi'+str(5)))
    circuit.append(cirq.reset(qubits[1]))
    circuit.append(cirq.reset(qubits[6]))
    circuit.append(cirq.measure(qubits[0], key='psi'+str(6)))
    circuit.append(cirq.measure(qubits[7], key='phi'+str(6)))
    circuit.append(cirq.reset(qubits[0]))
    circuit.append(cirq.reset(qubits[7]))
    
    if show is True:
        print(circuit.to_text_diagram(transpose=True))

    return circuit

    return circuit

def secondOrderCircuitFast(params1,params2,g,t,show=False):
    '''
    2nd order evolution circuit without mid circuit measurement and resets
    - instead just keeps the qubit and uses a new one so can measure all at the end
    - only good for in simulation
    '''
    psi = stateAnsatzXZ(params1)
    phi = stateAnsatzXZ(params2)
    
    circuit = cirq.Circuit()
    qubits = [cirq.GridQubit(i,0) for i in range(16)]
    noOfQubits= len(qubits)
    
    circuit.append(psi.on(qubits[0],qubits[1]))
    circuit.append(psi.on(qubits[1],qubits[2]))
    circuit.append(psi.on(qubits[2],qubits[3]))
    
    circuit.append(WoGate0205().on(qubits[1],qubits[2]))
    
    circuit.append(psi.on(qubits[3],qubits[4]))
    circuit.append(psi.on(qubits[4],qubits[5]))
    
    circuit.append(WoGate0205().on(qubits[3],qubits[4]))
    circuit.append(WeGate0205().on(qubits[1],qubits[4]))
    
    circuit.append(psi.on(qubits[5],qubits[6]))
    circuit.append(psi.on(qubits[6],qubits[7]))
    
    circuit.append(WoGate0205().on(qubits[5],qubits[6]))
    circuit.append(WeGate0205().on(qubits[3],qubits[6]))
    
    circuit.append(psi.on(qubits[15],qubits[14]))
    circuit.append(phi.on(qubits[14],qubits[13]))
    circuit.append(phi.on(qubits[13],qubits[12]))
    
    circuit.append(WoGate0205().on(qubits[14],qubits[13]))
    
    circuit.append(phi.on(qubits[12],qubits[11]))
    circuit.append(phi.on(qubits[11],qubits[10]))
    
    circuit.append(WoGate0205().on(qubits[12],qubits[11]))
    
    circuit.append(phi.on(qubits[10],qubits[9]))
    circuit.append(phi.on(qubits[9],qubits[8]))
    
    circuit.append(WoGate0205().on(qubits[10],qubits[9]))
    
    circuit.append(cirq.CNOT(qubits[0],qubits[15]))
    circuit.append(cirq.H(qubits[0]))
    circuit.append(cirq.CNOT(qubits[2],qubits[13]))
    circuit.append(cirq.H(qubits[2]))
    circuit.append(cirq.CNOT(qubits[1],qubits[14]))
    circuit.append(cirq.H(qubits[1]))
    circuit.append(cirq.CNOT(qubits[4],qubits[11]))
    circuit.append(cirq.H(qubits[4]))
    circuit.append(cirq.CNOT(qubits[3],qubits[12]))
    circuit.append(cirq.H(qubits[3]))
    circuit.append(cirq.CNOT(qubits[6],qubits[9]))
    circuit.append(cirq.H(qubits[6]))
    circuit.append(cirq.CNOT(qubits[5],qubits[10]))
    circuit.append(cirq.H(qubits[5]))
    
    circuit.append(cirq.measure(qubits[0], key='psi'+str(0)))
    circuit.append(cirq.measure(qubits[15], key='phi'+str(0)))
    circuit.append(cirq.measure(qubits[2], key='psi'+str(1)))
    circuit.append(cirq.measure(qubits[13], key='phi'+str(1)))
    circuit.append(cirq.measure(qubits[1], key='psi'+str(2)))
    circuit.append(cirq.measure(qubits[14], key='phi'+str(2)))
    circuit.append(cirq.measure(qubits[4], key='psi'+str(3)))
    circuit.append(cirq.measure(qubits[11], key='phi'+str(3)))
    circuit.append(cirq.measure(qubits[3], key='psi'+str(4)))
    circuit.append(cirq.measure(qubits[12], key='phi'+str(4)))
    circuit.append(cirq.measure(qubits[6], key='psi'+str(5)))
    circuit.append(cirq.measure(qubits[9], key='phi'+str(5)))
    circuit.append(cirq.measure(qubits[5], key='psi'+str(6)))
    circuit.append(cirq.measure(qubits[10], key='phi'+str(6)))
    
    if show is True:
        print(circuit.to_text_diagram(transpose=True))

    return circuit

def firstOrderCircuitFast(params1, params2, g,t,length):
    '''
    1st order evolution circuit without mid circuit measure resets and with a more efficient representation of the W gate and ansatz
    - instead just keeps the qubit and uses a new one so can measure all at the end
    - only good for in simulation
    '''
    psi = stateAnsatzXZ(params1)
    phi = stateAnsatzXZ(params2)
    W1 = W(g,t)
    
    
    # length = 2
    circuit = cirq.Circuit()
    qubits = [cirq.GridQubit(i,0) for i in range(4+length*4)]
    noOfQubits = len(qubits)
    
    circuit.append(psi.on(qubits[0],qubits[1]))
    circuit.append(psi.on(qubits[noOfQubits-1],qubits[noOfQubits-2]))
    #circuit.append(psi.on(qubits[noOfQubits-2],qubits[noOfQubits-1]))
    circuit.append(cirq.CNOT(qubits[0],qubits[noOfQubits-1]))
    circuit.append(cirq.H(qubits[0]))
    
    for i in range(1,length+2,2):
        circuit.append(psi.on(qubits[i], qubits[i+1]))
        circuit.append(phi.on(qubits[noOfQubits-(i+1)],qubits[noOfQubits-(i+2)]))
        circuit.append(psi.on(qubits[i+1], qubits[i+2]))
        circuit.append(phi.on(qubits[noOfQubits-(i+2)],qubits[noOfQubits-(i+3)]))
        circuit.append(cirq.ion.two_qubit_matrix_to_ion_operations(qubits[i],qubits[i+1],W1))

        circuit.append(cirq.CNOT(qubits[i],qubits[noOfQubits-(i+1)]))
        circuit.append(cirq.H(qubits[i]))
        circuit.append(cirq.CNOT(qubits[i+1],qubits[noOfQubits-(i+2)]))
        circuit.append(cirq.H(qubits[i+1]))

    for i in range(2*length+1):
        circuit.append(cirq.measure(qubits[i], key='psi'+str(i)))
        circuit.append(cirq.measure(qubits[noOfQubits-(i+1)], key='phi'+str(i)))
        
    return circuit

def cost_func(params,set_params,dt,shots,order):
    n = 0
    i = 0
    g = 0.2
    #dt = 0.2
    tm_cost = expectation(set_params,params,g,dt)
    if order == 2:
        cost = swapTest(calc_and(simulate_noiseless(secondOrderCircuitFast(set_params,params,g,dt),shots)),7)
    if order == 1:
        cost = swapTest(calc_and(simulate_noiseless(firstOrderCircuitFast(set_params,params,g,dt,2),shots)),5)
    return -cost

def fullRun(xInit,p0,p1,path_to_savefile,steps = 9,dt=0.2,iterations=6,shots=1000,a=0.025,c=0.5,showPlot=False):
    '''
    
    '''
    now = datetime.now()
    timestamp = now.strftime("%Y%m%dT%H%M%S")
    paramData = np.load('TMparams100000.npy')
    
    t=[0,0.2,0.4]+[0.4+i*dt for i in range(1,steps-1)]
    params1=[xInit,p0,p1]
    params2=[xInit,p0,p1]
    
    for i in tqdm(range(2,steps)):
        g = 0.2
        timestep = i*dt
        #t.append(timestep)
        
        
        #param_guess1 = linExtrap(t[i-1],t[i],params1[i-1],params1[i],t[i+1])
        #print(t[i-1],t[i],t[i+1])
        #opt1 = minimizeSPSA(cost_func,x0=param_guess1, args = [params1[i],dt,shots,1],niter=iterations, paired=False,a=a,c=c,restart_point=0)
        #newParam1 = np.array(opt1.x)
        #newParam1 = optimise(params1[i-2],params1[i-1],dt=dt,iterations=iterations,shots=shots,a=a,c=c,order=1)
        #params1.append(newParam1)
        
        param_guess2 = linExtrap(t[i-1],t[i],params2[i-1],params2[i],t[i+1])
        opt2 = minimizeSPSA(cost_func,x0=param_guess2, args = [params2[i],dt,shots,2],niter=iterations, paired=False,a=a,c=c,restart_point=0)
        newParam2 = np.array(opt2.x)
        #newParam2 = optimise(params2[i-2],params2[i-1],dt=dt,iterations=iterations,shots=shots,a=a,c=c,order=2)
        params2.append(newParam2)
       
        
    #np.save('paper_trotter_data/PARAMS_order_1_dt_'+str(dt)+'_its_'+str(iterations)+'_shots_'+str(shots)+'_a_'+str(a)+'_c_'+str(c)+'_'+timestamp+'.npy',params1)
    np.save(str(path_to_savefile)+'_PARAMS_order_2_dt_'+str(dt)+'_its_'+str(iterations)+'_shots_'+str(shots)+'_a_'+str(a)+'_c_'+str(c)+'_'+timestamp+'.npy',params2)
    
    tm_losch = [-np.log(overlap(xInit,p)) for p in paramData]
    #losch_values1 = [-np.log(overlap(xInit,p)) for p in params1]
    losch_values2 = [-np.log(overlap(xInit,p)) for p in params2]

    #np.save('paper_trotter_data/LOSCH_order_1_dt_'+str(dt)+'_its_'+str(iterations)+'_shots_'+str(shots)+'_a_'+str(a)+'_c_'+str(c)+'_'+timestamp+'.npy',losch_values1)
    np.save(str(path_to_savefile)+'_LOSCH_order_2_dt_'+str(dt)+'_its_'+str(iterations)+'_shots_'+str(shots)+'_a_'+str(a)+'_c_'+str(c)+'_'+timestamp+'.npy',losch_values2)
    
    print(timestamp)
    fig = plt.figure()
    plt.plot(ltimes,correct_ls,ls = ':',c='r',label='exact')
    plt.plot([0.2*i for i in range(len(tm_losch))],tm_losch,ls=':',c='orange',label = 'TM simulation')
    
    #plt.plot(t,losch_values1,marker = 'x',label = '1st order noiseless simulation')
    plt.plot(t,losch_values2,'x',label = '2nd order noiseless simulation')
    
    plt.xlim(-0.1,steps*dt+0.1)
    plt.legend()#bbox_to_anchor=(1.1, 1.05))
    plt.xlabel('time')
    plt.ylabel(r'-log$|\langle\psi(t)|\psi(0)\rangle|^2$')
    plt.minorticks_on()
    plt.grid(True, which='major')
    plt.grid(True, which='minor', ls = ':')
    plt.savefig(str(path_to_savefile)+'_loschmidtEcho_'+timestamp+'.png')
    if showPlot == True:
        plt.show()
    if showPlot == False:
        plt.close(fig)

    #return t,params1,params2,losch_values1,losch_values2,timestamp
    return t,params2,losch_values2,timestamp