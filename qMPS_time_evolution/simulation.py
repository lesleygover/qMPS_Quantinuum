import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad
import cirq
import time
import json
from datetime import datetime
from tqdm import tqdm

from SPSA import minimizeSPSA
from classical import expectation, overlap, param_to_tensor, linFit
from Loschmidt import loschmidt_paper
from ansatz import stateAnsatzXZ, timeLikeAnsatzXZ
from Hamiltonian import evolution_op, evolution_circuit_op,Wgate0202

def simulate_noiseless(circuit, shots):
    """
    Simulate a circuit without noise
    Inputs:
        circuit: cirq circuit to simulate
        shots: number of shots to run of the circuit
    """
    sim = cirq.Simulator()
    results = sim.run(circuit, repetitions=shots)
    return results

class SWAP_measure(cirq.Gate):
	""" SWAP test and measurement step for circuits in time """
	def __init__(self, index):
		self.index = index

	def _decompose_(self, qubits):
		return [
				cirq.CNOT(qubits[0], qubits[1]),
				cirq.H(qubits[0]),
				cirq.measure(qubits[0], key='psi'+str(self.index)),
				cirq.measure(qubits[1], key='phi'+str(self.index)),
                cirq.ops.reset(qubits[0]),
				cirq.ops.reset(qubits[1]),
            ]

	def num_qubits(self):
		return 2

	def _circuit_diagram_info_(self, args):
		return ['S','S']
	
def fastEvolutionCircuit(params1, params2,length):
    '''
    Timelike evolution circuit without mid circuit measurement and resets
    - instead just keeps the qubit and uses a new one so can measure all at the end
    - speeds up cirq simulation
	Inputs:
        params1,params2: (np.array) array of parameters to parameterise the two sets of iMPS unitaries
		length: (int) number of repeats of the transfer matrix
	Outputs:    
        circuit: cirq circuit for evaluting the evolution cost function
    '''
    psi = stateAnsatzXZ(params1)
    phi = stateAnsatzXZ(params2)
    W = Wgate0202()
    
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
        circuit.append(Wgate0202().on(qubits[i],qubits[i+1]))

        circuit.append(cirq.CNOT(qubits[i],qubits[noOfQubits-(i+1)]))
        circuit.append(cirq.H(qubits[i]))
        circuit.append(cirq.CNOT(qubits[i+1],qubits[noOfQubits-(i+2)]))
        circuit.append(cirq.H(qubits[i+1]))

    for i in range(2*length+1):
        circuit.append(cirq.measure(qubits[i], key='psi'+str(i)))
        circuit.append(cirq.measure(qubits[noOfQubits-(i+1)], key='phi'+str(i)))
        
    return circuit

def calc_and(results):
	'''
	Calculate the bitwise AND of the measurement bitstrings
	Inputs:
        results: result data from simulation
	Outputs:
        bitstrings: bitstrings of the bitwise ANDs
	
	'''
	measure = results.measurements
	length = int(len(measure)/2)

	ands = {}
	for i in range(length):
		curr_psi = [item for sub in measure['psi'+str(i)] for item in sub]
		curr_phi = [item for sub in measure['phi'+str(i)] for item in sub]
		ands[i] = np.bitwise_and(curr_psi, curr_phi).flatten()

	# Choose to return Z as a matrix for ease of post processing
	# (arguably there is no ease)
	bitstrings = (np.array([ands[i] for i in range(length)])).T

	return bitstrings


def swapTest(bitstrings,length):
	"""
    Perform the swap test on the circuit results
	Inputs:
        bitstrings: bitstrings of bitwiseANDs (output of calc_and)
		length: how many pairs of measurements there are per circuit
	Outputs:
        overlap: returns the fidelity density calculated using the swap test
    """
	shots = np.size(bitstrings,0)
	parity = 0
	for shot in range(shots):
		if (sum(bitstrings[shot, :length]) % 2 == 0):
			parity += 1
	prob = parity/shots
	overlap = np.sqrt((np.abs(2*prob - 1)))
	#overlap = 2*prob - 1
	return overlap

def swapTestRatio(bitstrings, length):
	"""
    Calculates ratio of the swap tests on the circuit results for length and length-1
	Inputs:
        bitstrings: bitstrings of bitwiseANDs (output of calc_and)
		length: how many pairs of measurements there are per circuit
	Outputs:
        overlap: returns the fidelity density calculated using the swap test
    """
	shots = np.size(bitstrings,0)
	#length = len(bitstrings[0])
	parity_full_bitstring = 0
	parity_minus_one = 0

	for shot in range(shots):
		if (sum(bitstrings[shot, :length]) % 2 == 0):
			parity_full_bitstring += 1
		if (sum(bitstrings[shot, :length-1]) % 2 == 0):
			parity_minus_one += 1

	prob_full = parity_full_bitstring/shots
	prob_minus = parity_minus_one/shots

	overlap_full = np.sqrt(np.abs(2*prob_full - 1))
	overlap_minus = np.sqrt(np.abs(2*prob_minus - 1))

	overlap = overlap_full/overlap_minus

	return overlap**2

def evolSwapTestRatio(bitstrings, length):
    """
    Calculates ratio of the swap tests on the circuit results for length and length-2
    - between length and length-2 as is a two-site transfers matrix with the evolution operator
    Inputs:
        bitstrings: bitstrings of bitwiseANDs (output of calc_and)
        length: how many pairs of measurements there are per circuit
    Outputs:
        overlap: returns the fidelity density calculated using the swap test
    """
    shots = np.size(bitstrings,0)
    #length = len(bitstrings[0])
    parity_full_bitstring = 0
    parity_minus_two = 0

    for shot in range(shots):
            if (sum(bitstrings[shot, :length]) % 2 == 0):
                    parity_full_bitstring += 1
            if (sum(bitstrings[shot, :length-2]) % 2 == 0):
                    parity_minus_two += 1

    prob_full = parity_full_bitstring/shots
    prob_minus = parity_minus_two/shots

    overlap_full = np.sqrt(np.abs(2*prob_full - 1))
    overlap_minus = np.sqrt(np.abs(2*prob_minus - 1))

    overlap = overlap_full/overlap_minus

    return overlap**2


g0, g1 = 1.5, 0.2
max_time = 2
ltimes = np.linspace(0.0, max_time, 800)
correct_ls = [loschmidt_paper(t, g0, g1) for t in ltimes]

paramData = np.load('TMparams100000.npy')
x0 = paramData[0]
x1 = paramData[1]
x2 = paramData[2]

def cost_func(params,set_params,path_to_savefile,timestamp,shots):
    '''
    Cost function for iMPS evolution with noiseless circuit simulation through cirq
    Inputs:
        params: (np.array) parameters calculated by optimiser for timestep t+dt
        set_params: (np.array) parameters of current timestep t
        path_to_savefile: (str) the name of the file to save the data to incase of failure
        timestamp: (str) timestamp to append to filename (calculated in the optimise function)
        shots: (int) number of shots to evalute the circuit with
    '''
    n = 0
    i = 0
    g = 0.2
    dt = 0.2
    length = 2
    swapTestLength = 5
    tm_cost = expectation(set_params,params,g,dt)
    cost = evolSwapTestRatio(calc_and(simulate_noiseless(fastEvolutionCircuit(set_params,params,g,dt,length),shots)),swapTestLength)
    
    with open(str(path_to_savefile)+'_'+timestamp+'.json', 'r+') as file:
        opt_tracker = json.load(file)
        opt_tracker['exp_cost'].append(cost)
        opt_tracker['actual_cost'].append(tm_cost)
        file.seek(0)
        json.dump(opt_tracker, file, indent=4)
    
    return -cost

def transfer_matrix_cost_func(params,set_params,path_to_savefile,timestamp,shots=0):
    '''
    Cost function for iMPS evolution with exact in ansatz transfer matrix simulation through cirq
    Inputs:
        params: (np.array) parameters calculated by optimiser for timestep t+dt
        set_params: (np.array) parameters of current timestep t
        path_to_savefile: (str) the name of the file to save the data to incase of failure
        timestamp: (str) timestamp to append to filename (calculated in the optimise function)
        shots: just included so matches format of ciruit simulation(set=0 but doesn't matter)
    '''
    g = 0.2
    t = 0.2
    W = evolution_op(g,t)
    cost = expectation(set_params,params,g,t)
    with open(str(path_to_savefile)+'_transfer_matrix_'+timestamp+'.json', 'r+') as file:
        opt_tracker = json.load(file)
        opt_tracker['cost'].append(cost)
        file.seek(0)
        json.dump(opt_tracker, file, indent=4)
    return -cost

def optimiseForCompleteRunSim(p0,p1,path_to_savefile,cf=cost_func,iterations=6,shots=1000,a=0.02,c=0.35):
    '''
    Optimise just one time step 
    Uses either the linear fit to make an initial guess of the next parameter
    Then uses SPSA to further optimise the parameter to find the next step of the evolution
    Inputs:
        p0,p1: the parameters for the previous two time steps (used to do the fitting)
        cf: which cost function to use
            transfer_matrix_cost_func: transfer matrix simulation
            cost_func: noiseless circuit simulation 
        path_to_savefile: (str) the name of the file to save the data to incase of failure
        iterations: (default=6) the number of iterations of the SPSA algorithm to run
        shots: (default = 1000) the number of shots with which to evaluate the cost function
        a,c: parameters of SPSA: used to adjust the search parameters (use a = 0.02, c = 0.35 for emulator runs)

    Outputs:
        new_param_guess: array of the updated set of parameters
    '''
    
    now = datetime.now()
    timestamp = now.strftime("%Y%m%dT%H%M%S")

    def callback_fn(xk):
        cost = expectation(set_params,xk,g,dt)
        loschmidt = overlap(xInit,xk)
        params = xk        
        with open(str(path_to_savefile)+'_'+timestamp+'.json', 'r+') as file:
            opt_tracker = json.load(file)
            opt_tracker['final_cost'].append(expectation(set_params,xk,g,dt))
            opt_tracker['losch_overlap'].append(-np.log(overlap(xInit,xk)))
            opt_tracker['params'].append(list(xk))
            file.seek(0)
            json.dump(opt_tracker, file, indent=4)
    
    xInit = x0 # the parameters at time t = 0 for calculating the Loschmidt echo
    
    
    data = {
        'job_id': [],
        'measurement_data': [],
        'exp_cost': [],
        'actual_cost': [],
        'final_cost': [],
        'losch_overlap': [],
        'params': [],
    }
    json_data = json.dumps(data)
    with open(str(path_to_savefile)+'_'+timestamp+'.json', 'w') as file:
        file.write(json_data)
    
    g = 0.2
    dt = 0.2
    
    set_params = p1
    param_guess = linFit(p0,p1)
    #param_guess = p1

    # SPSA optmisation
    opt = minimizeSPSA(cf,x0 = param_guess, args = [set_params, path_to_savefile, timestamp, shots],  niter = iterations, paired=False,a=a, c=c, callback=callback_fn, restart_point=0)
    
    
    new_param_guess = np.array(opt.x)


    return new_param_guess

def completeOptimiseSim(xInit,p0,p1,path_to_savefile,cf = cost_func, iterations=6,shots=1000,a=0.02,c=0.35):
    '''
    Iteratively perform time evolution.
    At each timestep applies optimiseForCompleteRun to update the parameters to the next timestep
    Inputs:
        p0,p1: the parameters for the previous two time steps (used to do the fitting)
        cf: which cost function to use
            transfer_matrix_cost_func: transfer matrix simulation
            cost_func: noiseless circuit simulation 
        path_to_savefile: (str) the name of the file to save the data to
        iterations: (default=6) the number of iterations of the SPSA algorithm to run
        shots: (default = 1000) the number of shots with which to evaluate the cost function
        a,c: parameters of SPSA: used to adjust the search parameters (use a = 0.02, c = 0.35 for emulator runs)
    Outputs:
        new_param_guess: array of the updated set of parameters
    '''
    now = datetime.now()
    timestamp = now.strftime("%Y%m%dT%H%M%S")
    paramData = np.load('TMparams100000.npy')
    
    i=0
    t = [0,0.2,0.4]
    params = [xInit,p0,p1]
    set_params = p1
    
    for i in range(3,9):
        g = 0.2
        dt = 0.2
        timestep = i*dt
        t.append(timestep)
        filename = str(path_to_savefile)+'_step'+str(i)
        newParam = optimiseForCompleteRunSim(p0,p1,filename,cf, iterations=iterations,shots=shots,a=a,c=c)
        params.append(newParam)
        p0 = params[i-1]
        p1 = params[i]
    
    np.save(str(path_to_savefile)+'_parameter_results'+timestamp+'.npy',params)
    
    tm_losch = [overlap(xInit,p) for p in paramData]
    losch_values = [overlap(xInit,p) for p in params]
    
    np.save(str(path_to_savefile)+'_loschmidt_echo_results'+timestamp+'.npy',losch_values)
    
    plt.plot(ltimes,correct_ls, ls = ':', c='r', label = 'exact')
    plt.plot([0.2*i for i in range(len(tm_losch))],-np.log(tm_losch), ls = ':', c= 'orange', label = 'tm simulation')
    plt.plot([0.2*i for i in range(len(losch_values))],-np.log(losch_values), label = 'noiseless simulation', marker = 'x', ls = '--', c='b')

    plt.xlim(-0.1,2.1)
    plt.legend()
    plt.xlabel('time')
    plt.ylabel(r'-log$|\langle\psi(t)|\psi(0)\rangle|^2$')
    plt.minorticks_on()
    plt.grid(True, which='major')
    plt.grid(True, which='minor', ls = ':')
    plt.savefig(str(path_to_savefile)+'_loschmidtEcho_'+timestamp+'.png')
    plt.show()

    return params,losch_values,timestamp

def overlapCircuit(params1, params2, length, show=False):
    """
    Circuit for calculating state overlap of two states represented by \chi = 2 MPS
    Parameters
    ===========
        length : int
            Number of repeats of the transfer matrix
            psi : cirq.Gate
                representation of state 1, taking ansatz from ansatz.py
            phi: cirq.Gate
                representation of state 2, taking ansatz from ansatz.py
            show: bool
                optional flag to print circuit (recommended only for short lengths)
    """
    #psi = ToTimeLike(stateAnsatzXZ(params1))
    #phi = ToTimeLike(stateAnsatzXZ(params2))
    psi = timeLikeAnsatzXZ(params1)
    phi = timeLikeAnsatzXZ(params2)

    circuit = cirq.Circuit()
    qubits  = [cirq.GridQubit(i, 0) for i in range(4)]

    circuit.append(psi.on(qubits[0], qubits[1]))
    circuit.append(phi.on(qubits[3], qubits[2]))
    circuit.append(SWAP_measure(0).on(qubits[1], qubits[2]))
    for i in range(1, length+1):
        circuit.append(psi.on(qubits[0], qubits[1]))
        circuit.append(phi.on(qubits[3], qubits[2]))
        circuit.append(SWAP_measure(i).on(qubits[1], qubits[2]))
    circuit.append(cirq.ops.reset(qubits[0]))
    circuit.append(cirq.ops.reset(qubits[3]))

    if show is True:
        print(circuit.to_text_diagram(transpose=True))

    return circuit


def evolutionCircuit(params1, params2, g,t, length, show=False):
    """
    Circuit for calculating state overlap of two states represented by \chi = 2 MPS with a time evolution operator included
    Parameters
    ===========
        length : int
            Number of repeats of the transfer matrix
        psi : cirq.Gate
            representation of state 1, taking ansatz from ansatz.py
        phi: cirq.Gate
            representation of state 2, taking ansatz from ansatz.py
    W: cirq.Gate
    representation of the time evolution operator taken from hamiltonian.py
    show: bool
    optional flag to print circuit (recommended only for short lengths)
    """

    #psi = ToTimeLike(stateAnsatzXZ(params1))
    #phi = ToTimeLike(stateAnsatzXZ(params2))
    psi = timeLikeAnsatzXZ(params1)
    phi = timeLikeAnsatzXZ(params2)
    W = evolution_circuit_op(g,t)
    circuit = cirq.Circuit()
    qubits  = [cirq.GridQubit(i, 0) for i in range(6)]

    circuit.append(psi.on(qubits[0],qubits[1]))
    circuit.append(psi.on(qubits[5],qubits[4]))
    circuit.append(SWAP_measure(0).on(qubits[1], qubits[4]))
    for i in range(1,2*length,2):
        circuit.append(psi.on(qubits[0], qubits[1]))
        circuit.append(phi.on(qubits[5],qubits[4]))
        circuit.append(psi.on(qubits[0], qubits[2]))
        circuit.append(phi.on(qubits[5],qubits[4]))
        circuit.append(cirq.ion.two_qubit_matrix_to_ion_operations(qubits[1],qubits[2],W))
        circuit.append(SWAP_measure(i).on(qubits[1], qubits[4]))
        circuit.append(SWAP_measure(i+1).on(qubits[2], qubits[3]))
        circuit.append(cirq.ops.reset(qubits[1]))
        circuit.append(cirq.ops.reset(qubits[2]))
        circuit.append(cirq.ops.reset(qubits[3]))
        circuit.append(cirq.ops.reset(qubits[4]))

    circuit.append(cirq.ops.reset(qubits[0]))
    circuit.append(cirq.ops.reset(qubits[5]))

    if show is True:
        print(circuit.to_text_diagram(transpose=True))

    return circuit

def initialisationSchemes(xInit,p0,p1,path_to_savefile,steps=9,dt=0.2,iterations=6,shots=1000,a=0.05,c=0.5,guessType='linFit',showPlot=False):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%dT%H%M%S")
    paramData = np.load('TMparams100000.npy')
    
    
    if guessType == 'linFit':
        t=[0,0.2,0.4]+[0.4+i*dt for i in range(1,steps-1)]
        params=[xInit,p0,p1]

        for i in tqdm(range(2,steps)):
            g=0.2
            timestep = i*dt
            param_guess = linFit(params[i-1],params[i])
            opt = minimizeSPSA(cost_func,x0=param_guess, args = [params[i],'linFIT',timestamp,shots],niter=iterations, paired=False,a=a,c=c,restart_point=0)
            newParam = np.array(opt.x)
            params.append(newParam)
        
    
    if guessType == 'current':
        t = [0]
        params=[xInit]
        for i in tqdm(range(1,steps)):
            g=0.2
            timestep = i*dt
            t.append(timestep)
            param_guess = params[i-1]
            opt = minimizeSPSA(cost_func,x0=param_guess, args = [params[i-1],'current',timestamp,shots],niter=iterations, paired=False,a=a,c=c,restart_point=0)
            newParam = np.array(opt.x)
            params.append(newParam)
           
    if guessType == 'random':
        t = [0]
        params=[xInit]
        for i in tqdm(range(1,steps)):
            g=0.2
            timestep = i*dt
            t.append(timestep)
            param_guess = np.random.uniform(-np.pi,np.pi,(8))
            opt = minimizeSPSA(cost_func,x0=param_guess, args = [params[i-1],'random',timestamp,shots],niter=iterations, paired=False,a=a,c=c,restart_point=0)
            newParam = np.array(opt.x)
            params.append(newParam)
                

    np.save(str(path_to_savefile)+'_PARAMS_dt_'+str(dt)+'_its_'+str(iterations)+'_shots_'+str(shots)+'_a_'+str(a)+'_c_'+str(c)+'_'+timestamp+'.npy',params)
    #paramData = np.load('TMparams100000.npy')
    #tm_losch_plot = [-np.log(overlap(xInit,p)) for p in np.load('TMparams100000.npy')]
    losch_values = [-np.log(overlap(xInit,p)) for p in params]
    
    np.save(str(path_to_savefile)+'_LOSCH_dt_'+str(dt)+'_its_'+str(iterations)+'_shots_'+str(shots)+'_a_'+str(a)+'_c_'+str(c)+'_'+timestamp+'.npy',losch_values)
    
    print(timestamp)
    
    fig = plt.figure()
    plt.plot(ltimes,correct_ls,ls = ':',c='r',label='Analytically Exact')
    #plt.plot([0.2*i for i in range(len(tm_losch_plot))],tm_losch_plot,ls=':',c='orange',label = 'TM simulation')
    
    plt.plot(t,losch_values,'x',label = 'noiseless simulation')
    
    plt.xlim(-0.1,steps*dt+0.1)
    plt.legend()#bbox_to_anchor=(1.1, 1.05))
    plt.xlabel(r'$t$')
    plt.ylabel(r'-log$|\langle\psi(t)|\psi(0)\rangle|^2$')
    plt.minorticks_on()
    plt.grid(True, which='major')
    plt.grid(True, which='minor', ls = ':')
    #plt.savefig('dt_'+str(dt)+'higher_trotter//loschmidtEcho_'+timestamp+'.png')
    if showPlot == True:
        plt.show()
    if showPlot == False:
        plt.close(fig)

    #return t,params1,params2,losch_values1,losch_values2,timestamp
    return t,params,losch_values,timestamp