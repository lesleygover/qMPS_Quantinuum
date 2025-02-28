import matplotlib.pyplot as plt
import numpy as np
import cirq
import json
from datetime import datetime
from tqdm import tqdm

from SPSA import minimizeSPSA
from classical import expectation, overlap, linFit, linExtrap
from Loschmidt import loschmidt_paper
from ansatz import stateAnsatzXZ, timeLikeAnsatzXZ
from Hamiltonian import evolution_circuit_op,Wgate0202

g0, g1 = 1.5, 0.2
max_time = 2
ltimes = np.linspace(0.0, max_time, 800)
correct_ls = [loschmidt_paper(t, g0, g1) for t in ltimes]

paramData = np.load('TMparams100000.npy')
x0 = paramData[0]
x1 = paramData[1]
x2 = paramData[2]

def simulate_noiseless(circuit, shots):
    """
    Simulate a circuit without noise
    =============
    Inputs:
        circuit (cirq.Circuit): cirq circuit to simulate
        shots (int): number of shots to run of the circuit
    =============
    Outputs:
        result (cirq.study.result.ResultDict): bitstrings of measurements from the simulation
    """
    sim = cirq.Simulator()
    results = sim.run(circuit, repetitions=shots)
    return results

class SWAP_measure(cirq.Gate):
	""" 
    Gate encompassing the SWAP test and measurement step for circuits in time 
    =============
    Inputs:
        index (int or str): labels the measurement pair resulting from the swap test 
    """
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
    =============
	Inputs:
        params1,params2 (np.array): array of parameters to parameterise the two sets of iMPS unitaries
		length (int): number of repeats of the transfer matrix
    =============
	Outputs:    
        circuit (cirq.Circuit): circuit for evaluting the evolution cost function
    '''
    psi = stateAnsatzXZ(params1)
    phi = stateAnsatzXZ(params2)
    W = Wgate0202()
    
    circuit = cirq.Circuit()
    qubits = [cirq.GridQubit(i,0) for i in range(4+length*4)]
    noOfQubits = len(qubits)
    
    circuit.append(psi.on(qubits[0],qubits[1]))
    circuit.append(psi.on(qubits[noOfQubits-1],qubits[noOfQubits-2]))
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
	Calculate the bitwise AND of the measurement bitstrings for results from cirq noiseless simulation
	=============
    Inputs:
        results (cirq.study.result.ResultDict): result data from simulation
    =============
	Outputs:
        bitstrings (np.array): bitstrings of the bitwise ANDs
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
    =============
	Inputs:
        bitstrings (np.array): bitstrings of bitwiseANDs (output of calc_and)
		length (int): how many pairs of measurements there are per circuit
    =============
	Outputs:
        overlap (float): returns the fidelity density calculated using the swap test
    """
	shots = np.size(bitstrings,0)
	parity = 0
	for shot in range(shots):
		if (sum(bitstrings[shot, :length]) % 2 == 0):
			parity += 1
	prob = parity/shots
	overlap = np.sqrt((np.abs(2*prob - 1)))
	return overlap

def swapTestRatio(bitstrings, length):
	"""
    Calculates ratio of the swap tests on the circuit results for length and length-1
    =============
	Inputs:
        bitstrings (np.array): bitstrings of bitwiseANDs (output of calc_and)
		length (int): how many pairs of measurements there are per circuit
    =============
	Outputs:
        overlap (float): returns the fidelity density calculated using the swap test
    """
	shots = np.size(bitstrings,0)
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
    =============
    Inputs:
        bitstrings (np.array): bitstrings of bitwiseANDs (output of calc_and)
        length (int): how many pairs of measurements there are per circuit
    =============
    Outputs:
        overlap (float): returns the fidelity density calculated using the swap test
    """
    shots = np.size(bitstrings,0)
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


def noiseless_cost_func(params,set_params,path_to_savefile,timestamp,shots):
    '''
    Cost function for iMPS evolution with noiseless circuit simulation through cirq
    =============
    Inputs:
        params (np.array): parameters calculated by optimiser for timestep t+dt
        set_params (np.array): parameters of current timestep t
        path_to_savefile (str): the name of the file to save the data to incase of failure
        timestamp (str): timestamp to append to filename (calculated in the optimise function)
        shots (int): number of shots to evalute the circuit with
    =============
    Ouputs:     
        cost (float): the negative of the overlap between the evolved iMPS state and the new iMPS state
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
    Cost function for iMPS evolution with exact in ansatz transfer matrix simulation
    =============
    Inputs:
        params (np.array): parameters calculated by optimiser for timestep t+dt
        set_params (np.array): parameters of current timestep t
        path_to_savefile (str): the name of the file to save the data to incase of failure
        timestamp (str): timestamp to append to filename (calculated in the optimise function)
        shots: just included so matches format of ciruit simulation(set=0 but doesn't matter)
    =============
    Ouputs:     
        cost (float): the negative of the overlap between the evolved iMPS state and the new iMPS state
    '''
    g = 0.2
    t = 0.2
    cost = expectation(set_params,params,g,t)
    with open(str(path_to_savefile)+'_transfer_matrix_'+timestamp+'.json', 'r+') as file:
        opt_tracker = json.load(file)
        opt_tracker['cost'].append(cost)
        file.seek(0)
        json.dump(opt_tracker, file, indent=4)
    return -cost

def optimiseSim(p0,p1,path_to_savefile,cf=noiseless_cost_func,iterations=6,shots=1000,a=0.02,c=0.35):
    '''
    Optimise just one time step 
    Uses either the linear fit to make an initial guess of the next parameter
    Then uses SPSA to further optimise the parameter to find the next step of the evolution
    =============
    Inputs:
        p0,p1 (np.array): the parameters for the previous two time steps (used to do the fitting)
        cf (func): which cost function to use
            transfer_matrix_cost_func: transfer matrix simulation
            cost_func: noiseless circuit simulation 
        path_to_savefile (str): the name of the file to save the data to incase of failure
        iterations (int): (default=6) the number of iterations of the SPSA algorithm to run
        shots (int): (default = 1000) the number of shots with which to evaluate the cost function
        a,c (float): parameters of SPSA: used to adjust the search parameters (use a = 0.02, c = 0.35 for emulator runs)
    =============
    Outputs:
        new_param_guess (np.array): array of the updated set of parameters
    '''
    
    now = datetime.now()
    timestamp = now.strftime("%Y%m%dT%H%M%S")

    def callback_fn(xk):
        cost = expectation(set_params,xk,g,dt)
        loschmidt = -np.log(overlap(xInit,xk))
        params = list(xk)       
        with open(str(path_to_savefile)+'_'+timestamp+'.json', 'r+') as file:
            opt_tracker = json.load(file)
            opt_tracker['final_cost'].append(cost)
            opt_tracker['losch_overlap'].append(loschmidt)
            opt_tracker['params'].append(params)
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

    # SPSA optmisation
    opt = minimizeSPSA(cf,x0 = param_guess, args = [set_params, path_to_savefile, timestamp, shots],  niter = iterations, paired=False,a=a, c=c, callback=callback_fn, restart_point=0)
    
    
    new_param_guess = np.array(opt.x)


    return new_param_guess

def completeOptimiseSim(xInit,p0,p1,path_to_savefile,cf = noiseless_cost_func, iterations=6,shots=1000,a=0.02,c=0.35):
    '''
    Iteratively perform time evolution.
    At each timestep applies optimiseForCompleteRun to update the parameters to the next timestep
    Saves the parameters, loschmidt echo values and plot to 
        - <path_to_savefile>_parameter_results_<timestamp>.npy
        - <path_to_savefile>_loschmidt_echo_results_<timestamp>.npy
        - <path_to_savefile>'_loschmidtEcho_<timestamp>.svg
    respectively
    =============
    Inputs:
        xInit (np.array): the parameters at time t=0 for calculating the Loschmidt echo
        p0,p1 (np.array): the parameters for the start point timestep and timestep before (used to do the fitting)
        cf (func): which cost function to use
            transfer_matrix_cost_func: transfer matrix simulation
            cost_func: noiseless circuit simulation 
        path_to_savefile (str): the name of the file to save the data to
        iterations (int): (default=6) the number of iterations of the SPSA algorithm to run
        shots (int): (default = 1000) the number of shots with which to evaluate the cost function
        a,c (float): parameters of SPSA: used to adjust the search parameters (use a = 0.02, c = 0.35 for emulator runs)
    =============
    Outputs:
        params (np.array): array containing the parameters at each timestep
        losch_values (np.array): array containing the loschmidt echo value at each timestep
        timestamp (str): timestamp for the evolution run (used in filenames saved to)
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
        newParam = optimiseSim(p0,p1,filename,cf, iterations=iterations,shots=shots,a=a,c=c)
        params.append(newParam)
        p0 = params[i-1]
        p1 = params[i]
    
    np.save(str(path_to_savefile)+'_parameter_results_'+timestamp+'.npy',params)
    
    tm_losch = [overlap(xInit,p) for p in paramData]
    losch_values = [overlap(xInit,p) for p in params]
    
    np.save(str(path_to_savefile)+'_loschmidt_echo_results_'+timestamp+'.npy',losch_values)
    
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
    plt.savefig(str(path_to_savefile)+'_loschmidtEcho_'+timestamp+'.svg')
    plt.show()

    return params,losch_values,timestamp

def overlapCircuit(params1, params2, length, show=False):
    """
    Circuit for calculating state overlap of two states represented by \chi = 2 MPS
    ===========
    Inputs:
        length  (int): Number of repeats of the transfer matrix
        params1,params2 (np.array): array of parameters to parameterise the two sets of iMPS unitaries
        show (bool): optional flag to print circuit (recommended only for short lengths)
    ===========
    Outputs: 
        circuit (cirq.Circuit): the overlap circuit
    """
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
    ===========
    Inputs:
        length  (int): Number of repeats of the transfer matrix
        params1,params2 (np.array): array of parameters to parameterise the two sets of iMPS unitaries
        g (float): coupling strength of the TFIM Hamiltonian
        t (float): timestep size for the evolution operator
        show (bool): optional flag to print circuit (recommended only for short lengths)
    ===========
    Outputs: 
        circuit (cirq.Circuit): the overlap circuit
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

class W2Gate0205(cirq.Gate):
    """
    2nd order time evolution operator for TFIM with g=0.2, dt = 0.5
    broken down into CNOT, rx,ry,rz
    """
    def __init__(self):
        super(W2Gate0205,self)
        
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
        return ['W', 'W']
    
def secondOrderCircuit(params1,params2):
    '''
    Timelike evolution circuit with 2nd order Trotterisation of the evolution operator 
    Evolution operator used TFIM Hamiltonian with g=0.2,dt=0.5
    =============
	Inputs:
        params1,params2 (np.array): array of parameters to parameterise the two sets of iMPS unitaries
    =============
	Outputs:    
        circuit (cirq.Circuit): circuit for evaluting the evolution cost function
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
    circuit.append(W2Gate0205().on(qubits[1],qubits[3]))
    circuit.append(phi.on(qubits[6],qubits[4]))
    circuit.append(phi.on(qubits[4],qubits[7]))
    circuit.append(W2Gate0205().on(qubits[6],qubits[4]))
    circuit.append(cirq.CNOT(qubits[3],qubits[4]))
    circuit.append(cirq.H(qubits[3]))
    circuit.append(cirq.measure(qubits[3], key='psi'+str(1)))
    circuit.append(cirq.measure(qubits[4], key='phi'+str(1)))
    circuit.append(cirq.reset(qubits[3]))
    circuit.append(cirq.reset(qubits[4]))
    
    circuit.append(psi.on(qubits[0],qubits[2]))
    circuit.append(psi.on(qubits[2],qubits[3]))
    circuit.append(W2Gate0205().on(qubits[0],qubits[2]))
    circuit.append(W2Gate0205().on(qubits[1],qubits[2]))
    circuit.append(phi.on(qubits[7],qubits[5]))
    circuit.append(phi.on(qubits[5],qubits[4]))
    circuit.append(W2Gate0205().on(qubits[7],qubits[5]))
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
    circuit.append(W2Gate0205().on(qubits[3],qubits[1]))
    circuit.append(W2Gate0205().on(qubits[0],qubits[1]))
    circuit.append(phi.on(qubits[4],qubits[6]))
    circuit.append(phi.on(qubits[6],qubits[5]))
    circuit.append(W2Gate0205().on(qubits[4],qubits[6]))
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

    return circuit

def secondOrderCircuitFast(params1,params2):
    '''
    Timelike evolution circuit with 2nd order Trotterisation of the evolution operator 
    Evolution operator used TFIM Hamiltonian with g=0.2,dt=0.5
    Is without mid circuit measurement and resets
    - instead just keeps the qubit and uses a new one so can measure all at the end
    - only good for in simulation
    =============
	Inputs:
        params1,params2 (np.array): array of parameters to parameterise the two sets of iMPS unitaries
    =============
	Outputs:    
        circuit (cirq.Circuit): circuit for evaluting the evolution cost function
    '''
    psi = stateAnsatzXZ(params1)
    phi = stateAnsatzXZ(params2)
    
    circuit = cirq.Circuit()
    qubits = [cirq.GridQubit(i,0) for i in range(16)]
    noOfQubits= len(qubits)
    
    circuit.append(psi.on(qubits[0],qubits[1]))
    circuit.append(psi.on(qubits[1],qubits[2]))
    circuit.append(psi.on(qubits[2],qubits[3]))
    
    circuit.append(W2Gate0205().on(qubits[1],qubits[2]))
    
    circuit.append(psi.on(qubits[3],qubits[4]))
    circuit.append(psi.on(qubits[4],qubits[5]))
    
    circuit.append(W2Gate0205().on(qubits[3],qubits[4]))
    circuit.append(W2Gate0205().on(qubits[1],qubits[4]))
    
    circuit.append(psi.on(qubits[5],qubits[6]))
    circuit.append(psi.on(qubits[6],qubits[7]))
    
    circuit.append(W2Gate0205().on(qubits[5],qubits[6]))
    circuit.append(W2Gate0205().on(qubits[3],qubits[6]))
    
    circuit.append(psi.on(qubits[15],qubits[14]))
    circuit.append(phi.on(qubits[14],qubits[13]))
    circuit.append(phi.on(qubits[13],qubits[12]))
    
    circuit.append(W2Gate0205().on(qubits[14],qubits[13]))
    
    circuit.append(phi.on(qubits[12],qubits[11]))
    circuit.append(phi.on(qubits[11],qubits[10]))
    
    circuit.append(W2Gate0205().on(qubits[12],qubits[11]))
    
    circuit.append(phi.on(qubits[10],qubits[9]))
    circuit.append(phi.on(qubits[9],qubits[8]))
    
    circuit.append(W2Gate0205().on(qubits[10],qubits[9]))
    
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

    return circuit

def cost_func_2ndOrder(params,set_params,shots,speed='fast'):
    '''
    Cost function for iMPS evolution with noiseless circuit simulation through cirq
    Uses a second order Trotterisation of the time evolution operator
    =============
    Inputs:
        params (np.array): parameters calculated by optimiser for timestep t+dt
        set_params (np.array): parameters of current timestep 
        shots (int): number of shots to evalute the circuit with
        speed (str): determines whether to do a faster simulation circuit or not (default='fast')
            'fast': uses the version of the circuit without the qubit resetting
            'slow': uses the version of the circuit with the qubit resetting (as done on device)
    =============
    Ouputs:     
        cost (float): the negative of the overlap between the evolved iMPS state and the new iMPS state
    '''
    if speed == 'fast':
        cost = swapTest(calc_and(simulate_noiseless(secondOrderCircuitFast(set_params,params),shots)),7)
    else: 
        cost = swapTest(calc_and(simulate_noiseless(secondOrderCircuit(set_params,params),shots)),7)
    return -cost

def fullRun2ndOrder(xInit,p0,p1,path_to_savefile,steps = 9,dt=0.5,iterations=6,shots=1000,a=0.025,c=0.5,showPlot=False):
    '''
    Iteratively perform time evolution of iMPS state with a second order Trotterisation of the time evolution operator
    Saves the parameters, loschmidt echo values and plot to 
        - <path_to_savefile>_PARAMS_order_2_dt_<dt>_its_<iterations>_shots_<shots>_a_<a>_c_<c>_<timestamp>.npy
        - <path_to_savefile>_LOSCH_order_2_dt_<dt>_its_<iterations>_shots_<shots>_a_<a>_c_<c>_<timestamp>.npy
        - <path_to_savefile>_LoschmidtEcho_order_2_dt_<dt>_its_<iterations>_shots_<shots>_a_<a>_c_<c>_<timestamp>.svg
    respectively
    =============
    Inputs:
        XInit (np.array): the parameters at time t=0 for calculating the Loschmidt echo
        p0,p1 (np.array): the parameters for the start point timestep and timestep before (used to do the fitting)
        path_to_savefile (str): the name of the file to save the data to
        steps (int): how many time evolution updates to do (default = 9)
        dt (float): timestep being used in the evolution (default = 0.5)
        iterations (int): the number of iterations of the SPSA algorithm to run (default=6)
        shots (int): the number of shots with which to evaluate the cost function (default = 1000) 
        a,c (float): parameters of SPSA: used to adjust the search parameters (use a = 0.02, c = 0.35 for emulator runs)
        showPlot (bool): flag to choose whether to display the final plot or not (default = False)
    =============
    Outputs:
        t (np.array): array of the times each parameter set corresponds to
        params (np.array): array containing the parameters at each timestep
        losch_values (np.array): array containing the loschmidt echo value at each timestep
        timestamp (str): timestamp for the evolution run (used in filenames saved to)
    '''
    now = datetime.now()
    timestamp = now.strftime("%Y%m%dT%H%M%S")
    paramData = np.load('TMparams100000.npy')
    
    t=[0,0.2,0.4]+[0.4+i*dt for i in range(1,steps-1)]
    params=[xInit,p0,p1]
    
    for i in tqdm(range(2,steps)):
        
        param_guess = linExtrap(t[i-1],t[i],params[i-1],params[i],t[i+1])
        opt = minimizeSPSA(cost_func_2ndOrder,x0=param_guess, args = [params[i],dt,shots,2],niter=iterations, paired=False,a=a,c=c,restart_point=0)
        newParam = np.array(opt.x)
        params.append(newParam)
       
    np.save(str(path_to_savefile)+'_PARAMS_order_2_dt_'+str(dt)+'_its_'+str(iterations)+'_shots_'+str(shots)+'_a_'+str(a)+'_c_'+str(c)+'_'+timestamp+'.npy',params)
    
    tm_losch = [-np.log(overlap(xInit,p)) for p in paramData]
    losch_values = [-np.log(overlap(xInit,p)) for p in params]

    np.save(str(path_to_savefile)+'_LOSCH_order_2_dt_'+str(dt)+'_its_'+str(iterations)+'_shots_'+str(shots)+'_a_'+str(a)+'_c_'+str(c)+'_'+timestamp+'.npy',losch_values)
    
    print(timestamp)
    fig = plt.figure()
    plt.plot(ltimes,correct_ls,ls = ':',c='r',label='exact')
    plt.plot([0.2*i for i in range(len(tm_losch))],tm_losch,ls=':',c='orange',label = 'TM simulation')
    plt.plot(t,losch_values,'x',label = '2nd order noiseless simulation')
    
    plt.xlim(-0.1,steps*dt+0.1)
    plt.legend()#bbox_to_anchor=(1.1, 1.05))
    plt.xlabel('time')
    plt.ylabel(r'-log$|\langle\psi(t)|\psi(0)\rangle|^2$')
    plt.minorticks_on()
    plt.grid(True, which='major')
    plt.grid(True, which='minor', ls = ':')
    plt.savefig(str(path_to_savefile)+'_LoschmidtEcho_order_2_dt_'+str(dt)+'_its_'+str(iterations)+'_shots_'+str(shots)+'_a_'+str(a)+'_c_'+str(c)+'_'+timestamp+'.svg')
    if showPlot == True:
        plt.show()
    if showPlot == False:
        plt.close(fig)

    return t,params,losch_values,timestamp