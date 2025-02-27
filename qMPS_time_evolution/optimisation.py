from qtuum.api_wrappers import QuantinuumAPI as QAPI
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime

from SPSA import minimizeSPSA
from classical import expectation, overlap, linFit
from Loschmidt import loschmidt_paper
from circuits import evolutionCircuit, doubledEvolutionCircuit, tripledEvolutionCircuit
from processing import evolSwapTestRatio

g0, g1 = 1.5, 0.2
max_time = 2
ltimes = np.linspace(0.0, max_time, 800)
correct_ls = [loschmidt_paper(t, g0, g1) for t in ltimes]

paramData = np.load('TMparams100000.npy')
x0 = paramData[0]
x1 = paramData[1]
x2 = paramData[2]


def evolutionSimulation(params, set_params, shots, machine, circuit_type, path_to_savefile, timestamp):
    '''
    Cost function for iMPS evolution, submits job to Quantinuum processor then evaluates fidelity density
    =============
    Inputs: 
        params (np.array): parameters calculated by optimiser for timestep t+dt
        set_params (np.array): parameters of current timestep t
        shots (int): number of shots to evalute the circuit with
        machine (str):
            for emulator: machine= 'H1-1E'
            for actual hardware: machine = 'H1-1' or 'H1-2'
        circuit_type (str):
            'single': one copy of the circuit per circuit
            'double': two copies of the circuit per circuit
            'triple': three copies of the circuit per circuit
        path_to_savefile (str): the path to and name of the file to save the data to incase of failure
        timestamp (str): timestamp to append to filename (calculated in the optimise function)
    =============
    Ouputs:     
        cost (float): the negative of the overlap between the evolved iMPS state and the new iMPS state
    '''
    qapi = QAPI(machine=machine)
    
    if circuit_type == 'single':
        n_len = shots*2
    elif circuit_type == 'double':
        n_len = shots*4
    elif circuit_type == 'triple':
        n_len = shots*6
    else:
        raise Exception("circuit_type must be one of: 'single', 'double', or 'triple'")
    
    job_id = qapi.submit_job(evolutionCircuit(set_params,params,circuit_type), 
                            shots=shots, 
                            machine=machine, # emulator = 'H1-1E'
                            name=str(circuit_type)+'EvolutionCircuitSPSA')
    n = 0
    i = 0
    
    # check something has stopped on the machine
    ## raises exception if result retrieval takes too long
    while n != n_len:
        results = qapi.retrieve_job(job_id)
        keys = results['results'].keys()
        n = sum([len(results['results'][key]) for key in keys])
        time.sleep(3)
        i += 1
        if i == 200:
            raise Exception('Results not fully populated after 10 minutes') 
        
    cost = evolSwapTestRatio(results['results'],5)      
    ### SAVE THE JOB ID TO FILE
    with open(str(path_to_savefile)+'_'+timestamp+'.json', 'r+') as file:
        opt_tracker = json.load(file)
        opt_tracker['job_id'].append(job_id)
        opt_tracker['measurement_data'].append(results['results'])
        opt_tracker['exp_cost'].append(cost)
        file.seek(0)
        json.dump(opt_tracker, file, indent=4)
    return -cost


def optimise(machine,circuit_type,p0,p1,path_to_savefile,timestep,iterations=6,shots=1000,a=0.02,c=0.35,show=True):
    '''
    Optimise just one time step 
    - Uses a linear extrapolation to make an initial guess of the next parameter
    - Then uses SPSA to further optimise the parameter to find the next step of the evolution
    =============
    Inputs:
        machine (str):
            for emulator: machine= 'H1-1E'
            for actual hardware: machine = 'H1-1' or 'H1-2'
        circuit_type (str):
            'single': one copy of the circuit per circuit
            'double': two copies of the circuit per circuit
            'triple': three copies of the circuit per circuit
        p0,p1 (np.array): the parameters for the previous two time steps (used to do the fitting)
        path_to_savefile (str): the name of the file to save the data to incase of failure
        timestep (str): which timestep is being evolved from
        iterations (int): the number of iterations of the SPSA algorithm to run (default=6)
        shots (int): the number of shots with which to evaluate the cost function (default = 1000)
        a,c (float): parameters of SPSA: used to adjust the search parameters (default a = 0.02, c = 0.35) 
        show (bool): Set True to see the Loschmidt echo plotted for the current and updated timestep (default=True)
    =============
    Outputs:
        new_param_guess (np.array): array of the updated set of parameters
    Also (if show=True) shows a plot of the Loschmidt echo with this time evolution step plotted on it for comparison. Serves as a check the evolution is progresssing correctly
    '''
    qapi = QAPI(machine=machine) # emulator = 'H1-1E',  machine = 'H1-1' or 'H1-2'?
    now = datetime.now()
    timestamp= now.strftime("%Y%m%dT%H%M%S")
    print('Associated timestamp: ' + timestamp)
    
    def callback_fn(xk):
        cost = expectation(set_params,xk,g,dt)
        loschmidt = overlap(xInit,xk)
        params = list(xk)        
        with open(str(path_to_savefile)+'_'+timestamp+'.json', 'r+') as file:
            opt_tracker = json.load(file)
            opt_tracker['tm_cost'].append(cost)
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
    opt = minimizeSPSA(evolutionSimulation,x0 = param_guess, args = [set_params, shots, machine, circuit_type, path_to_savefile, timestamp],  niter = iterations, paired=False,a=a, c=c, callback=callback_fn, restart_point=0)
    
    
    new_param_guess = np.array(opt.x)
    if show == True:
        tm_losch = [overlap(xInit,params) for params in paramData]
        plt.plot(ltimes,correct_ls, ls = ':', label = 'exact')
        plt.plot([0.2*i for i in range(len(tm_losch))],-np.log(tm_losch), ls = ':', label = 'tm simulation')
        plt.plot([timestep, timestep+0.2],-np.log([overlap(xInit,p1),overlap(xInit,new_param_guess)]), label = 'emulator simulation', marker = 'x')
        plt.xlim(-0.1,2.1)
        plt.legend()
        plt.xlabel('time')
        plt.ylabel(r'-log$|\langle\psi(t)|\psi(0)\rangle|^2$')
        plt.show()
    return new_param_guess

def restartFromFail(machine,circuit_type,p1,timestamp,path_to_savefile,timestep,iterations=6,shots=1000,a=0.02,c=0.35,show=True):
    '''
    Finds the failure point in a timestep evolution and finishes the SPSA run
    =============
    Inputs:
        machine (str):
            for emulator: machine= 'H1-1E'
            for actual hardware: machine = 'H1-1' or 'H1-2'
        circuit_type (str):
            'single': one copy of the circuit per circuit
            'double': two copies of the circuit per circuit
            'triple': three copies of the circuit per circuit
        p1 (np.array): the parameters for the current time step 
        path_to_savefile (str): the name of the file to save the data to incase of failure
        timestep (str): which timestep is being evolved from
        iterations (int): the number of iterations of the SPSA algorithm to run (default=6)
        shots (int): the number of shots with which to evaluate the cost function (default = 1000)
        a,c (float): parameters of SPSA: used to adjust the search parameters (default a = 0.02, c = 0.35) 
        show (bool): Set True to see the Loschmidt echo plotted for the current and updated timestep (default=True)
    =============
    Outputs:
        new_param_guess (np.array): array of the updated set of parameters
    Also (if show=True) shows a plot of the Loschmidt echo with this time evolution step plotted on it for comparison. Serves as a check the evolution is progresssing correctly
    '''
    xInit = x0

    with open(str(path_to_savefile)+'_'+timestamp+'.json', 'r') as file:
        tracker = json.load(file)
    start_point = len(tracker['params'])
    
    qapi = QAPI(machine=machine)
    
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
        
         
    g = 0.2
    dt = 0.2
    set_params = p1
    param_guess = tracker['params'][start_point-1]
    opt = minimizeSPSA(evolutionSimulation,x0 = param_guess, args = [set_params, shots, machine, circuit_type, path_to_savefile, timestamp],  niter = iterations, paired=False,a=a, c=c, callback=callback_fn, restart_point=start_point)
 
    new_param_guess = np.array(opt.x)

    if show == True:
        tm_losch = [overlap(xInit,params) for params in paramData]
        plt.plot(ltimes,correct_ls, ls = ':', label = 'exact')
        plt.plot([0.2*i for i in range(len(tm_losch))],-np.log(tm_losch), ls = ':', label = 'tm simulation')
        plt.plot([timestep, timestep+0.2],-np.log([overlap(xInit,p1),overlap(xInit,new_param_guess)]), label = 'emulator simulation', marker = 'x')
        plt.xlim(-0.1,2.1)
        plt.legend()
        plt.xlabel('time')
        plt.ylabel(r'-log$|\langle\psi(t)|\psi(0)\rangle|^2$')
        plt.show()
    return new_param_guess


def completeOptimise(machine,circuit_type,xInit,p0,p1,path_to_savefile,iterations=6,shots=1000,a=0.02,c=0.35,show=True):
    '''
    Iteratively perform time evolution.
    At each timestep applies optimise to update the parameters to the next timestep
    Saves the parameters, loschmidt echo values and plot of the loschmidt echo to file
    =============
    Inputs:
        machine (str):
            for emulator: machine= 'H1-1E'
            for actual hardware: machine = 'H1-1' or 'H1-2'
        circuit_type (str):
            'single': one copy of the circuit per circuit
            'double': two copies of the circuit per circuit
            'triple': three copies of the circuit per circuit
        xInit (np.array): the parameters at time t=0 for calculating the Loschmidt echo
        p0,p1 (np.array): the parameters for the two time steps prior to the first one being evaluated (used to do the fitting)
        path_to_savefile (str): the name of the file to save the data to incase of failure
        iterations (int): the number of iterations of the SPSA algorithm to run (default=6)
        shots (int): the number of shots with which to evaluate the cost function (default = 1000)
        a,c (float): parameters of SPSA: used to adjust the search parameters (default a = 0.02, c = 0.35) 
        show (bool): Set True to display the Loschmidt echo (default=True)
    =============
    Outputs:
        params (np.array): array containing the parameters at each timestep
        losch_values (np.array): array containing the loschmidt echo value at each timestep
    Also (if show=True) shows a plot of the full Loschmidt echo 
    '''
    
    now = datetime.now()
    timestamp= now.strftime("%Y%m%dT%H%M%S")
    paramData = np.load('TMparams100000.npy')
    #fullTMdata = np.load('TMparams.npy')
    i=0
    t = [0,0.2,0.4]
    params = [xInit,p0,p1]
    set_params = p1
    
    for i in range(3,9):
        g = 0.2
        dt = 0.2
        timestep = i*dt
        t.append(timestep)
        filename = path_to_savefile+'step'+str(i)
        newParam = optimise(machine,circuit_type,p0,p1,filename,timestep,iterations,shots,a,c, show=False)

        params.append(newParam)
        p0 = params[i-1]
        p1 = params[i]
    
    np.save(str(path_to_savefile)+'_parameter_results'+timestamp+'.npy',params)
    
    tm_losch = [overlap(xInit,p) for p in paramData]
    losch_values = [-np.log(overlap(xInit,p)) for p in params]
    
    np.save(str(path_to_savefile)+'_loschmidt_echo_results'+timestamp+'.npy',losch_values)
    
    
    plt.plot(ltimes,correct_ls, ls = ':', c='r', label = 'exact')
    plt.plot([0.01*i for i in range(len(tm_losch))],-np.log(tm_losch), ls = ':', c= 'orange', label = 'tm simulation')
    plt.plot([0.2*i for i in range(len(losch_values))],losch_values, label = 'emulator simulation', marker = 'x', ls = '--', c='b')

    plt.xlim(-0.1,2.1)
    plt.legend()
    plt.xlabel('time')
    plt.ylabel(r'-log$|\langle\psi(t)|\psi(0)\rangle|^2$')
    plt.savefig(str(path_to_savefile)+'_loschmidtEcho_'+timestamp+'.png')
    if show == True:
        plt.show()

    return params,losch_values