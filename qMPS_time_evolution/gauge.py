import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.linalg import expm
import cirq
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime
from tqdm import tqdm
from noisyopt import minimizeSPSA
from ncon import ncon

from classical import expectation, overlap, param_to_tensor, linFit, map_AB,right_fixed_point,left_fixed_point,unitary_to_RCF_tensor
from ansatz import stateAnsatzXZ
from simulation import simulate_noiseless, SWAP_measure, calc_and

def phase_calc(U):
    '''
    Construct overall phase of iMPS unitary U
    '''
    phase = np.conj(np.sqrt(U[0,0]*U[1,1]/np.abs(U[0,0]*U[1,1])))
    return phase

def calcG(mixedTM):
    '''
    Calculate the gauge matrix
    - this is the right environment of the mixed transfer matrix between the current and evolved iMPS
    '''
    Gs = right_fixed_point(mixedTM.reshape(4,4))[1].reshape(2,2)
    #u, s, v = np.linalg.svd(Gs)
    #Gs = ncon([u,v],([-1,1],[1,-2]))
    Gs = scipy.linalg.polar(Gs)[0]
    Gs =phase_calc(Gs)*Gs
    return Gs

def cost_function(params,gauged_tensor,original_params):
    '''
    Calculates overlap between iMPS tensor parametrised by params and the gauged iMPS tensor
    '''
    A = param_to_tensor(params)
    
    II = np.array([[1,0],[0,1]])
    p2 = np.einsum('inj, knj -> ik', gauged_tensor, A.conj())
    p3 = np.einsum('inj, knj -> ik', A, gauged_tensor.conj())
    p1 = np.einsum('ik,ik -> ',p2,p3)
    p4 = II@II.T.conj()
    c1 = p1-np.trace(p2@II)-np.trace(II@p3)+np.trace(p4)
    c2 = np.trace((p2-p4)@(p2-p4).T.conj())
    return np.real(c1)

def cost_func_0147(params,GA,original_params):
    '''
    Calculates overlap between iMPS tensor parametrised by params and the gauged iMPS tensor
    Only allows parameters 0,1,4 & 7 to vary
    '''
    params = np.insert(params,2,original_params[2])
    params = np.insert(params,3,original_params[3])
    params = np.insert(params,5,original_params[5])
    params = np.insert(params,6,original_params[6])
    A = param_to_tensor(params)
    
    II = np.array([[1,0],[0,1]])
    p2 = np.einsum('inj, knj -> ik', GA, A.conj())
    p3 = np.einsum('inj, knj -> ik', A, GA.conj())
    p1 = np.einsum('ik,ik -> ',p2,p3)
    p4 = II@II.T.conj()
    c1 = np.trace(p1-p2@II-II@p3+p4)
    c2 = np.trace((p2-p4)@(p2-p4).T.conj())
    return np.real(c2)

def optParameters(guessData,gauged_tensor,original_params,cost_func=cost_function):
    '''
    Find optimal parameters for an iMPS with highest fidelity with a gauged iMPS
    Inputs:
        guessData: (np.array) initial guess for the optimimal parameters
        gauged_tensor: (np.array) iMPS tensor with gauge G applied to it
        original_params: (np.array) the original parameters of the iMPS tensor (used only if some of the parameters are fixed)
        cost_function: the cost function to use to calculate the fidelity 
    Outputs:
        fidelity: the final fidelity between the optimised parameters and the gauged iMPS tensor
        params: the optimal params calculated by the optimiser
    '''
    res = minimize(cost_func,x0=guessData,args=(gauged_tensor,original_params),method='Nelder-Mead',tol = 1e-9,options={'maxiter':50000})
    params = res.x
    fidelity = res.fun
    print(res.success)
        
    return fidelity,params

def fidelity(params,original_params,gauged_tensor, changed_parameters=None):
    '''
    Calculate the fidelity between the original iMPS tensor and the gauged iMPS tensor
    Inputs:
        params:new parameters of the iMPS tensor calculated by the optimiser
        original_params: the parameters of the original iMPS tensor (used if keeping some parameters fixed in the optimisation)
        gauged_tensor: the gauged iMPS tensor
        changed_parameters: (np.array) numbers which parameters aren't kept fixed (default=none)
    '''
    fullParam = np.copy(original_params)
    if changed_parameters != None:
        for i in range(len(changed_parameters)):
            fullParam[changed_parameters[i]]=params[i]
    return right_fixed_point(map_AB(gauged_tensor,param_to_tensor(fullParam)))

