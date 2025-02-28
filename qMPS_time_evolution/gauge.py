import numpy as np
import scipy
from scipy.optimize import minimize

from classical import param_to_tensor, map_AB,right_fixed_point

def phase_calc(U):
    '''
    Construct overall phase of gauge unitary
    ========
    Inputs:
        U (np.ndarray): 2x2 matrix of the gauge unitary
    ========
    Outputs:
        phase (np.complex128): the phase of the unitary
    '''
    phase = np.conj(np.sqrt(U[0,0]*U[1,1]/np.abs(U[0,0]*U[1,1])))
    return phase

def calcG(mixedTM):
    '''
    Calculate the gauge matrix
    - this is the right environment of the mixed transfer matrix between the current and evolved iMPS
    ========
    Inputs:
        mixedTM (np.ndarray): mixed transfer matrix (matrix of shape (2,2,2,2) or (4,4))
    ========
    Outputs:
        G (np.ndarray): 2x2 matrix of the gauge unitary
    '''
    G = right_fixed_point(mixedTM.reshape(4,4))[1].reshape(2,2)
    #u, s, v = np.linalg.svd(Gs)
    #G = ncon([u,v],([-1,1],[1,-2]))
    G = scipy.linalg.polar(G)[0]
    G =phase_calc(G)*G
    return G

def cost_function(params,gauged_tensor,original_params):
    '''
    Calculates trace distance between the mixed transfer matrix between the gauged iMPS and the optimised iMPS and the identity
    ========
    Inputs:
        params (np.array): parameters of the iMPS tensor from optimiser
        gauged_tensor (np.ndarray): original iMPS tensor with the gauge unitary applied
        original_params (np.array): the original parameters of the iMPS tensor (ignore in this cost_function, there to match formatting)
    ========
    Outputs:
        cost (float): the overlap between the two iMPS tensors
    '''    
    A = param_to_tensor(params)
    II = np.array([[1,0],[0,1]])
    p2 = np.einsum('inj, knj -> ik', gauged_tensor, A.conj())
    p3 = np.einsum('inj, knj -> ik', A, gauged_tensor.conj())
    p1 = np.einsum('ik,ik -> ',p2,p3)
    p4 = II@II.T.conj()
    c1 = p1-np.trace(p2@II)-np.trace(II@p3)+np.trace(p4)
    c2 = np.trace((p2-p4)@(p2-p4).T.conj())
    cost = np.real(c1)
    return cost

def cost_func_0147(params,GA,original_params):
    '''
    Calculates trace distance between the mixed transfer matrix between the gauged iMPS and the optimised iMPS and the identity
    Only allows parameters 0,1,4 & 7 to vary
    ========
    Inputs:
        params (np.array): parameters of the iMPS tensor from optimiser
        gauged_tensor (np.ndarray): original iMPS tensor with the gauge unitary applied
        original_params (np.array): the original parameters of the iMPS tensor (to add back into optimised parameters to form a full parametrisation)
    ========
    Outputs:
        cost (float): the overlap between the two iMPS tensors
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
    cost = np.real(c2)
    return cost

def optParameters(guessData,gauged_tensor,original_params,cost_func=cost_function):
    '''
    Find optimal parameters for an iMPS with highest fidelity with a gauged iMPS
    ========
    Inputs:
        guessData (np.array): initial guess for the optimimal parameters
        gauged_tensor (np.array): iMPS tensor with gauge G applied to it
        original_params (np.array): the original parameters of the iMPS tensor (used only if some of the parameters are fixed)
        cost_function (func): the cost function to use to calculate the fidelity 
    ========
    Outputs:
        fidelity (float): the final fidelity between the optimised parameters and the gauged iMPS tensor
        params (np.array): the optimal params calculated by the optimiser
    '''
    res = minimize(cost_func,x0=guessData,args=(gauged_tensor,original_params),method='Nelder-Mead',tol = 1e-9,options={'maxiter':50000})
    params = np.array(res.x)
    fidelity = res.fun
    print(res.success)
        
    return fidelity,params

def fidelity(params,original_params,gauged_tensor, changed_parameters=None):
    '''
    Calculate the fidelity between the iMPS tensor and the gauged iMPS tensor
    ========
    Inputs:
        params (np.array):new parameters of the iMPS tensor calculated by the optimiser
        original_params (np.array): the parameters of the original iMPS tensor (used if keeping some parameters fixed in the optimisation)
        gauged_tensor (np.ndarray): the gauged iMPS tensor
        changed_parameters (np.array): numbers which parameters aren't kept fixed (default=none)
    Outputs:
        fidelity (float): the fidelity between the iMPS tensor and the gauged iMPS tensor
    '''
    fullParam = np.copy(original_params)
    if changed_parameters != None:
        for i in range(len(changed_parameters)):
            fullParam[changed_parameters[i]]=params[i]
    fidelity = right_fixed_point(map_AB(gauged_tensor,param_to_tensor(fullParam)))
    return fidelity

