import cirq
import numpy as np
import ncon
from scipy.linalg import eig
from ansatz import stateAnsatzXZ
from Hamiltonian import evolution_op



def unitary_to_RCF_tensor(U):
    """
    Convert a unitary U to a right canonical form tensor A such that
    |0>     i
    |     |
    |     |     |
    ---U---     | direction of unitary
    |     |     |
    |     |     v
    j     k
    A.shape = (i, j, k)
    i == A == k
         |
         j
    Parameters
    ============
    Inputs:
        U (np.ndarray): A 4x4 unitary representing a 2 qubit operation.
    ============
    Outputs:
        A (np.ndarray): A bond dim. 2 MPS tensor representation of the unitary U with shape (2, 2, 2)
    ============
    Raises
        ValueError: If U is not a 4x4 unitary matrix
    """
    if U.shape != (4, 4):
        raise ValueError(f"Expected 4x4 unitary matrix, got shape {U.shape}")
    
    n = 2
    zero = np.array([1., 0.])
    
    Ucontr = [-2, -3, -1, 1]
    A = ncon.ncon([U.reshape(*2 * n * [2]), zero], [Ucontr, [1,]])
    return A.reshape(2, 2, 2)


def unitary_to_LCF_tensor(U):
    """
    Convert a unitary U to a left canonical form tensor A such that:
    |0>    k
    |     |
    |     |     |
    ---U---     | direction of unitary
    |     |     |
    |     |     v
    i     j
    A.shape = (i, j, k)
    i == A == k
         |
         j
    ===========
    Inputs 
    U (np.ndarray): A unitary matrix representing a multi-qubit operation.
    ===========
    Outputs 
        A (np.ndarray): A MPS tensor representation of the unitary U
    ============
    Raises
        ValueError: If U is not a square matrix with dimensions that are powers of 2
    """
    if U.shape[0] != U.shape[1] or not (U.shape[0] & (U.shape[0] - 1) == 0):
        raise ValueError(f"Expected square unitary with dimensions as power of 2, got shape {U.shape}")
    
    n = int(np.log2(U.shape[0]))
    zero = np.array([1.0, 0.0])

    Ucontr = list(range(-1, -n-1, -1)) + [1] + list(range(-n-1, -2*n, -1))
    A = ncon.ncon([U.reshape(*2 * n * [2]), zero], [Ucontr, [1,]])
    return A.reshape(2**(n-1), 2, 2**(n-1))


def map_AB(tensorA, tensorB):
    """
    Contract tensors A and B to form a transfer matrix.
    
    Contract tensors as:
    i -- A -- j    ,   k -- B -- l
         |                  |
    =
    i -- A -- j
         |
    k -- B -- l
    
    ===========
    Inputs 
        tensorA, tensorB (np.ndarray): Two iMPS tensors A & B
    ===========
    Outputs:
        (np.ndarray) Mixed transfer matrix of the iMPS tensors A and B with shape (i*k, j*l)
        
    """
    i, _, j = tensorA.shape
    k, _, l = tensorB.shape
    return np.einsum('inj, knl -> ikjl', tensorA, tensorB.conj()).reshape(i*k, j*l)


def right_fixed_point(E, all_evals= False):
    """
    Calculate the right fixed point of a transfer matrix E.
    
    ===========
    Inputs:
        E (np.ndarray): Transfer matrix with shape (N, N)
        all_evals (bool, optional): If True, return all eigenvalues along with the leading one. Default is False.
    
    ===========
    Outputs:
        mu (complex): The leading order eigenvalue of the transfer matrix E
        r  (np.ndarray): The right leading order eigenvector of the transfer matrix E
        sorted_evals (np.ndarray, optional): All eigenvalues in descending order of absolute value (only if all_evals=True)
    """
    evals, evecs = eig(E, left=False, right=True)
    sorted_evals = sorted(evals,reverse=True,key= np.abs)
    mu = sorted_evals[0]
    r = evecs[:,0]
    
    if all_evals:
        return mu, r, sorted_evals
    return mu, r


def left_fixed_point(E, all_evals = False):
    """
    Calculate the left fixed point of a transfer matrix E.
    
    ===========
    Inputs:
        E (np.ndarray): Transfer matrix with shape (N, N)
        all_evals (bool, optional): If True, return all eigenvalues along with the leading one. Default is False.
    
    ===========
    Outputs:
        mu (complex): The leading order eigenvalue of the transfer matrix E
        l  (np.ndarray): The left leading order eigenvector of the transfer matrix E
        sorted_evals (np.ndarray, optional): All eigenvalues in descending order of absolute value (only if all_evals=True)
    """
    evals, evecs = eig(E, left=True,right=False)
    sorted_evals = sorted(evals,reverse=True,key= np.abs)
    mu = sorted_evals[0]
    l = evecs[:,0]
    
    if all_evals:
        return mu, l, sorted_evals
    return mu, l


def map_AWB(tensorA, operator, tensorB):
    """
    Contract tensors A & B with operator W to form a transfer matrix with operators.
    
    Tensor structure:
    i -- A -- j    ,   k -- B -- l,    | |
         |                  |           W
                                       | |    
    =
    i -- A -- A -- j
         |    |
         W    W
         |    |
    k -- B -- B -- l
    
    ===========
    Inputs 
	    tensorA, tensorB (np.ndarray): Two iMPS tensors A & B with shape (2,2,2)
        operator (np.ndarray): 2 qubit unitary operator with shape (2,2,2,2)
    
    ===========
    Outputs:
        (np.ndarray) Mixed transfer matrix of the iMPS tensors A and B with shape (i*k, j*l)
    ===========
    Raises
        ValueError: If operator does not have shape (2, 2, 2, 2)
    """
    if operator.shape != (2, 2, 2, 2):
        raise ValueError(f"Expected operator with shape (2, 2, 2, 2), got {operator.shape}")
    
    i, _, j = tensorA.shape
    k, _, l = tensorB.shape
    tm = np.einsum('inm,moj,prno,kpq,qrl -> ikjl', tensorA, tensorA, operator, tensorB.conj(), tensorB.conj())
    return tm.reshape(i*k, j*l)


def param_to_tensor(params):
    """
    Convert parameters to an iMPS tensor via unitary ansatz stateAnsatzXZ.
    
    ===========
    Inputs 
        params (np.array): array of the parameters used to create the unitary
    ===========
    Outputs 
        A (np.ndarray): iMPS tensor created from parameters 'params'
    ===========
    Raises
        ValueError: If there aren't exactly 8 parameters
    """
    # check length of params
    if len(params) != 8:
        raise ValueError(f"Expected 8 parameters, got {len(params)}")
    
    a = stateAnsatzXZ(params)
    A = unitary_to_RCF_tensor(cirq.unitary(a))
    
    return A


def expectation(paramA, paramB, g, dt):
    """
    Calculate the expectation value of time evolution operator between two iMPS states.
    - works by calculating the principal eigenvalue of the transfer matrix:
    i -- A -- A -- j
         |    |
         W    W
         |    |
    k -- B -- B -- l
    where A is generated from by U(t) (parameterised by paramA), and B by U(t+dt) (parameterised by paramB)
    and g = g in the TFIM Hamiltonian used to create the evolution operator W
    ===========
    Inputs:
        paramA,paramB (np.array): parameter sets A & B which parametrize iMPS tensors A & B
        g (float): the transverse field magnitude in TFIM Hamiltonian
        dt (float): the time step for evolution
    ===========
    Outputs:
        overlap (float): the expectation value of the time evolution operator with the TFIM Hamiltonian and iMPS tensor A and B
    ===========
    Raises
        ValueError: If paramA or paramB don't have exactly 8 parameters
    """
    
    # check length of params
    if len(paramA) != 8:
        raise ValueError(f"Expected 8 parameters in paramA, got {len(paramA)}")
    if len(paramB) != 8:
        raise ValueError(f"Expected 8 parameters in paramB, got {len(paramB)}")
        
    A = param_to_tensor(paramA)
    B = param_to_tensor(paramB)
    W = evolution_op(g, dt)
    E = map_AWB(A, W, B)
    overlap = abs(right_fixed_point(E)[0])**2
    return overlap


def overlap(params1, params2):
    """
    Calculate the overlap between two MPS states.
    - works by calculating the principal eigenvalue of the transfer matrix:
    i -- A -- j
         |      
    k -- B -- l
    where A is parameterised by params1 and B by params2 
    ===========
    Inputs:
        paramA,paramB (np.array): parameter sets A & B which parametrize iMPS tensors A & B
    ===========
    Outputs:
        overlap (float): the overlap between two iMPS with tensors A & B respectively
    ===========
    Raises
        ValueError: If params1 or params2 don't have exactly 8 parameters
    """
    # check length of params
    if len(params1) != 8:
        raise ValueError(f"Expected 8 parameters in params1, got {len(params1)}")
    if len(params2) != 8:
        raise ValueError(f"Expected 8 parameters in params2, got {len(params2)}")
       
    A = param_to_tensor(params1)
    B = param_to_tensor(params2)
    E = map_AB(A, B)
    overlap = abs(right_fixed_point(E)[0])**2
    return overlap


def linFit(ya, yb):
    """
    Linear extrapolation from two points a and b with fixed x-axis gap of 0.2.
    
    ===========
    Inputs:
        ya,yb (np.array or float): the y values of the points to be extrapolated from 
    
    ===========
    Outputs:
        The extrapolated y value (np.array,float or int)
    """
    return 2 * yb - ya


def linExtrap(xa, xb, ya, yb, x):
    """
    Generic linear extrapolation from two points a and b.
    
    ===========
    Inputs:
        xa,xb (float): the x values of the points to be extrapolated from
        ya,yb (np.array or float): the y values of the points to be extrapolated from 
        x (float): the x value of the point extrapolating to
    ===========
    Outputs:
        y (np.array or float): The extrapolated y value 
    ===========
    Raises
        ValueError: If xa and xb are the same (division by zero)
    """
    if abs(xb - xa) < 1e-12:
        raise ValueError("X values are too close for stable extrapolation")
    
    y = ya + (x - xa) * (yb - ya) / (xb - xa)
    return y