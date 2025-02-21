import cirq
import numpy as np
import ncon
from scipy.linalg import eig
from ansatz import stateAnsatzXZ
from Hamiltonian import evolution_op

def unitary_to_RCF_tensor(U):
    '''
    Take a unitary U and make it a right canonical tensor A such that
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
    ----------
    INPUTS:
        U: numpy.ndarray
            A 4x4 unitary representing a 2 qubit operation.
    OUTPUTS:
        A: numpy.ndarray
            A MPS tensor representation of the unitary U
    '''
    n=2
    zero=np.array([1.,0.])

    Ucontr = [-2,-3,-1,1]
    A=ncon.ncon([U.reshape(*2 *n *[2]),zero], [Ucontr,[1,]])
    A= A.reshape(2,2,2)
    return A

def unitary_to_LCF_tensor(U):
    '''
    Take a unitary U and make it a left canonical tensor A such that
    |0>     k
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
    Parameters
    ----------
    INPUTS:
        U: numpy.ndarray
            A 4x4 unitary representing a 2 qubit operation.
    OUTPUTS:
        A: numpy.ndarray
            A MPS tensor representation of the unitary U
    '''
    n = int(np.log2(U.shape[0]))
    zero = np.array([1., 0.])

    Ucontr = list(range(-1, -n-1, -1)) + [1] + list(range(-n-1, -2*n, -1))
    A = ncon.ncon([U.reshape(*2 * n *[2]), zero], [Ucontr, [1,]])
    A = A.reshape(2**(n-1), 2, 2**(n-1))
    return A    

def map_AB(tensorA, tensorB):
	'''Contract A, B as follows
	i -- A -- j    ,   k -- B -- l
	     |                  |
	=
	i -- A -- j
	     |
	k -- B -- l
	where the shape of the output is (i*k, j*l)
	Parameters
	----------
    INPUTS:
	    tensorA, tensorB: numpy.ndarray
            Two iMPS tensors A & B
    OUTPUTS:
        Mixed transfer matrix of the iMPS tensors A and B
	'''
	i, _, j = tensorA.shape
	k, _, l = tensorB.shape
	return np.einsum('inj, knl -> ikjl', tensorA, tensorB.conj()).reshape(i*k, j*l)


def right_fixed_point(E, all_evals=False):
    '''
    Calculate the right fixed point of a transfer matrix E
    E.shape = (N, N)
    INPUTS:
        E: numpy.ndarray
            Transfer matrix E
    OUTPUTS:
        mu: the leading order eigenvalue of the transfer matrix E
        r: the right leading order eigenvector of the transfer matrix E
    '''
    evals, evecs = eig(E, left=False,right=True)
    sorted_evals = sorted(evals,reverse=True,key= np.abs)
    mu = sorted_evals[0]
    r = evecs[:,0]
    
    return mu,r

    
def left_fixed_point(E, all_evals=False):
    '''
    Calculate the left fixed point of a transfer matrix E
    E.shape = (N, N)
    INPUTS:
        E: numpy.ndarray
            Transfer matrix E
    OUTPUTS:
        mu: the leading order eigenvalue of the transfer matrix E
        l: the left leading order eigenvector of the transfer matrix E
    '''
    evals, evecs = eig(E, left=True,right=False)
    sorted_evals = sorted(evals,reverse=True,key= np.abs)
    mu = sorted_evals[0]
    l = evecs[:,0]

    return mu, l

def map_AWB(A,W,B):
    '''Contract A, W, B as follows
    i -- A -- j    ,   k -- B -- l,    | |
         |                  |           W
                                       | |    
    =
    i -- A -- A -- j
         |    |
         W    W
         |    |
    k -- B -- B -- l
    where the shape of the output is (i*k, j*l)
    Parameters
    ----------
    INPUTS
    A, W, B: numpy.ndarray
        A,B: iMPS tensors A & B
        W: evolution operator
    OUTPUTS:
        Mixed transfer matrix of the iMPS tensors A and B with evolution operator W
    '''
    i,_,j = A.shape
    k,_,l = B.shape
    tm = np.einsum('inm,moj,prno,kpq,qrl -> ikjl', A,A,W,B.conj(),B.conj())
    return tm.reshape(i*k, j*l)

def param_to_tensor(params):
    '''
    Converts the unitary parameterised by 'params' into an MPS tensor
    INPUTS:
        params: (np.ndarray) array of the parameters used to create the unitary
    OUTPUTS:
        A: (np.ndarray) iMPS tensor created from parameters 'params'
    '''
    a = stateAnsatzXZ(params)
    A = unitary_to_RCF_tensor(cirq.unitary(a))
    return A

def expectation(paramA,paramB,g,dt):
    '''
    Calculate the principal eigenvalue of the transfer matrix
    i -- A -- A -- j
         |    |
         W    W
         |    |
    k -- B -- B -- l
    where A is generated from by U(t) (parameterised by paramA), and B by U(t+dt) (parameterised by paramB)
    and g = g in the TFIM Hamiltonian used to create the evolution operator W
    INPUTS:
        paramA,paramB: (np.ndarray) parameter sets A & B which parametrize iMPS tensors A & B
        g: (float) the transverse field magnitude
        dt: (float) the time step
    OUTPUTS:
        overlap: (float) the expectation value of the time evolution operator with the TFIM Hamiltonian and iMPS tensor A and B
    '''
    A = param_to_tensor(paramA)
    B = param_to_tensor(paramB)
    W = evolution_op(g,dt)
    E = map_AWB(A,W,B)
    overlap = abs(right_fixed_point(E)[0])**2
    return overlap

def overlap(params1,params2):
    '''
    Calculate the principle eigenvalue of the transfer matrix
    i -- A -- j
         |      
    k -- B -- l
    where A is parameterised by params1 and B by params2 
    INPUTS:
        paramA,paramB: (np.ndarray) parameter sets A & B which parametrize iMPS tensors A & B
    OUTPUTS:
        overlap: (float) the overlap between two iMPS with tensors A & B respectively
    '''
    A = param_to_tensor(params1)
    B = param_to_tensor(params2)
    E = map_AB(A,B)
    overlap = abs(right_fixed_point(E)[0])**2
    return overlap

def linFit(a,b):
    '''Calculates a linear extrapolation from the two previous points a and b '''
    return 2*b - a