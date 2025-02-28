from qtuum.api_wrappers import QuantinuumAPI as QAPI
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
import cirq
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from SPSA import minimizeSPSA
from classical import expectation, overlap, linExtrap
from Loschmidt import loschmidt_paper
from ansatz import stateAnsatzXZ
from processing import evolSwapTestRatio
from Hamiltonian import Hamiltonian
from simulation import simulate_noiseless,swapTest,calc_and
from circuits import higherTrotterQasm

g0, g1 = 1.5, 0.2
max_time = 2
ltimes = np.linspace(0.0, max_time, 800)
correct_ls = [loschmidt_paper(t, g0, g1) for t in ltimes]

paramData = np.load('TMparams100000.npy')
x0 = paramData[0]
x1 = paramData[1]
x2 = paramData[2]


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