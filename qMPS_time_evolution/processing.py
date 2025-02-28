import numpy as np

def bitwiseAND(results):
    '''
    Returns the bitwise AND of the two sets of measurement results from running on Quantinuum device
    =============
    Inputs:
        results: result data from device run
    =============
	Outputs:
        bitstrings (np.array): bitstrings of the bitwise ANDs
    '''
    keys = [key for key in results.keys()]
    length = int(len(results[keys[0]]))
    
    ands = []
    for j in range(int(len(keys)/2)):
        for i in range(length):
            curr_psi = [int(item) for sub in results[keys[(j*2)+1]][i] for item in sub]
            curr_phi = [int(item) for sub in results[keys[j*2]][i] for item in sub]
            ands.append(np.bitwise_and(curr_psi, curr_phi))#.flatten())
    bitstrings = np.array(ands)
    return bitstrings

def evolSwapTestRatio(results, length):
    '''
    Calculates the overlap from the measurement results
        Performs the swap test on the bitstring produced by bitwiseAND of the measurement results by checking the parity and counting when it is even
        Calculates the ratio of the that of the full circuit of length 'length' and that with one transfer matrix removed 
    =============
    Inputs:
        results: result data from device run
        length (int): how many pairs of measurements there are per circuit
    =============
	Outputs:
        overlap (float): the overlap between the two states being measured
	'''
    bitstrings = bitwiseAND(results)
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

    overlap = (overlap_full/overlap_minus)**2

    return overlap