import numpy as np

def getHopping(N, chemicalPotential=0):

    # Assume the first N/2 are full, the rest are empty
    p = int(N / 2)
    occupations = np.zeros(N)
    for occ in range(p): occupations[occ] = 1

    hoppingMatrix = np.zeros((N,N))

    for i in range(N):
        if i == 0:
            hoppingMatrix[i,i+1] = 1
            hoppingMatrix[i,N-1] = 1
        elif i == N-1:
            hoppingMatrix[i,0] = 1
            hoppingMatrix[i,i-1] = 1
        else:
            hoppingMatrix[i,i+1] = 1
            hoppingMatrix[i,i-1] = 1

    if chemicalPotential != 0:
        for i in range(N):
            hoppingMatrix[i,i] = -chemicalPotential
            
    return occupations, hoppingMatrix

def sortEigenstates(eigVals, eigVecs):
    idx = eigVals.argsort()
    eigVals = eigVals[idx]
    eigVecs = eigVecs[:,idx]
    return eigVals, eigVecs

if __name__ == "__main__":
    N = 5

    occupations, hoppingMatrix = getHopping(N)
    
    eigVals, eigVecs = np.linalg.eig(hoppingMatrix)
    eigVals, eigVecs = sortEigenstates(eigVals, eigVecs)
    diagMatrix = np.diag(eigVals)
    matrixUdag = eigVecs
    matrixU    = eigVecs.T

    expected_ni = np.zeros(N)
    filled_states = np.where(occupations == 1)[0]
    for i in range(N):
        for stateNum in filled_states:
            expected_ni[i] += np.abs(eigVecs[i, stateNum])**2

    devs =  expected_ni - expected_ni.sum()/N
