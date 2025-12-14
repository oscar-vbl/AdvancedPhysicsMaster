import numpy as np
import sys, os
sys.path.append(os.getcwd())
from OtherScripts.QuantumComputation.AQC_NumberFactorisation import getNumberStates, getReducedDensityMatrix


sigma_x = np.array([[0,1],[1,0]],    dtype=np.complex128)
sigma_y = np.array([[0,-1j],[1j,0]], dtype=np.complex128)
sigma_z = np.array([[1,0],[0,-1]],   dtype=np.complex128)

eps = 1e-5

def insertTensorProduct():

    pass

def getIsingHamiltonian_(numQubits, gamma):
    ham = np.zeros((2**numQubits, 2**numQubits), dtype=np.complex128)
    for i in range(numQubits-1):
        if i == 0:
            m_1 = sigma_z
            m_2 = sigma_z
            m_3 = np.array([[1]])
            for j in range(numQubits - 2):
                m_3 = np.kron(m_3, np.identity(2))
        elif i == numQubits - 2:
            m_1 = np.array([[1]])
            for j in range(numQubits - 2):
                m_1 = np.kron(m_1, np.identity(2))
            m_2 = sigma_z
            m_3 = sigma_z

        else:
            m_1 = np.array([[1]])
            for j in range(i):
                m_1 = np.kron(m_1, np.identity(2))
            m_2 = np.kron(sigma_z, sigma_z)
            m_3 = np.array([[1]])
            for j in range((numQubits - 1) - (i+1)):
                m_3 = np.kron(m_3, np.identity(2))

        ham += (-1) * np.kron(m_1, np.kron(m_2, m_3))

    return ham

def matrixFromString(mat):

    matrix = np.array([[1]])
    for char in mat:
        if char == "I":   matrix_i = np.identity(2)
        elif char == "X": matrix_i = sigma_x
        elif char == "Y": matrix_i = sigma_y
        elif char == "Z": matrix_i = sigma_z
        else:             matrix_i = char * np.identity(2)
        matrix = np.kron(matrix, matrix_i)
    return matrix

def getIsingHamiltonian(numQubits, gamma):
    ham = np.zeros((2**numQubits, 2**numQubits), dtype=np.complex128)
    for i in range(numQubits-1):
        base      = ["I"] * numQubits
        base[i]   = "Z"
        base[i+1] = "Z"

        ham += (-1) * matrixFromString(base)

    if gamma != 0:
        for i in range(numQubits):
            base      = ["I"] * numQubits
            base[i]   = "X"
        
        ham += (-1) * gamma * matrixFromString(base)

    return ham

def getMagnetization(numQubits, state, axis="Z"):
    '''Average magnetization along given axis for a given state.'''

    avMag = 0
    for i in range(numQubits):
        base    = ["I"] * numQubits
        base[i] = axis
        M_gs_i = matrixFromString(base)
        val = state @ M_gs_i @ gsStateKet
        print("".join(base), val)
        avMag += 1 / (numQubits) * val.real
    return avMag

def getSpinSpinCorrelation(numQubits, state, axis="Z"):
    '''Spin-spin correlation function between nearest neighbors.'''
    sig_sig_ij = 0
    for i in range(numQubits-1):
        base    = ["I"] * numQubits
        base[i]   = axis
        base[i+1] = axis
        M_gs_ij = matrixFromString(base)
        val = state.conj().T @ M_gs_ij @ state
        print("".join(base), val)
        sig_sig_ij += 1 / (numQubits) * val.real
    return avMag

def getExpectedValue(numQubits, state, axis="Z", operator="Magnetization"):
    '''Spin-spin correlation function between nearest neighbors.'''
    sig_sig_ij = 0
    for i in range(numQubits):
        base    = ["I"] * numQubits
        if operator == "Magnetization":
            base[i] = axis
            
        if operator == "Spin-Spin Correlation":
            if i == numQubits - 1: continue
            base[i]   = axis
            base[i+1] = axis

        M_gs_ij = matrixFromString(base)
        val = state.conj().T @ M_gs_ij @ state
        print("".join(base), val)
        sig_sig_ij += val.real
    
    avMag = sig_sig_ij / numQubits
    return avMag

def getExpectedValues(numQubits, gamma_values, operators):
    expValuesOverGamma = {}
    numberStates = getNumberStates(numQubits)
    for gamma in gamma_values:
        expValuesOverGamma[gamma] = {}
        for operator in operators:
            for method in ["Max", "Min", "Mean"]:
                for axis in ["Z", "X"]:
                    expValuesOverGamma[gamma][f"{operator} {axis}-{method}"] = ""

        expValuesOverGamma[gamma] = {
            "Magnetization Z-Max":  "",
            "Magnetization Z-Min":  "",
            "Magnetization Z-Mean": "",
            "Magnetization X-Max":  "",
            "Magnetization X-Min":  "",
            "Magnetization X-Mean": "",
            "Spin-Spin Correlation X-Max":  "",
            "Spin-Spin Correlation X-Min":  "",
            "Spin-Spin Correlation X-Mean": "",
            "Spin-Spin Correlation Z-Max":  "",
            "Spin-Spin Correlation Z-Min":  "",
            "Spin-Spin Correlation Z-Mean": "",
            "Energy Gap": "",
            "Entanglement Entropy": "",
        }
        ham = getIsingHamiltonian(numQubits, gamma)
        eigVals, eigVecs = np.linalg.eig(ham)
        gsStates = np.where(np.abs(eigVals - eigVals.min()) < eps)[0]

        for axis in ["X", "Z"]:
            # Average axis magnetization
            for operator in operators:
                if operator not in ["Magnetization", "Spin-Spin Correlation"]:
                    continue
                if len(gsStates) > 1:
                    avOpMax, avOpMin = None, None
                    # Below gamma_c, degenerate ground state
                    for gsIdx in gsStates:
                        gsState = eigVecs[gsIdx]
                        avOp = getExpectedValue(numQubits, gsState, axis=axis, operator=operator)
                        if avOp > 0:
                            avOpMax = avOp
                        elif avOp < 0:
                            avOpMin = avOp
                    if avOpMax and avOpMin: avOpMean = np.mean([avOpMax, avOpMin])
                    elif avOpMax: avOpMean = avOpMax; avOpMin = avOpMax
                    elif avOpMin: avOpMean = avOpMin; avOpMax = avOpMin
                    else: avOpMax = avOp; avOpMin = avOp; avOpMean = avOp
                else:
                    gsState = eigVecs[gsStates[0]]
                    # Average axis magnetization
                    avOp = getExpectedValue(numQubits, gsState, axis=axis, operator=operator)
                    avOpMax  = avOp
                    avOpMin  = avOp
                    avOpMean = avOp

                expValuesOverGamma[gamma][f"{operator} {axis}-Max"]  = avOpMax
                expValuesOverGamma[gamma][f"{operator} {axis}-Min"]  = avOpMin
                expValuesOverGamma[gamma][f"{operator} {axis}-Mean"] = avOpMean

        expValuesOverGamma[gamma]["Energy Gap"] = np.min(np.delete(eigVals, gsStates)) - eigVals.min()
        # Entanglement entropy across bipartition
        qubitsNumA = numQubits // 2
        rho = np.zeros((2**numQubits, 2**numQubits), dtype=np.complex128)
        for groundStateIdx in gsStates:
            # Suppose rho is a weighted sum over degenerate GS states
            # We take equal weights
            groundState = eigVecs[:, groundStateIdx]
            rho += 1 / np.sqrt(len(gsStates)) * np.outer(groundState, groundState)
        reduced_rho = getReducedDensityMatrix(numQubits, qubitsNumA, rho, dtype=np.complex128)
        # Calculate entropy (use trace of matrix is eq to sum of eigvalues)
        redEigVals = np.linalg.eigvalsh(reduced_rho)
        # 0 * log(0) is defined as 0
        entropy = np.array([ - eig * np.log2(eig) for eig in redEigVals if eig > 1e-10]).sum()
        expValuesOverGamma[gamma]["Entanglement Entropy"] = entropy

    return expValuesOverGamma

def plotData(valuesOverGamma, operators):
    import matplotlib.pyplot as plt 
    for operator in operators:
        if operator in ["Energy Gap", "Entanglement Entropy"]:
            plt.figure()
            plt.title(f"{operator} vs Gamma")
            plt.xlabel("Gamma")
            plt.ylabel(f"{operator}")
            plt.xscale("log")
            gamma_values = sorted(valuesOverGamma.keys())
            mean_values = [valuesOverGamma[gamma][f"{operator}"] for gamma in gamma_values]
            plt.plot(gamma_values, mean_values, marker='o')
            plt.axvline(x=1.0, color='r', linestyle='--', label='Critical Point (Gamma=1)')
            plt.legend()
            plt.grid()
        else:
            for axis in ["Z", "X"]:
                plt.figure()
                plt.title(f"{operator} {axis} vs Gamma")
                plt.xlabel("Gamma")
                plt.ylabel(f"{operator} {axis}")
                plt.xscale("log")
                gamma_values = sorted(valuesOverGamma.keys())
                max_values = [valuesOverGamma[gamma][f"{operator} {axis}-Max"] for gamma in gamma_values]
                min_values = [valuesOverGamma[gamma][f"{operator} {axis}-Min"] for gamma in gamma_values]
                mean_values = [valuesOverGamma[gamma][f"{operator} {axis}-Mean"] for gamma in gamma_values]
                plt.plot(gamma_values, max_values, label="Max", marker='o')
                plt.plot(gamma_values, min_values, label="Min", marker='o')
                plt.plot(gamma_values, mean_values, label="Mean", marker='o')
                plt.axvline(x=1.0, color='r', linestyle='--', label='Critical Point (Gamma=1)')
                plt.legend()
                plt.grid()
    plt.show()

if __name__ == "__main__":
    numQubits=4
    gamma_values = np.linspace(0,1,10)
    gamma_values = np.concatenate([gamma_values, np.linspace(1,5,20) ])
    gamma_values = np.concatenate([gamma_values, np.linspace(5,50,10)])
    operators = [
        "Magnetization",
        "Spin-Spin Correlation",
        "Energy Gap",
        "Entanglement Entropy",
    ]
    gamma = 1
    ham = getIsingHamiltonian(numQubits, gamma)

    values = getExpectedValues(numQubits, gamma_values, operators)
    plotData(values, operators)
    
    eigVals, eigVecs = np.linalg.eig(ham)
    gsStates = np.where(eigVals == eigVals.min())[0]
    avMag = 0
    for gsIdx in gsStates:
        gsStateBra = eigVecs[gsIdx]
        gsStateKet = np.atleast_2d(eigVecs[gsIdx]).T
        for i in range(numQubits):
            base = ["I"] * numQubits
            base[i] = "Z"
            M_gs_i = matrixFromString(base)
            val = np.matmul(gsStateBra, np.matmul(M_gs_i,gsStateKet ))[0]
            print("".join(base), val)
            avMag += 1 / (numQubits * len(gsStates)) * val

    np.matmul(np.matmul(np.linalg.eigh(ham)[1][0], matrixFromString("IXXI")), np.atleast_2d(np.linalg.eigh(ham)[1][0]).T)
