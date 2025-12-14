# Model that simulates 
# an adiabatic quantum computation 
# to factor biprime numbers

from math import floor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def getQubitsNumber(n):

    sqrt_ = np.sqrt(n)

    if np.abs(np.log2(sqrt_) / floor(np.log2(sqrt_)) - 1) < 1e-5:
        return floor(np.log2(sqrt_))
    else:
        return floor(np.log2(sqrt_)) + 1

def getPauliMatrix(coord):

    if coord == "x": return np.array([[0,1],[1,0]])
    if coord == "y": return np.array([[0,-np.sqrt(-1)],[np.sqrt(-1),0]])
    if coord == "z": return np.array([[1,0],[0,-1]])


def buildH0(bitsNum, inputMat):

    qubitDim = 2

    matrixList = [inputMat]
    matrixList += [np.identity(qubitDim)] * (bitsNum - 1)
    matrixList = [np.identity(qubitDim)] * bitsNum

    fullMatrix = np.zeros((2**bitsNum, 2**bitsNum))
    indQubitMatrices = {}
    for i in range(bitsNum):
        matrixList[i] = inputMat
        for j in range(len(matrixList)):
            if j == 0:
                qubitMat = matrixList[j]
            else:
                qubitMat = np.kron(matrixList[j], qubitMat)

        newMatrixList = [matrixList[-1]] + matrixList[:-1]
        matrixList    = newMatrixList.copy()

        indQubitMatrices[i] = qubitMat
        fullMatrix += qubitMat


    return fullMatrix, indQubitMatrices

def buildH1(bitsNum, numToFactor, numberStates=None):

    fullMatrix = np.zeros((2**bitsNum, 2**bitsNum))

    if numberStates: iterValues = list(numberStates.keys())
    else:            iterValues = list(range(2 ** bitsNum))
    for num in iterValues:
        pos = iterValues.index(num)
        try: fullMatrix[pos][pos] = numToFactor % num
        except ZeroDivisionError: fullMatrix[pos][pos] = 0

    return fullMatrix

def getNumberStates(bitsNum):

    indQubitzMatrices = buildH0(bitsNum, getPauliMatrix("z"))[1]

    # qubit states with form x=|p1 p2 p3 p4>, with x=p1*2^0+p2*2^1+p3*2^2+p4*2^3
    # sigma_z |0> = +|0>
    # sigma_z |1> = -|1>

    numberOfStates = 2**bitsNum
    baseState = np.array([[0] for i in range(numberOfStates)])

    indQubitNumbers = [0] * numberOfStates

    # Build all eigenvectors
    allIndStates = []
    statesNumberDict = {}
    for i in range(numberOfStates):
        indState = baseState.copy()
        indState[i] = 1
        allIndStates += [indState]

        indNum = [0] * bitsNum # Number in bits as a list of 0 and 1
        for qubitSpace in range(bitsNum):
            indQubitzMatrix = indQubitzMatrices[qubitSpace]
            indSigmazEigVec = np.matmul(indQubitzMatrix, indState)

            if (indSigmazEigVec == indState).all():
                qubitSpaceValue = 0
            elif (indSigmazEigVec == - indState).all():
                qubitSpaceValue = 1
            else: print("State:", i, "Substate:", qubitSpace)

            indNum[qubitSpace] = qubitSpaceValue

        # Check what eigenvector is for each number
        number = 0
        for qubitSpace in range(bitsNum):
            number += indNum[qubitSpace] * 2**qubitSpace

        statesNumberDict[number] = {} 
        statesNumberDict[number]["State"] = indState 
        statesNumberDict[number]["Binary"] = indNum 
        statesNumberDict[number]["BinaryString"] = "".join([str(indNum_) for indNum_ in indNum])


    return statesNumberDict

def getReducedDensityMatrix(qubitsNum, qubitsNumA, rho, twoQubit_numberStates=None, dtype=None):
    '''
    Get reduced density matrix from rho_AB
    Trace over two-qubit subspace
    rho_A = Tr_B (rho) = sum_i I_A ox <i|_B rho_AB I_A ox |i>_B
    '''
    if twoQubit_numberStates is None:
        twoQubit_numberStates = getNumberStates(qubitsNumA)
    rho_reduced = np.zeros((qubitsNumA**2, qubitsNumA**2), dtype=dtype)
    for stateNum in twoQubit_numberStates:
        stateA = np.identity(qubitsNum)
        stateB = twoQubit_numberStates[stateNum]["State"]
        leftProduct  = np.kron(stateA, np.atleast_2d(stateB).T).astype(dtype)
        rightProduct = np.kron(stateA, stateB).astype(dtype)
        rho_reduced += np.matmul(np.matmul(leftProduct, rho), rightProduct)
    return rho_reduced


def factorNumber(numToFactor, lambdaGap=0.01, dropRows = False, plotData = True, exportData=False):

    qubitsNum = getQubitsNumber(numToFactor) # Required bits number of the factor we are searching
    qubitsNumA = qubitsNum // 2
    qubitsNumB = qubitsNum - qubitsNumA

    numberStates = getNumberStates(qubitsNum) # Dictionary with entry {number: state}, with number in base 10 and base in the computational basis

    # Build decomposition in two 2-qubit subspaces
    twoQubit_numberStates = getNumberStates(qubitsNumA)

    # +2 because we discard first numbers (0 and 1), with eigenvalue 0 both, but we are not looking them
    trueNumberStates = {}
    for num in numberStates: trueNumberStates[num + 2] = {"State": numberStates[num]["State"]}

    H0 = buildH0(qubitsNum, getPauliMatrix("x"))[0]
    H1 = buildH1(qubitsNum, numToFactor, trueNumberStates)

    if dropRows:
        H0 = np.delete(H0, [0,1], axis=0)
        H0 = np.delete(H0, [0,1], axis=1)
        H1 = np.delete(H1, [0,1], axis=0)
        H1 = np.delete(H1, [0,1], axis=1)
    lambdaValues  = [i * lambdaGap for i in range(int(1/lambdaGap) + 1)]
    gapValues     = []
    e1Values      = []
    e2Values      = []
    entropyValues = []
    for lam in lambdaValues:
        H_lam = lam * H1 + (1-lam) * H0

        eigVals, eigVecs = np.linalg.eigh(H_lam)

        e_1 = min(eigVals)
        groundState = eigVecs[:, eigVals.argmin()]

        # Entropy: partial trace over rho
        # rho = |gs> ox <gs|
        rho = np.outer(groundState, groundState)

        # Get reduced rho_A = Tr_B (rho) = sum_i I ox <i| rho I ox |i>
        rho_reduced = getReducedDensityMatrix(qubitsNum, qubitsNumA, rho, twoQubit_numberStates)
        
        # Calculate entropy (use trace of matrix is eq to sum of eigvalues)
        redEigVals = np.linalg.eigvalsh(rho_reduced)
        entropy = np.array([ - eig * np.log2(eig) for eig in redEigVals if eig > 1e-10]).sum()

        eigVals = list(eigVals)
        eigVals.remove(e_1)
        e_2 = min(eigVals)

        gap = e_2 - e_1
        gapValues     += [gap]
        e1Values      += [e_1]
        e2Values      += [e_2]
        entropyValues += [entropy]

    minGapValue = min(gapValues)
    minLambda   = lambdaValues[gapValues.index(minGapValue)]
    maxEntropy  = max(entropyValues)
    timeAQC     = sum([1 / gap ** 2 * lambdaGap for gap in gapValues])
    print("Minimum value of the gap:", minGapValue)
    print("Minimum value of lambda:", minLambda)
    print("Maximum value of entropy:", maxEntropy)
    print("Time of AQC:", timeAQC)

    if plotData:
        plt.figure(1)   
        plt.title(f"Gap energético en función de $\lambda$ (N={numToFactor})")
        plt.xlabel("$\lambda$")
        plt.ylabel("$E_2 (\lambda) - E_1 (\lambda)$")
        plt.plot(lambdaValues, gapValues, label="$E_2 (\lambda) - E_1 (\lambda)$")
        plt.legend()
        plt.show()

        plt.figure(2)
        plt.title(f"Entropía en función de $\lambda$ (N={numToFactor})")
        plt.xlabel("$\lambda$")
        plt.ylabel("$S (\lambda)$")
        plt.plot(lambdaValues, entropyValues, label="$S (\lambda)$")
        plt.legend()
        plt.show()

    if exportData:
        df = pd.DataFrame(index=lambdaValues)
        df["Gap"]     = np.array(gapValues)
        df["E_1"]     = np.array(e1Values)
        df["E_2"]     = np.array(e2Values)
        df["Entropy"] = np.array(entropyValues)

    return minGapValue, minLambda, maxEntropy, timeAQC

def checkFactors(bitsNum, numList):

    trueList = []
    for num in numList:
        maxNum = 2 ** (bitsNum * 2 - 1)
        numsInRange = list(range(2, 2**bitsNum + 2))

        for i in range(2, maxNum):
            if num % i == 0:
                div1 = int(i)
                break

        div2 = int(num / div1)

        if div1 in numsInRange and div2 in numsInRange:
            pass
        else:
            trueList += [num]

    return trueList
            
def compareNumbers(numList, plotEntGap=True, plotGapTimeAQC=False, printTable=True):
    biprimesList = checkFactors(4, numList)
    values = pd.DataFrame(columns=["Number", "Max Entropy", "Min Gap"])
    idx = 0
    for prime in biprimesList:
        minGapValue, minLambda, maxEntropy, timeAQC = factorNumber(prime, plotData=False)
        values.loc[idx, "Number"]      = prime
        values.loc[idx, "Max Entropy"] = round(maxEntropy, 2)
        values.loc[idx, "Min Gap"]     = round(minGapValue, 2)
        values.loc[idx, "Time AQC"]    = round(timeAQC, 2)
        idx += 1

    if plotEntGap:
        fig, ax = plt.subplots()
        x = values["Min Gap"].values
        y = values["Max Entropy"].values
        n = values["Number"].values
        ax.scatter(x, y)
        for i, txt in enumerate(n):
            ax.annotate(txt, (x[i], y[i]))
        plt.xlabel("Valor mínimo del gap")
        plt.ylabel("Valor máximo de la entropía")
        plt.title("Gap mínimo y entropía máxima")

    if plotGapTimeAQC:
        fig, ax = plt.subplots()
        x = values["Min Gap"].values
        y = values["Time AQC"].values
        n = values["Number"].values
        ax.scatter(x, y)
        for i, txt in enumerate(n):
            ax.annotate(txt, (x[i], y[i]))
        plt.xlabel("Valor mínimo del gap")
        plt.ylabel("Tiempo de computación")
        plt.title("Gap mínimo y tiempo de AQC")
        plt.show()
    
    if printTable:
        print(values.to_latex(index=False))


if __name__ == "__main__":
    biprimesList_ = [65, 69, 74, 77, 82, 85, 86, 87, 91,
        93, 94, 95, 106, 111, 115, 118, 119, 121, 122, 123,
        129, 133, 134, 141, 142, 145, 146, 155, 158,
        159, 161, 166, 169, 177, 178, 183, 185, 187
    ]

    # Factor main number of problem (121=11x11)
    factorNumber(121)

    # Factor number with two factors in range (not valid, 143=11x13)
    factorNumber(143)

    # Compare different numbers in range
    compareNumbers(biprimesList_, plotEntGap=True, plotGapTimeAQC=True, printTable=True)
