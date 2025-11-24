# Algorith to solve differential equations 
# Using Galerkin Finite Element Method
# with linear and quadratic interpolation functions
# given domain and contour conditions.
# Equation has the form:
#   d^2u/dx^2 + q(x)u = f(x)
# Contour conditions can be Dirichlet or Neumann

import numpy as np
from sympy import *
from sympy.abc import x, y
import matplotlib.pyplot as plt

def equationSolver(numElements, leftLimit, rightLimit,
        qFunc, fFunc, contourConds, analyticalSol,
        method="Linear", numDec=5, plotError=False):

    if method == "Linear": results = solveSystemLinearInterp(numElements, leftLimit, rightLimit, qFunc, fFunc, contourConds, analyticalSol)
    if method == "Quad":   results = solveSystemQuadInterp(numElements, leftLimit, rightLimit, qFunc, fFunc, contourConds, analyticalSol)
    
    return results

def gaussSeidelSolver(K, b, symsMat, tol=1e-10, maxIter=10000):
    '''
    Solve the linear system K * u = b using the Gauss-Seidel method
    K: Coefficient matrix (sympy Matrix)
    b: Right-hand side vector (sympy Matrix)
    symsMat: Symbols for the unknowns (sympy Matrix)
    '''
    # Gauss-Seidel
    n = shape(K)[0]
    m = shape(K)[1]

    #  valores iniciales
    u = zeros(shape(b)[0],shape(b)[1])

    diff = ones(n, 1)
    error = 2*tol

    iter = 0
    while not(error<=tol or iter>maxIter):
        # por fila
        for i in range(n):
            # por columna
            sum = 0 
            for j in range(m):
                # excepto diagonal de A
                if (i!=j): 
                    sum -= K[i,j]*u[j,0]
            
            newItem = float((b[i,0]+sum)/K[i,i])
            diff[i,0] = np.abs(newItem-u[i,0])
            u[i,0] = newItem
        error = max([val for val in diff.values()])
        iter += 1

    print(iter, "Iteraciones")

    uSol = [{}]
    for row in range(shape(symsMat)[0]):
        uSol[0][symsMat[row,0]] = u[row,0]

    return uSol

def applyCountourConditions(contourConds, K, b, uVec, derSyms, rows, cols):
    '''
    Apply contour conditions, with conditions as a list of dicts:
        "Type": "Neu" (Neumann), "Dir" (Dirichlet)

        "Loc":  "First" (first element), "Last" (last element)

        "Value": Numerical value
    '''
    for cond in contourConds:
        condType = cond["Type"]
        loc      = cond["Loc"]
        value    = cond["Value"]

        if condType == "Dir":
            if loc == "First":
                knownVal = K[1,0] * value
                b[1,0] = b[1,0] - knownVal
                K.row_del(0)
                b.row_del(0)
                uVec.row_del(0)
                rows -= 1
                K.col_del(0)
                cols -= 1
            elif loc == "Last":
                knownVal = K[rows-1-1,rows-1] * value
                b[rows-1-1,rows-1] = b[rows-1-1,rows-1] - knownVal
                K.row_del(rows-1)
                b.row_del(rows-1)
                uVec.row_del(rows-1)
                rows -= 1
                K.col_del(cols-1)
                cols -= 1
        
        elif condType == "Neu":
            if loc  == "First":
                expr = b[1, 0]
                expr = expr.subs(derSyms[-1], value)
                b[1, 0] = expr
            elif loc  == "Last":
                expr = b[rows-1, 0]
                expr = expr.subs(derSyms[-1], value)
                b[rows-1, 0] = expr

    return K, b, uVec

def solveSystemLinearInterp(numElements, leftLimit, rightLimit, qFunc, fFunc, contourConds, analyticalSol):

    hVal = Rational((rightLimit - leftLimit) / numElements)

    h = symbols("h")

    # Build K matrix 
    K = zeros(numElements+1, numElements+1)
    
    uSyms = list(symbols(f"u0:{numElements+1}"))
    derSyms = [symbols("dua"),symbols("dub")]

    b = zeros(numElements+1,1)

    qAv_i, fAv_i = [], []

    for i in range(numElements):
        xL, xR = i * h, (i+1) *h

        try: qAv = Rational(0.5)*(qFunc.subs(x, xL) + qFunc.subs(x, xR))
        except: qAv = qFunc # If q is constant for ex.

        try: fAv = Rational(0.5)*(fFunc.subs(x, xL) + fFunc.subs(x, xR))
        except: fAv = fFunc # If f is constant for ex.

        qAv_i += [qAv]
        fAv_i += [fAv]

    for i in range(numElements+1):
        xL, xR = i * h, (i+1) * h

        n = i-1

        # Build K
        if i == 0:
            K[i,i] = 1/h - qAv_i[n+1] * h / 3

        else:
            if i == numElements:
                K[i,i] = 1/h - qAv_i[n] * h / 3

            else:
                K[i,i] = (1/h - qAv_i[n] * h / 3) + (1/h - qAv_i[n+1] * h / 3)
            
            K[i,i-1] = -1/h - qAv_i[n] * h / 6
            K[i-1,i] = -1/h - qAv_i[n] * h / 6

        # Now build b
        if i == 0:
            b[i,0] = - fAv_i[n+1] * h / 2 - derSyms[0]

        elif i == numElements:
            b[i,0] = - fAv_i[n] * h / 2 + derSyms[1]

        else:
            b[i,0] = - fAv_i[n+1] * h / 2 - fAv_i[n] * h / 2

    uVec = Matrix(uSyms)

    print_latex(K.subs(h, hVal))
    print_latex(uVec)
    print_latex(b.subs(h, hVal))

    # Now apply contour conditions
    rows, cols = numElements+1, numElements+1
    K, b, uVec = applyCountourConditions(contourConds, K, b, uVec, derSyms, rows, cols)

    sist = gaussSeidelSolver(K.subs(h, hVal), b.subs(h, hVal), uVec)
    
    numSol = []
    for key in sist[0]: numSol += [sist[0][key]]
    xList = [i*hVal for i in range(numElements+1)]
    anSol = [analyticalSol.subs(x, i*hVal) for i in range(numElements+1)]

    return [float(x) for x in numSol], [float(x) for x in anSol[1:]], xList

def solveSystemQuadInterp(numElements, leftLimit, rightLimit, qFunc, fFunc, contourConds, analyticalSol):

    hVal = Rational((rightLimit - leftLimit) / numElements)


    h = symbols("h")
    x = symbols("x")
    
    cL, cC, cR = symbols("cL"),symbols("cC"),symbols("cR")
    xL, xC, xR = symbols("xL"),symbols("xC"),symbols("xR")

    N_L =   (2/h**2) * (x - xC) * (x - xR)
    N_C = - (4/h**2) * (x - xL) * (x - xR)
    N_R =   (2/h**2) * (x - xL) * (x - xC)

    uFunc  = cL * N_L + cC * N_C + cR * N_R

    i = 0
    #xL, xC, xR = i * h/2, h/2, (i+1) * h/2
    interpFuncs = {}
    for i in range(numElements):
        element = i+1
        interpFuncs[element] = {"L":"","C":"","R":""}
        xLVal, xCVal, xRVal = Rational(i) * h, Rational((i+1/2)) * h, Rational((i+1)) * h

        for N_K in [N_L,N_C,N_R]:

            integrand = - diff(uFunc, x) * diff(N_K, x) + qFunc * uFunc * N_K - fFunc * N_K

            indInt = integrate(integrand, x)

            integral = indInt.subs(x, xR) - indInt.subs(x, xL)

            integralVal = integral.subs(xL,xLVal).subs(xC,xCVal).subs(xR,xRVal)

            value = simplify(integralVal)

            if N_K == N_L:   interpFuncs[element]["L"] = value
            elif N_K == N_C: interpFuncs[element]["C"] = value
            elif N_K == N_R: interpFuncs[element]["R"] = value

    # Build K matrix 
    K = zeros(2*numElements+1, 2*numElements+1)

    uSyms = list(symbols(f"u0:{2*numElements+1}"))
    derSyms = [symbols("dua"),symbols("dub")]

    b = zeros(2*numElements+1,1)

    firstCol = 2
    for i in range(numElements):
        element = i+1

        xLVal, xCVal, xRVal = i * h, (i+1/2) * h, (i+1) * h

        if element == 1: # For first element
            # Left
            elementFirstRow = interpFuncs[element]["L"].copy()
            indepTerm = - elementFirstRow.subs(cL,0).subs(cC,0).subs(cR,0)
            elementFirstRow += indepTerm
            indepTerm -= derSyms[0]

            kRow = 0

            K[kRow,firstCol-2] = elementFirstRow.subs(cL,1).subs(cC,0).subs(cR,0)
            K[kRow,firstCol-1] = elementFirstRow.subs(cL,0).subs(cC,1).subs(cR,0)
            K[kRow,firstCol]   = elementFirstRow.subs(cL,0).subs(cC,0).subs(cR,1)

            b[kRow,0] = indepTerm

            kRow += 1

            # Center
            elementSecRow = interpFuncs[element]["C"].copy()
            indepTerm = - elementSecRow.subs(cL,0).subs(cC,0).subs(cR,0)

            elementSecRow += indepTerm

            K[kRow,firstCol-2] = elementSecRow.subs(cL,1).subs(cC,0).subs(cR,0)
            K[kRow,firstCol-1] = elementSecRow.subs(cL,0).subs(cC,1).subs(cR,0)
            K[kRow,firstCol]   = elementSecRow.subs(cL,0).subs(cC,0).subs(cR,1)

            b[kRow,0] = indepTerm

            kRow += 1
        
        else:
            # Left from previous and right from actual 
            prevRightInterp = interpFuncs[element-1]["R"].copy()
            prevIndepTerm   = - prevRightInterp.subs(cL,0).subs(cC,0).subs(cR,0)
            prevRightInterp += prevIndepTerm

            actLeftInterp   = interpFuncs[element]["L"].copy()
            actIndepTerm    = - actLeftInterp.subs(cL,0).subs(cC,0).subs(cR,0)
            actLeftInterp   += actIndepTerm

            jointIndepTerm = prevIndepTerm + actIndepTerm

            K[kRow,firstCol-2] = prevRightInterp.subs(cL,1).subs(cC,0).subs(cR,0)
            K[kRow,firstCol-1] = prevRightInterp.subs(cL,0).subs(cC,1).subs(cR,0)
            K[kRow,firstCol]   = prevRightInterp.subs(cL,0).subs(cC,0).subs(cR,1) + actLeftInterp.subs(cL,1).subs(cC,0).subs(cR,0)
            K[kRow,firstCol+1] = actLeftInterp.subs(cL,0).subs(cC,1).subs(cR,0)
            K[kRow,firstCol+2] = actLeftInterp.subs(cL,0).subs(cC,0).subs(cR,1)

            b[kRow,0] = jointIndepTerm

            kRow += 1

            # Center
            elementSecRow = interpFuncs[element]["C"].copy()
            indepTerm = - elementSecRow.subs(cL,0).subs(cC,0).subs(cR,0)

            elementSecRow += indepTerm

            K[kRow,firstCol]   = elementSecRow.subs(cL,1).subs(cC,0).subs(cR,0)
            K[kRow,firstCol+1] = elementSecRow.subs(cL,0).subs(cC,1).subs(cR,0)
            K[kRow,firstCol+2] = elementSecRow.subs(cL,0).subs(cC,0).subs(cR,1)

            b[kRow,0] = indepTerm

            kRow += 1


            if element == numElements:

                # Right (it is the last, no join)
                elementRow = interpFuncs[element]["R"].copy()
                indepTerm = - elementRow.subs(cL,0).subs(cC,0).subs(cR,0)

                elementRow += indepTerm
                indepTerm  += derSyms[-1]

                K[kRow,firstCol]   = elementRow.subs(cL,1).subs(cC,0).subs(cR,0)
                K[kRow,firstCol+1] = elementRow.subs(cL,0).subs(cC,1).subs(cR,0)
                K[kRow,firstCol+2] = elementRow.subs(cL,0).subs(cC,0).subs(cR,1)

                b[kRow,0] = indepTerm

                kRow += 1

            else:
                firstCol += 2
                continue


    uVec = Matrix(uSyms)

    #print("K")
    print_latex(K.subs(h, hVal))
    #print("u")
    print_latex(uVec)
    #print("b")
    print_latex(b.subs(h, hVal))

    # Now apply contour conditions
    rows, cols = 2*numElements+1, 2*numElements+1
    K, b, uVec = applyCountourConditions(contourConds, K, b, uVec, derSyms, rows, cols)

    sist = gaussSeidelSolver(K.subs(h, hVal), b.subs(h, hVal), uVec)

    numSol = []
    for key in sist[0]: numSol += [sist[0][key]]
    xList =  [i*hVal/2 for i in range(2*numElements+1)]
    anSol = [analyticalSol.subs(x, i*hVal/2) for i in range(2*numElements+1)]
    print([float(x) for x in numSol])

    return [float(x) for x in numSol], [float(x) for x in anSol[1:]], xList

if __name__=="__main__":
    # Define equation params
    leftLimit, rightLimit = 0, 1
    qFunc = 1
    fFunc = -x
    contourConds = [
        {"Type": "Dir", "Loc": "First", "Value":0},
        {"Type": "Neu", "Loc": "Last",  "Value":0}
    ]
    # Analytical sol (if has) or None
    analyticalSol = simplify((1/np.cos(1)) * sin(x) - x)

    # Params of algorithm (method and number of elements)
    params = [("Linear", 4), ("Linear", 8), ("Quad", 2), ("Quad", 4)]
    numDec = 5
    plotError = True
    for param in params:
        method, numElements = param[0], param[1]
        results = equationSolver(numElements, leftLimit, rightLimit, qFunc, fFunc, contourConds, analyticalSol, method)
        numSol, anSol, xValues = results[0], results[1], results[2]

        errors = [abs(1 - numSol[i]/anSol[i]) * 100 for i in range(len(numSol))]

        if method == "Linear": label = f"{numElements} elementos, funciones lineales"
        if method == "Quad":   label = f"{numElements} elementos, funciones cuadráticas"

        print(f"\n{label}\n")
        print("Analytical:")
        print([round(num,numDec) for num in anSol])
        print("Numerical:")
        print([round(num,numDec) for num in numSol])
        print("Error:")
        print([round(num,numDec) for num in errors])

        if plotError:
            plt.figure(1)
            plt.title("Error relativo respecto a la solución analítica (%)")
            plt.xlabel("x")
            plt.ylabel("Error relativo (%)")
            plt.yscale("log")
            plt.plot(xValues[1:], errors, label=label)
            plt.legend()
            plt.show()
