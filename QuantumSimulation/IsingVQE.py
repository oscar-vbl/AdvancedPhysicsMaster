from circuitBuilder import buildCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library.n_local.efficient_su2 import EfficientSU2
from qiskit.primitives import EstimatorResult, StatevectorSampler

if __name__ == "__main__":
    # H = - \sum Z_i Z_{i+1} - h \sum X_i
    numQubits = 3
    h_field = 0.5
    ham_list = []
    baseMatrix = ["I"] * numQubits
    for qNum in range(1,numQubits+1):
        # Interaction term
        if qNum < numQubits:
            transverseTerm = baseMatrix.copy()
            transverseTerm[qNum-1]   = "Z"
            transverseTerm[qNum+1-1] = "Z"
            ham_list += [("".join(transverseTerm), -1.0)]
        # Transverse field term
        fieldTerm = baseMatrix.copy()
        fieldTerm[qNum-1] = "X"
        ham_list += [("".join(fieldTerm), -h_field)]
        
    hamiltonian = SparsePauliOp.from_list(ham_list)

    ansatz = EfficientSU2(numQubits, reps=1)

    #sampler = StatevectorSampler()
    estimator = EstimatorResult()

    from qiskit.primitives.base import BaseEstimatorV2
    energy = estimator.expectation_value(ansatz, hamiltonian).value

    numAncilla = 0
    gates = []
    configuration = {
        "QubitsNumber": numQubits,
        "AncillaQubits": numAncilla,
        "Gates": gates
    }
    circuit = buildCircuit(configuration)