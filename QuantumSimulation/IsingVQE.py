# Basic VQE solver for the Transverse Ising model using Qiskit

from circuitBuilder import buildCircuit, addGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.fake_provider import GenericBackendV2

import numpy as np

def sigma_expectation(counts, qubit_indexes):
    shots = sum(counts.values())
    expected = 0
    
    for qubit_label, count in counts.items():
        # Doing these because Qiskit uses little-endian bitstrings
        # Returns reversed, so, to select from a string like '001'
        # the first qubit (index 0, value '1'), we need to do this
        rev_qubit_label = qubit_label[::-1]  # reverse bitstring
        ind_label_expectation = count / shots
        for qubit_index in qubit_indexes:
            qubit_state = rev_qubit_label[qubit_index]   # reverse because Qiskit uses little-endian
            if qubit_state == '0': value = +1
            else:                  value = -1
            ind_label_expectation *= value
        expected += ind_label_expectation
    
    return expected

def getVQEAnsatzConfig(numQubits, theta_angles):
    configuration = {
        "QubitsNumber": numQubits,
        "AncillaQubits": 0,
        "MeasurementQubits": numQubits,
        "Gates": []
    }

    # First: Pseudo-Hadamard gate on first qubit
    configuration["Gates"] += [{'gate': 'RZ', 'qubit': {"Number": 0}, 'angle': theta_angles[0]}]
    configuration["Gates"] += [{'gate': 'RY', 'qubit': {"Number": 0}, 'angle': theta_angles[1]}]

    # Second: Entangling CNOTs
    for qubit in range(1, numQubits):
        configuration["Gates"] += [
            {'gate': 'CNOT', 'control': {"Number": 0}, 'target': {"Number": qubit}},
        ]
    
    # Third: RY rotations on each qubit
    for qubit in range(numQubits):
        configuration["Gates"] += [
            {'gate': 'RY', 'qubit': {"Number": qubit}, 'angle': theta_angles[2]},
        ]

    
    
    return configuration

class VQESolver():
    def __init__(self, numQubits, hamiltonian):
        self.numQubits = numQubits
        self.hamiltonian = hamiltonian
        self.backend = GenericBackendV2(num_qubits=self.numQubits)


    def getEnergy(self, theta_angles):
        configuration = getVQEAnsatzConfig(self.numQubits, theta_angles)
        ansatz_circuit = buildCircuit(configuration)
        energy = 0.0
        for pauli_string, qubits, coeff in self.hamiltonian.to_sparse_list():
            ind_circuit = ansatz_circuit.copy()
            for qubit in qubits:
                # Measure the apportation to the energy for this Pauli string
                pauli = pauli_string[qubits.index(qubit)]
                if pauli == 'X':
                    hadamard_gate = {
                        "gate": "H",
                        "qubit": {"Number": qubit}
                    }
                    addGate(ind_circuit, hadamard_gate)
                elif pauli == 'Y':
                    ry_gate = {
                        "gate": "RY",
                        "qubit": {"Number": qubit},
                        "angle": -np.pi/2
                    }
                    hadamard_gate = {
                        "gate": "H",
                        "qubit": {"Number": qubit}
                    }
                    addGate(ind_circuit, ry_gate)
                    addGate(ind_circuit, hadamard_gate)
                
                if qubit == qubits[0]: ind_circuit.barrier()
                # For 'Z' and 'I', no basis change is needed
                measure_gate = {
                    "gate": "Measure",
                    "qubit": {"Number": qubit},
                    "classicalBit": qubit
                }
                addGate(ind_circuit, measure_gate)
            job = self.backend.run(ind_circuit)
            counts = job.result().get_counts()
            
            energy += sigma_expectation(counts, qubits) * coeff

        return energy

    
    def minimizeEnergy(self, initial_angles):
        from scipy.optimize import minimize

        def objective(theta):
            return self.getEnergy(theta)

        result = minimize(objective,
                          initial_angles,
                          method='COBYLA')
        return result

def build_simple_Ising_hamiltonian(numQubits, h_field):
    '''
    Builds the transverse Ising Hamiltonian:
    H = - \sum Z_i Z_{i+1} - h \sum X_i
    for a chain of numQubits qubits with transverse field h_field.
    Returns a SparsePauliOp representing the Hamiltonian.
    '''
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
    return hamiltonian

def VQE_Energy_Field(numQubits, h_values, plot=True):
    energy_values = []
    for h_field in h_values:
        hamiltonian = build_simple_Ising_hamiltonian(numQubits, h_field)

        solver = VQESolver(numQubits, hamiltonian)
        result = solver.minimizeEnergy([0.1, 0.2, 0.3])
        print("Optimal angles:", result.x)
        print("Minimum energy:", result.fun)
        energy_values += [result.fun]
    
    if plot:
        import matplotlib.pyplot as plt 
        plt.plot(h_values, energy_values, marker='o')
        plt.xlabel("Transverse field h")
        plt.ylabel("Ground state energy (VQE)")
        plt.title(f"VQE Ground State Energy for Transverse Ising Model ({numQubits} qubits)")
        plt.grid()  
        plt.show()
    
    return energy_values

def VQE_Energy_ComputationTime(numQubits, h_field):
    import time
    hamiltonian = build_simple_Ising_hamiltonian(numQubits, h_field)

    solver = VQESolver(numQubits, hamiltonian)
    start_time = time.time()
    result = solver.minimizeEnergy([0.1, 0.2, 0.3])
    end_time = time.time()
    computation_time = end_time - start_time
    print("Optimal angles:", result.x)
    print("Minimum energy:", result.fun)
    print(f"Computation time for {numQubits} qubits: {computation_time:.4f} seconds")
    return result.fun, computation_time

if __name__ == "__main__":
    # H = - \sum Z_i Z_{i+1} - h \sum X_i
    numQubits = 3
    h_field = 0.5
    h_values = np.linspace(0.0, 2.0, 30)

    if False:
        energy_values = VQE_Energy_Field(numQubits, h_values, plot=True)
    
    if True:
        numQubitsValues = [4, 6, 8, 10, 15, 20]
        compTimeValues = []
        for numQubits in numQubitsValues:
            energy, comp_time = VQE_Energy_ComputationTime(numQubits, h_field)
            compTimeValues += [comp_time]
        import matplotlib.pyplot as plt
        plt.plot(numQubitsValues, compTimeValues, marker='o')
        plt.xlabel("Number of Qubits")
        plt.ylabel("Computation Time (s)")

