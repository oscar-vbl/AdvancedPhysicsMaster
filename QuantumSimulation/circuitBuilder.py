import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit, Aer, transpile, assemble, execute, AncillaRegister

def buildCircuit(configuration):
    '''
    Build quantum circuit given a configuration dictionary.

    Params:
        *configuration*: dict

    Entries of configuration:
        - "QubitsNumber": int, number of qubits in the circuit
        - "AncillaQubits": int, number of ancilla qubits (optional)
        - "Gates": list of dicts, each dict specifies a gate and its parameters

    Params of each gate dict:
        - "gate": str, type of gate ("CNOT", "RZ", etc.)
        - Entries for single-qubit gates:
            - "qubit": qubit-dict
        - Entries for double qubit gates:
            - "control": qubit-dict for control qubit
            - "target": qubit-dict for target qubit

    Params of each qubit-dict:
        - "Number": number of the qubit
        - "Ancilla": if it's an ancillary qubit (optional)
        - Specific parameters depending on the gate type
    '''
    numQubits = configuration["QubitsNumber"]
    ancillaQubits = configuration.get("AncillaQubits", 0)

    circuit = QuantumCircuit(numQubits)
    if ancillaQubits > 0:
        ancilla = AncillaRegister(ancillaQubits, 'ancilla')
        circuit.add_register(ancilla)
    
    for gate in configuration["Gates"]:
        gateType = gate["gate"]
        if gateType == "CNOT":
            # Control qubit of CNOT gate
            controlQubitNum = gate["control"]["Number"]
            isAncilla       = gate["control"].get("Ancilla", False)
            if isAncilla: controlQubit = ancilla[controlQubitNum]
            else:         controlQubit = controlQubitNum
            # Target qubit of CNOT gate
            targetQubitNum = gate["target"]["Number"]
            isAncilla      = gate["target"].get("Ancilla", False)
            if isAncilla: targetQubit = ancilla[targetQubitNum]
            else:         targetQubit = targetQubitNum
            # Add CNOT gate
            circuit.cnot(controlQubit, targetQubit)

        elif gateType == "RZ":
            # RZ (phase shift around z-axis) gate
            qubitNum  = gate["qubit"]["Number"]
            isAncilla = gate["qubit"].get("Ancilla", False)
            if isAncilla: qubit = ancilla[qubitNum]
            else:         qubit = qubitNum
            circuit.rz(gate["angle"], qubit)

        elif gateType == "U":
            # U gate, defined by 3 angles: alpha, beta, gamma
            qubitNum  = gate["qubit"]["Number"]
            isAncilla = gate["qubit"].get("Ancilla", False)
            if isAncilla: qubit = ancilla[qubitNum]
            else:         qubit = qubitNum
            circuit.u(*gate["angles"], qubit)

    return circuit

if __name__ == "__main__":
    numQubits = 3
    numAncilla = 1
    dt = 0.1
    gates = []
    for qubit in range(numQubits):
        gates += [
            {'gate': 'CNOT', 'control': {"Number": qubit}, 'target': {"Number": 0, "Ancilla":True}},
        ]
    gates += [{'gate': 'RZ', 'qubit': {"Number": 0, "Ancilla":True}, 'angle': 2*dt}]
    for qubit in reversed(range(numQubits)):
        gates += [
            {'gate': 'CNOT', 'control': {"Number": qubit}, 'target': {"Number": 0, "Ancilla":True}},
        ]
    configuration = {
        "QubitsNumber": numQubits,
        "AncillaQubits": numAncilla,
        "Gates": gates
    }
    circuit = buildCircuit(configuration)