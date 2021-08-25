import numpy as np

from qiskit.algorithms.optimizers import CG
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit.utils import QuantumInstance
from qiskit import Aer

from subroutineVQE import solveHamiltonianVQE, solveLagrangianVQE, writeJson

if __name__ == '__main__':
    options = {
        'molecule' : {
            'spin' : 0,
            'charge' : 1,
            'basis' : 'sto-6g',
        },
        'var_form_type' : 'TwoLocal',
        'quantum_instance' : QuantumInstance(Aer.get_backend('qasm_simulator'),
                                             shots = 1000),
        'optimizer' : CG(),
        'converter' : QubitConverter(mapper = ParityMapper(),
                                   two_qubit_reduction = True),
        'lagrange' : {
            'operator' : 'number',
            'value' : 2,
            'multiplier': 0.2
        }
    }

    dist = np.arange(.3, 3., .1)
    alt = np.sqrt(dist**2 - (dist/2)**2)

    varforms = ['TwoLocal', 'UCCSD', 'SO(4)']
    energies = {}
    for var in varforms:
        energies[var] = []

    for i in range(len(dist)):
        geometry = "H .0 .0 .0; H .0 .0 " + str(dist[i]) + "; H .0 " + str(alt[i]) + " " + str(dist[i]/2)
        options['molecule']['geometry'] = geometry

        for form in varforms:
            options['var_form_type'] = form
            result_tot = solveHamiltonianVQE(options)
            result = result_tot.total_energies[0]
            energies[form].append(result)
            print("D = ", np.round(dist[i], 2),
                  "\tVar = ", form,
                  "\tE = ", result)


    JsonOptions = {
        'file': __file__,
        'algo': {
            'name': 'vqe',
            'type': 'lagrangianvqe'
        },
        'backend': {
            'type': 'statevector_simulator',
            'noisy': 'None'
        },
        'dist': dist.tolist(),
        'energies' : energies
    }

    writeJson(JsonOptions)

