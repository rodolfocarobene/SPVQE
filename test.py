import numpy as np

from qiskit.algorithms.optimizers import CG
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit.utils import QuantumInstance
from qiskit import Aer

from subroutineVQE import SolveHamiltonianVQE, SolveLagrangianVQE, WriteJson

if __name__ == '__main__':
    options = {
        'molecule' : {
            'spin' : 0,
            'charge' : 1,
            'basis' : 'sto-6g',
            'geometry' : 'H'
        },
        'var_form_type' : 'TwoLocal',
        'quantum_instance' : QuantumInstance(Aer.get_backend('statevector_simulator')),
        'optimizer' : CG(),
        'converter' : QubitConverter(mapper = ParityMapper(),
                                   two_qubit_reduction = True),
        'lagrange' : {
            'operator' : 'spin-squared',
            'value' : 0,
            'multiplier': 0.2
        }
    }

    dist = np.arange(.2, 1.5, .1)
    alt = np.sqrt(dist**2 - (dist/2)**2)

    energies = {}
    energies['TwoLocal'] = []

    for i in range(len(dist)):
        geometry = "H .0 .0 .0; H .0 .0 " + str(dist[i]) + "; H .0 " + str(alt[i]) + " " + str(dist[i]/2)

        options['molecule']['geometry'] = geometry
        result_tot = SolveHamiltonianVQE(options)
        result = result_tot.total_energies[0]
        energies['TwoLocal'].append(result)
        print("D = ", np.round(dist[i], 2), "\tE = ", result)

    JsonOptions = {
        'file': __file__,
        'algo': {
            'name': 'vqe',
            'type': 'hamiltonianvqe'
        },
        'backend': {
            'type': 'statevector_simultor',
            'noisy': 'None'
        },
        'dist': dist,
        'energies' : energies
    }

    WriteJson(JsonOptions, 'dati_misura.json')

    '''
    print("RISULTATO: ",
              "\n\tEnergia calcolata: ", elec.computed_energies[0],
              "\n\tEnergia ioni: ", elec.nuclear_repulsion_energy,
              "\n\tTotale: ", elec.total_energies[0],
              "\n\tNum particelle: ", elec.num_particles,
              "\n\tSpin: ", elec.spin)
    '''
 
