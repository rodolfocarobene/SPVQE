import numpy as np
import sys
import itertools

from subroutineVQE import solveVQE

from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, ADAM, SLSQP
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit.utils import QuantumInstance
from qiskit import Aer

if __name__ == '__main__':
   
    dist = 2.5
    alt = np.sqrt(dist**2 - (dist/2)**2)
    options = {
        'molecule': {
            'geometry': "H 0. 0. 0.; H "+str(dist)+" 0. 0.; H "+str(dist/2)+" "+str(alt)+"  0.",
            'basis': "sto-6g",
            'spin' : 0,
            'charge': 1
        },
        'var_form_type': "TwoLocal",
        'quantum_instance': QuantumInstance(Aer.get_backend('statevector_simulator')),
        'optimizer': COBYLA(),
        'converter': QubitConverter(mapper = ParityMapper(), two_qubit_reduction=True),
        'lagrange': {
            'active': True,
            'operators': ('number', 2, 0)
        }

    }

    result_tot = solveVQE(options)



