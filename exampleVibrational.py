#import matplotlib.pyplot as plt
import numpy as np

from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.drivers.second_quantization import VibrationalStructureMoleculeDriver
from qiskit_nature.problems.second_quantization import VibrationalStructureProblem
from qiskit_nature.mappers.second_quantization import DirectMapper
from qiskit_nature.converters.second_quantization import QubitConverter

from qiskit_nature.algorithms import GroundStateEigensolver

from qiskit.algorithms.optimizers import L_BFGS_B, CG
from qiskit.algorithms import VQE

from qiskit.circuit import QuantumRegister, Parameter
from qiskit.circuit.library import TwoLocal

from qiskit_nature.circuit.library import VSCF, UVCCSD

from qiskit.utils import QuantumInstance

from qiskit import IBMQ
from qiskit import Aer
from qiskit import QuantumCircuit


def vqe_function(geometry,
                 spin = 0,
                 charge = 1,
                 basis = 'sto6g',
                 var_form_type = 'TwoLocal',
                 quantum_instance = QuantumInstance(Aer.get_backend('statevector_simulator')),
                 optimizer = CG(),
                 converter = QubitConverter(mapper = DirectMapper())
                 ):

    print("inizializzo problema")
    molecule = Molecule(geometry = geometry,
                        multiplicity = 2*spin+1,
                        charge = charge)

    vibrationalDriver = VibrationalStructureMoleculeDriver(molecule)

    num_modals = [2,2,1]
    problem = VibrationalStructureProblem(vibrationalDriver,
                                          num_modals=num_modals,
                                          truncation_order=2)

    print("inizializzo ansatz")
    init_state = VSCF(num_modals = num_modals)
                      #qubit_converter = converter)

    '''
    ansatz = UVCCSD(qubit_converter = converter,
                    num_modals=num_modals,
                    reps = 1,
                    initial_state = init_state)
    '''
    ansatz = TwoLocal(num_qubits = sum(num_modals),
                      rotation_blocks = 'ry',
                      entanglement_blocks = 'cx',
                      initial_state = init_state)

    def salvaTemp(count, pars, mean, std):
        print(count, ' ', pars)

    vqe_solver = VQE(ansatz = ansatz,
                     optimizer = optimizer,
                     initial_point = np.random.rand(ansatz.num_parameters),
                     quantum_instance = quantum_instance,
                     callback = salvaTemp)

    print("creo groundStateSolver")
    calc = GroundStateEigensolver(converter, vqe_solver)
    print("solve")
    elec = calc.solve(problem)
    print(elec)


#main
dist = 1.0
alt=np.sqrt(dist**2 - (dist/2)**2)
geometry = [('H', [0., 0., 0.]),
            ('H', [dist, 0., 0.]),
            ('H', [dist/2, alt, 0.])]


vqe_function(geometry, var_form_type='TwoLocal')
