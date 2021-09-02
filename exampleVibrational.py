#import matplotlib.pyplot as plt
import numpy as np

from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.drivers.second_quantization import VibrationalStructureMoleculeDriver, PySCFDriver
from qiskit_nature.drivers.second_quantization.electronic_structure_driver import MethodType
from qiskit_nature.problems.second_quantization import VibrationalStructureProblem
from qiskit_nature.mappers.second_quantization import DirectMapper
from qiskit_nature.converters.second_quantization import QubitConverter

from qiskit_nature.algorithms import VQEUCCFactory, GroundStateEigensolver

from qiskit.algorithms.optimizers import L_BFGS_B, CG
from qiskit.algorithms import VQE

from qiskit.circuit import QuantumRegister, Parameter
from qiskit.circuit.library import TwoLocal

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
    #setting up molecule driver

    electronicDriver = PySCFDriver(atom = geometry,
                         unit = UnitsType.ANGSTROM,
                         basis = basis,
                         spin = spin,
                         charge = charge,
                         method = MethodType.RHF)

    molecule = electronicDriver.run()
    vibrationalDriver = VibrationalStructureMoleculeDriver(molecule)

    problem = VibrationalStructureProblem(vibrationalDriver,num_modals=2, truncation_order=2)

    ansatz = TwoLocal(num_qubits = 7,
                      rotation_blocks = 'ry',
                      entanglement_blocks = 'cx',
                      entanglement = 'linear')

    vqe_solver = VQE(ansatz = ansatz,
                     optimizer = optimizer,
                     quantum_instance = quantum_instance)

    calc = GroundStateEigensolver(converter, vqe_solver)
    elec = calc.solve(problem)
    print(elec)
'''
    print("RISULTATO: ",
          "\n\tEnergia calcolata: ", elec.computed_energies[0],
          "\n\tEnergia ioni: ", elec.nuclear_repulsion_energy,
          "\n\tTotale: ", elec.total_energies[0],
          "\n\tNum particelle: ", elec.num_particles,
          "\n\tSpin: ", elec.spin)
'''
#main

dist = 1.0
alt=np.sqrt(dist**2 - (dist/2)**2)
geometry = "H .0 .0 .0; H .0 .0 " + str(dist) + "; H .0 " + str(alt) + " " + str(dist/2) 


vqe_function(geometry, var_form_type='TwoLocal')
