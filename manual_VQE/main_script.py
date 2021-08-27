from qiskit                                     import *
from qiskit.chemistry.drivers                   import UnitsType, HFMethodType, PySCFDriver
from qiskit.chemistry.core                      import Hamiltonian, TransformationType, QubitMappingType
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.aqua                                import QuantumInstance,aqua_globals
from qiskit.aqua                                import QuantumInstance,aqua_globals
from qiskit.aqua.operators                      import Z2Symmetries
from qiskit.aqua.algorithms.classical           import ExactEigensolver
import numpy as np
import functools

from qiskit.providers.aer.noise import NoiseModel
from qiskit.ignis.mitigation.measurement import (complete_meas_cal,tensored_meas_cal,CompleteMeasFitter,TensoredMeasFitter)


import sys
from subroutines import *


driver    = PySCFDriver(atom='''H 0.0000 0.0000 0.0000; H 0.0000 0.0000 0.742''',
                        unit=UnitsType.ANGSTROM,charge=0,spin=0,basis='sto-6g',hf_method=HFMethodType.RHF)
molecule  = driver.run()





#Now the Hamiltonian 
core      = Hamiltonian(transformation=TransformationType.FULL,qubit_mapping=QubitMappingType.PARITY,two_qubit_reduction=True,freeze_core=False,orbital_reduction=[0,1,2,5,6])
H_op,A_op = core.run(molecule)
dE        = core._energy_shift + core._ph_energy_shift + core._nuclear_repulsion_energy
A_op      = A_op[:3]
dA        = [0]*len(A_op)

## Initial state + variational ansatze to construct the circuit

init_state = HartreeFock(num_qubits=H_op.num_qubits,num_orbitals=core._molecule_info['num_orbitals'],
                    qubit_mapping=core._qubit_mapping,two_qubit_reduction=core._two_qubit_reduction,
                    num_particles=core._molecule_info['num_particles'])

num_qubits     = H_op.num_qubits
num_parameters = 6*(num_qubits-1)

def generate_circuit(parameters):
    circuit = init_state.construct_circuit()
    num_qubits = H_op.num_qubits
    circuit.barrier()
    p0 = 0
    for qubit1 in range(0,num_qubits-1,2):
        qubit2 = qubit1+1
        circuit.s(qubit1)
        circuit.s(qubit2)
        circuit.h(qubit2)
        circuit.cx(qubit2,qubit1)
        circuit.u3(parameters[p0+0],parameters[p0+1],parameters[p0+2],qubit1)
        circuit.u3(parameters[p0+3],parameters[p0+4],parameters[p0+5],qubit2); p0 += 6
        circuit.cx(qubit2,qubit1)
        circuit.h(qubit2)
        circuit.sdg(qubit1)
        circuit.sdg(qubit2)
    circuit.barrier()
    for qubit1 in range(1,num_qubits-1,2):
        qubit2 = qubit1+1
        circuit.s(qubit1)
        circuit.s(qubit2)
        circuit.h(qubit2)
        circuit.cx(qubit2,qubit1)
        circuit.u3(parameters[p0+0],parameters[p0+1],parameters[p0+2],qubit1)
        circuit.u3(parameters[p0+3],parameters[p0+4],parameters[p0+5],qubit2); p0 += 6
        circuit.cx(qubit2,qubit1)
        circuit.h(qubit2)
        circuit.sdg(qubit1)
        circuit.sdg(qubit2)
    circuit.barrier()
    return circuit

print(generate_circuit(np.zeros(num_parameters)).draw())

###Now choose the right instance


runtype    = ['qasm',None,False,8000]
#runtype    = ['noise_model','ibmq_rome',False,8000]
#runtype    = ['noise_model','ibmq_rome',True,8000]
#runtype    = ['hardware','ibmq_rome',True,8000]

if(runtype[0]=='qasm'): # ==== running VQE on QASM simulator, without noise
	print("Running on qasm simulator ")
	backend  = Aer.get_backend('qasm_simulator')
	instance = QuantumInstance(backend=backend,shots=runtype[3])
        
if(runtype[0]=='noise_model'): # ==== running VQE on QASM simulator, with noise model from an actual machine
	print("Running on qasm simulator with noise of a real hardware")
	print("Opening IBMQ account...")
	provider         = IBMQ.load_account()
	backend          = provider.get_backend(runtype[1])
	noise_model      = NoiseModel.from_backend(backend)
	coupling_map     = backend.configuration().coupling_map
	basis_gates      = noise_model.basis_gates
	simulator        = Aer.get_backend('qasm_simulator')
	if(runtype[2]):  # ==== usando readout error mitigation
		print("with error mitigation")
		instance = QuantumInstance(backend=simulator,noise_model=noise_model,coupling_map=coupling_map,basis_gates=basis_gates,
                                           measurement_error_mitigation_cls=CompleteMeasFitter,shots=runtype[3])
	else:            # ==== senza readout error mitigation
		print("without error mitigation")
		instance = QuantumInstance(backend=simulator,noise_model=noise_model,coupling_map=coupling_map,basis_gates=basis_gates,shots=runtype[3])
        

if(runtype[0]=='hardware'): # ==== running VQE on HARDWARE, I would do it with readout error mitigation
    print("Running on hardware")
    print("Running "+r)
    IBMQ.load_account()
    provider  = IBMQ.get_provider(hub='ibm-q-research',group='Stefan-Barison', project='main') 
    simulator = provider.get_backend(runtype[1])
    instance = QuantumInstance(backend=simulator,measurement_error_mitigation_cls=CompleteMeasFitter,shots=runtype[3],skip_qobj_validation=False)


algo = manual_VQE(H_op,dE,A_op,dA,generate_circuit,instance,num_parameters)
algo.run(t_mesh=np.arange(-2,2.1,0.5))
