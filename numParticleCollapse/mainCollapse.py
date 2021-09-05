import numpy as np

from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import ParityMapper

from qiskit.opflow import ExpectationFactory, StateFn, CircuitStateFn, CircuitSampler, ListOp

from qiskit import Aer
from qiskit.utils import QuantumInstance

from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit_nature.drivers import UnitsType
from qiskit_nature.drivers.second_quantization.electronic_structure_driver import MethodType

from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem

from qiskit_nature.circuit.library import HartreeFock

from qiskit.opflow.primitive_ops import PauliSumOp

from qiskit.circuit.library import TwoLocal

def convert_to_paulisumop(operator):
    if isinstance(operator, PauliSumOp):
        return operator
    try:
        primitive = SparsePauliOp(operator.primitive)
        return PauliSumOp(primitive, operator.coeff)
    except Exception as exc:
        raise ValueError(
                f"Invalid type of the operator {type(operator)} "
                "must be PauliSumOp, or castable to one."
        ) from exc

def eval_aux_ops(ansatz, parameters, aux_operators, expectation, quantum_instance, threshold = 1e-12):

    sampler = CircuitSampler(quantum_instance)

    aux_op_meas = expectation.convert(StateFn(ListOp(aux_operators), is_measurement=True))
    aux_op_expect = aux_op_meas.compose(CircuitStateFn(ansatz.bind_parameters(parameters)))
    values = np.real(sampler.convert(aux_op_expect).eval())

    aux_op_results = values * (np.abs(values) > threshold)
    _aux_op_nones = [op is None for op in aux_operators]
    aux_operator_eigenvalues = [
            None if is_none else [result]
            for (is_none, result) in zip(_aux_op_nones, aux_op_results)
    ]
    aux_operator_eigenvalues = np.array([aux_operator_eigenvalues], dtype=object)
    return aux_operator_eigenvalues

def myFunc(fixValue, threshold):
    dist = 1.0
    alt = np.sqrt(dist**2 - (dist/2)**2)
    geometry = "H 0. 0. 0.; H 0. " + str(dist) + " 0.; H " + str(alt) + " " + str(dist/2) + " 0."
    driver = PySCFDriver(atom = geometry,
                         unit = UnitsType.ANGSTROM,
                         basis = 'sto-6g',
                         charge = 1,
                         method = MethodType.RHF)
    problem = ElectronicStructureProblem(driver)

    opHamiltonian = problem.second_q_ops()[0]
    opNum_particles = problem.second_q_ops()[1]

    particleProperty = driver.run().get_property('ParticleNumber')
    num_particles = (particleProperty.num_alpha, particleProperty.num_beta)
    converter = QubitConverter(mapper = ParityMapper(), two_qubit_reduction = True)

    qHamiltonian = converter.convert(opHamiltonian,
                                     num_particles = num_particles)
    qNum_particles = converter.convert(opNum_particles,
                                       num_particles = num_particles)

    #probabilmente mi sere convertire l'op in un PauliSumOp... dunque:
    pauliNum_particles = convert_to_paulisumop(qNum_particles) #forse qNum??

    init_state = HartreeFock(particleProperty.num_spin_orbitals,
                            num_particles,
                            converter)
    ansatz = TwoLocal(num_qubits = 4,
                      rotation_blocks = 'ry',
                      entanglement_blocks = 'cx',
           #           initial_state = init_state,
                      entanglement = 'linear')


    aux_operators = [pauliNum_particles]
    quantum_instance = QuantumInstance(Aer.get_backend('statevector_simulator'))
    expectation = ExpectationFactory.build(
                        operator=qHamiltonian,
                        backend=quantum_instance,
                        include_custom=False,
                    )

    calcValue = 100
    i = 0
    while abs(calcValue - fixValue) > threshold:
        i = i + 1
        parameters = np.random.rand(ansatz.num_parameters)
        #parameters = np.zeros(ansatz.num_parameters)
        eigenvalues = eval_aux_ops(ansatz, parameters, aux_operators, expectation, quantum_instance, threshold = 1e-12)
        calcValue = eigenvalues[0][0][0]
        print(calcValue)

        if i % 10 == 0:
            print('iterazione ', i)

    print('ITERAZIONE ', i)
    print(parameters)
    print(calcValue)


if __name__ == '__main__':
    myFunc(2, 0.1)
