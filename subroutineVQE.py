import numpy as np
import logging

from datetime import datetime

from qiskit import IBMQ
from qiskit import Aer
from qiskit import QuantumCircuit

from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit_nature.drivers.second_quantization.electronic_structure_driver import MethodType
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.mappers.second_quantization import ParityMapper, BravyiKitaevMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.circuit.library import HartreeFock, UCCSD
from qiskit_nature.algorithms import VQEUCCFactory, GroundStateEigensolver
from qiskit_nature.results import EigenstateResult

from qiskit.algorithms.optimizers import L_BFGS_B, CG
from qiskit.algorithms import VQE
from qiskit.circuit import QuantumRegister, Parameter
from qiskit.circuit.library import TwoLocal, EfficientSU2
from qiskit.utils import QuantumInstance
from qiskit.opflow.primitive_ops import PauliOp
from qiskit.quantum_info import Pauli

def getDateTimeString():
    now = datetime.now()
    return now.strftime("%d_%m_%H_%M")

myLogger = logging.getLogger('myLogger')
myLogger.setLevel(logging.DEBUG)
ch = logging.FileHandler('./logs/' + getDateTimeString() + '.log')
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
myLogger.addHandler(ch)

def addSingleSO4Gate(circuit,
                     qubit1,
                     qubit2,
                     params,
                     p0):
    myLogger.info('Inizio di addSingleSO4Gate')

    circuit.s(qubit1)
    circuit.s(qubit2)
    circuit.h(qubit2)
    circuit.cx(qubit2,qubit1)
    circuit.u(params[p0], params[p0+1], params[p0+2], qubit1)
    p0 += 3
    circuit.u(params[p0], params[p0+1], params[p0+2], qubit2)
    p0 += 3
    circuit.cx(qubit2,qubit1)
    circuit.h(qubit2)
    circuit.sdg(qubit1)
    circuit.sdg(qubit2)

    myLogger.info('Fine di addSingleSO4Gate')

#definition of SO(4) variational form for arbitrary nuber of qubits

def constructSO4Ansatz(numqubits, init = None):
    myLogger.info('Inizio di constructSO4Ansatz')

    num_parameters = 6 * (numqubits - 1)
    parameters = []
    for i in range(num_parameters):
        name = "par" + str(i)
        new = Parameter(name)
        parameters.append(new)
    q = QuantumRegister(numqubits, name = 'q')
    if init != None:
        circ = init
    else:
        circ = QuantumCircuit(q)
    i = 0
    n = 0
    while i + 1 < numqubits:
        addSingleSO4Gate(circ , i, i+1, parameters , 6*n)
        n = n + 1
        i = i + 2
    i = 1
    while i + 1 < numqubits:
        addSingleSO4Gate(circ , i, i+1, parameters , 6*n)
        n = n +1
        i = i + 2

    myLogger.info('Fine di constructSO4Ansatz')

    return circ

def createLagrangeOperatorPS(hamiltonian,
                             auxiliary,
                             multiplier,
                             operator,
                             value):
    myLogger.info('Inizio di createLagrangeOperatorPS')

    # TODO: this switch sucks
    if operator == "number":
        idx = 0
    elif operator == "spin-squared":
        idx = 1
    elif operator == "spin-z":
        idx = 2

    x = np.zeros(hamiltonian.num_qubits)
    z = np.zeros(hamiltonian.num_qubits)

    equality = auxiliary[idx].add(PauliOp(Pauli((z,x)), -value))

    penaltySquared = (equality ** 2).mul(multiplier)

    lagrangian  =  hamiltonian.add(penaltySquared)

    myLogger.info('Fine di createLagrangeOperatorPS')

    return lagrangian

def createLagrangeOperatorAUG(hamiltonian,
                             auxiliary,
                             multiplierSimple,
                             multiplierSquare,
                             operator,
                             value):
    myLogger.info('Inizio di createLagrangeOperatorAUG')

    # TODO: this switch sucks
    if operator == "number":
        idx = 0
    elif operator == "spin-squared":
        idx = 1
    elif operator == "spin-z":
        idx = 2

    x = np.zeros(hamiltonian.num_qubits)
    z = np.zeros(hamiltonian.num_qubits)

    equality = auxiliary[idx].add(PauliOp(Pauli((z,x)), -value))

    penaltySquared = (equality ** 2).mul(multiplierSquare)
    penaltySimple = equality.mul(-multiplierSimple)

    lagrangian  =  hamiltonian.add(penaltySquared).add(penaltySimple)

    myLogger.info('Fine di createLagrangeOperatorAUG')

    return lagrangian

def prepareBaseVQE(options):
    myLogger.info('Inizio di prepareBaseVQE')

    spin = options['molecule']['spin']
    charge = options['molecule']['charge']
    basis = options['molecule']['basis']
    geometry = options['molecule']['geometry']

    var_form_type = options['var_form_type']
    quantum_instance = options['quantum_instance']
    optimizer = options['optimizer']
    converter = options['converter']
    init_point = options['init_point']


    #setting up molecule driver

    driver = PySCFDriver(atom = geometry,
                         unit = UnitsType.ANGSTROM,
                         basis = basis,
                         spin = spin,
                         charge = charge,
                         method = MethodType.RHF)

    problem = ElectronicStructureProblem(driver)
    main_op = problem.second_q_ops()[0]

    driverResult = driver.run()
    particleNumber = driverResult.get_property('ParticleNumber')

    alpha, beta = particleNumber.num_alpha, particleNumber.num_beta
    num_particles = (alpha, beta)
    num_spin_orbitals = particleNumber.num_spin_orbitals

    qubit_op = converter.convert(main_op, num_particles = num_particles)

    init_state = HartreeFock(num_spin_orbitals,
                             num_particles,
                             converter)

    num_qubits = qubit_op.num_qubits

    vqe_solver = createVQEFromAnsatzType(var_form_type,
                                         num_qubits,
                                         init_state,
                                         quantum_instance,
                                         optimizer,
                                         converter,
                                         num_particles,
                                         num_spin_orbitals,
                                         init_point)

    myLogger.info('Fine di prepareBaseVQE')

    return converter, vqe_solver, problem, qubit_op

def store_intermediate_result(count, par, energy, std):
    global parameters
    parameters.append(par)

def createVQEFromAnsatzType(var_form_type,
                            num_qubits,
                            init_state,
                            quantum_instance,
                            optimizer,
                            converter,
                            num_particles,
                            num_spin_orbitals,
                            initial_point):
    myLogger.info('Inizio createVQEFromAnsatzType')

    if var_form_type == 'TwoLocal':
        ansatz = TwoLocal(num_qubits = num_qubits,
                          rotation_blocks = 'ry',
                          entanglement_blocks = 'cx',
                          initial_state = init_state,
                          entanglement = 'linear')
        if None in initial_point:
            initial_point = np.random.rand(ansatz.num_parameters)
        vqe_solver = VQE(ansatz = ansatz,
                         optimizer = optimizer,
                         initial_point = initial_point,
                         callback = store_intermediate_result,
                         quantum_instance = quantum_instance)
    elif var_form_type == 'EfficientSU(2)':
        ansatz = EfficientSU2(num_qubits = num_qubits,
                              entanglement = 'linear',
                              initial_state = init_state)
        if None in initial_point:
            initial_point = np.random.rand(ansatz.num_parameters)
        vqe_solver = VQE(ansatz = ansatz,
                         optimizer = optimizer,
                         initial_point = initial_point,
                         callback = store_intermediate_result,
                         quantum_instance = quantum_instance)

    elif var_form_type == 'UCCSD':
        ansatz = UCCSD(qubit_converter = converter,
                       initial_state = init_state,
                       num_particles = num_particles,
                       num_spin_orbitals = num_spin_orbitals)._build()

        if None in initial_point:
            initial_point = np.random.rand(ansatz.num_parameters)
        vqe_solver = VQE(quantum_instance = quantum_instance,
                         ansatz = ansatz,
                         optimizer = optimizer,
                         callback = store_intermediate_result,
                         initial_point = initial_point)

    elif var_form_type == 'SO(4)':
        ansatz = constructSO4Ansatz(num_qubits,
                                    init = init_state)
        if None in initial_point:
            initial_point = np.random.rand(6*(num_qubits-1))
        vqe_solver = VQE(ansatz = ansatz,
                         optimizer = optimizer,
                         initial_point = initial_point,
                         callback = store_intermediate_result,
                         quantum_instance = quantum_instance)

    else:
        raise Exception("VAR_FORM_TYPE NOT EXISTS")

    myLogger.info('Fine  createVQEFromAnsatzType')

    return vqe_solver

def solveHamiltonianVQE(options):
    myLogger.info('Inizio solveHamiltonianVQE')
    myLogger.info('OPTIONS')
    myLogger.info(options)

    converter, vqe_solver, problem, qubit_op = prepareBaseVQE(options)

    calc = GroundStateEigensolver(converter, vqe_solver)
    result = calc.solve(problem)

    myLogger.info('Fine solveHamiltonianVQE')
    myLogger.info('RESULT')
    myLogger.info(result)

    return result

def solveLagrangianVQE(options):
    myLogger.info('Inizio solveLagrangianVQE')
    myLogger.info('OPTIONS')
    myLogger.info(options)

    converter, vqe_solver, problem, qubit_op = prepareBaseVQE(options)

    if options['var_form_type'] == 'UCCSD':
        vqe_solver = vqe_solver.get_solver(problem, converter)

    auxOpsNotConverted = problem.second_q_ops()[1:4]
    aux_ops = convertListFermOpToQubitOp(auxOpsNotConverted,
                                         converter,
                                         problem.num_particles)

    lagrange_op = qubit_op
    for operatore in options['lagrange']['operators']:
        operator = operatore[0]
        value = operatore[1]
        multiplier = operatore[2]
        lagrange_op = createLagrangeOperatorPS(lagrange_op,
                                               aux_ops,
                                               multiplier = multiplier,
                                               operator = operator,
                                               value = value)

    oldResult = vqe_solver.compute_minimum_eigenvalue(operator=lagrange_op,
                                                 aux_operators=aux_ops)

    newResult = problem.interpret(oldResult)

    myLogger.info('Fine solveLagrangianVQE')
    myLogger.info('RESULT')
    myLogger.info(newResult)

    return newResult

def solveAUGLagrangianVQE(options, lamb):
    myLogger.info('Inizio solveAUGLagrangianVQE')
    myLogger.info('OPTIONS')
    myLogger.info(options)

    converter, vqe_solver, problem, qubit_op = prepareBaseVQE(options)

    if options['var_form_type'] == 'UCCSD':
        vqe_solver = vqe_solver.get_solver(problem, converter)

    auxOpsNotConverted = problem.second_q_ops()[1:4]
    aux_ops = convertListFermOpToQubitOp(auxOpsNotConverted,
                                         converter,
                                         problem.num_particles)

    lagrange_op = qubit_op
    for operatore in options['lagrange']['operators']:
        operator = operatore[0]
        value = operatore[1]
        multiplier = operatore[2]
        lagrange_op = createLagrangeOperatorAUG(lagrange_op,
                                                aux_ops,
                                                multiplierSquare = multiplier,
                                                multiplierSimple = lamb,
                                                operator = operator,
                                                value = value)

    oldResult = vqe_solver.compute_minimum_eigenvalue(operator=lagrange_op,
                                                      aux_operators=aux_ops)

    newResult = problem.interpret(oldResult)

    myLogger.info('Fine solveAUGLagrangianVQE')
    myLogger.info('RESULT')
    myLogger.info(newResult)

    return newResult

def convertListFermOpToQubitOp(old_aux_ops, converter, num_particles):
    myLogger.info('Inizio convertListFermOpToQubitOp')

    new_aux_ops = []
    for op in old_aux_ops:
        opNew = converter.convert(op, num_particles)
        new_aux_ops.append(opNew)

    myLogger.info('Fine convertListFermOpToQubitOp')

    return new_aux_ops

def solveLagSeriesVQE(options):
    iter_max = options['series']['itermax']
    par = np.zeros(16)
    mult = 0.01
    step = options['series']['step']

    global parameters
    parameters = [par]
    for i in range(iter_max):
        tmp_mult = mult + step * i

        lagOpList = []

        for singleOp in options['lagrange']['operators']:
            operatore = (singleOp[0],
                         singleOp[1],
                         float(tmp_mult))
            lagOpList.append(operatore)

        options['lagrange']['operators'] = lagOpList

        options['init_point'] = par
        result = solveLagrangianVQE(options)
        par = parameters[len(parameters) - 1]
        penalty = tmp_mult*abs(result.num_particles[0] - operatore[1])
        log_str = "Iter " + str(i)
        log_str += " mult " + str(np.round(tmp_mult,2))
        log_str += "\tE = " + str(np.round(result.total_energies[0],7))
        log_str += "\tE-P = " + str(np.round(result.total_energies[0] - penalty,7))

        myLogger.info(log_str)

    return result

def solveLagAUGSeriesVQE(options):
    iter_max = options['series']['itermax']
    par = np.zeros(16)
    mult = 0.01
    step = options['series']['step']

    lamb = options['series']['lamb']

    global parameters
    parameters = [par]
    for i in range(iter_max):
        tmp_mult = mult + step * i

        lagOpList = []

        for singleOp in options['lagrange']['operators']:
            operatore = (singleOp[0],
                         singleOp[1],
                         float(tmp_mult))
            lagOpList.append(operatore)

        options['lagrange']['operators'] = lagOpList

        options['init_point'] = par

        result = solveAUGLagrangianVQE(options, lamb)

        penalty = tmp_mult*((result.num_particles[0] - operatore[1])**2) - lamb * (result.num_particles[0] - operatore[1])
        log_str = "Iter " + str(i)
        log_str += " mult " + str(np.round(tmp_mult,2))
        log_str += " lamb " + str(lamb)
        log_str += "\tE = " + str(np.round(result.total_energies[0],7))
        log_str += "\tE-P = " + str(np.round(result.total_energies[0] - penalty,7))
        myLogger.info(log_str)

        par = parameters[len(parameters) - 1]
        #print(lamb)
        lamb = lamb - tmp_mult*2*(result.num_particles[0] - operatore[1])

        #print(result.total_energies[0] - penalty, " ", penalty)

    return result

def solveVQE(options):
    global parameters
    parameters = []

    if not options['lagrange']['active']:
        return solveHamiltonianVQE(options)

    if not options['lagrange']['series']:
        return solveLagrangianVQE(options)
    else:
        if options['lagrange']['augmented']:
            return solveLagAUGSeriesVQE(options)
        else:
            return solveLagSeriesVQE(options)

