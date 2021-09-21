import logging
from datetime import datetime

import numpy as np

from qiskit import IBMQ
from qiskit import Aer
from qiskit import QuantumCircuit

from qiskit.algorithms.optimizers import L_BFGS_B, CG
from qiskit.algorithms import VQE
from qiskit.circuit import QuantumRegister, Parameter
from qiskit.circuit.library import TwoLocal, EfficientSU2
from qiskit.utils import QuantumInstance
from qiskit.opflow.primitive_ops import PauliOp
from qiskit.quantum_info import Pauli

from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit_nature.drivers.second_quantization.electronic_structure_driver import MethodType
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.mappers.second_quantization import ParityMapper, BravyiKitaevMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.circuit.library import HartreeFock, UCCSD
from qiskit_nature.algorithms import VQEUCCFactory, GroundStateEigensolver
from qiskit_nature.results import EigenstateResult

def get_date_time_string():
    now = datetime.now()
    return now.strftime("%d_%m_%H_%M")

myLogger = logging.getLogger('myLogger')
myLogger.setLevel(logging.DEBUG)
ch = logging.FileHandler('./logs/' + get_date_time_string() + '.log')
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
myLogger.addHandler(ch)

def add_single_SO4_gate(circuit,
                     qubit1,
                     qubit2,
                     params,
                     p0):
    myLogger.info('Inizio di add_single_SO4_gate')

    circuit.s(qubit1)
    circuit.s(qubit2)
    circuit.h(qubit2)
    circuit.cx(qubit2, qubit1)
    circuit.u(params[p0], params[p0+1], params[p0+2], qubit1)
    p0 += 3
    circuit.u(params[p0], params[p0+1], params[p0+2], qubit2)
    p0 += 3
    circuit.cx(qubit2, qubit1)
    circuit.h(qubit2)
    circuit.sdg(qubit1)
    circuit.sdg(qubit2)

    myLogger.info('Fine di add_single_SO4_gate')

def construct_SO4_ansatz(numqubits, init=None):
    myLogger.info('Inizio di construct_SO4_ansatz')

    num_parameters = 6 * (numqubits - 1)
    so4_parameters = []
    for i in range(num_parameters):
        name = "par" + str(i)
        new = Parameter(name)
        so4_parameters.append(new)
    quantum_reg = QuantumRegister(numqubits, name='q')
    if init is not None:
        circ = init
    else:
        circ = QuantumCircuit(quantum_reg)
    i = 0
    j = 0
    while i + 1 < numqubits:
        add_single_SO4_gate(circ, i, i+1, so4_parameters, 6*j)
        j = j + 1
        i = i + 2
    i = 1
    while i + 1 < numqubits:
        add_single_SO4_gate(circ, i, i+1, so4_parameters, 6*j)
        j = j +1
        i = i + 2

    myLogger.info('Fine di construct_SO4_ansatz')

    return circ

def create_lagrange_operator_PS(hamiltonian,
                             auxiliary,
                             multiplier,
                             operator,
                             value):
    myLogger.info('Inizio di create_lagrange_operator_PS')

    if operator == "number":
        idx = 0
    elif operator == "spin-squared":
        idx = 1
    elif operator == "spin-z":
        idx = 2

    list_x_zeros = np.zeros(hamiltonian.num_qubits)
    list_z_zeros = np.zeros(hamiltonian.num_qubits)

    equality = auxiliary[idx].add(PauliOp(Pauli((list_z_zeros,
                                                 list_x_zeros)),
                                          -value))

    penalty_squared = (equality ** 2).mul(multiplier)

    lagrangian = hamiltonian.add(penalty_squared)

    myLogger.info('Fine di create_lagrange_operator_PS')

    return lagrangian

def create_lagrange_operator_aug(hamiltonian,
                             auxiliary,
                             multiplier_simple,
                             multiplier_square,
                             operator,
                             value):
    myLogger.info('Inizio di create_lagrange_operator_aug')

    if operator == "number":
        idx = 0
    elif operator == "spin-squared":
        idx = 1
    elif operator == "spin-z":
        idx = 2

    list_x_zeros = np.zeros(hamiltonian.num_qubits)
    list_z_zeros = np.zeros(hamiltonian.num_qubits)

    equality = auxiliary[idx].add(PauliOp(Pauli((list_z_zeros,
                                                 list_x_zeros)),
                                          -value))

    penalty_squared = (equality ** 2).mul(multiplier_square)
    penalty_simple = equality.mul(-multiplier_simple)

    lagrangian = hamiltonian.add(penalty_squared).add(penalty_simple)

    myLogger.info('Fine di create_lagrange_operator_aug')

    return lagrangian

def prepare_base_VQE(options):
    myLogger.info('Inizio di prepare_base_VQE')

    spin = options['molecule']['spin']
    charge = options['molecule']['charge']
    basis = options['molecule']['basis']
    geometry = options['molecule']['geometry']

    var_form_type = options['var_form_type']
    quantum_instance = options['quantum_instance']
    optimizer = options['optimizer']
    converter = options['converter']
    init_point = options['init_point']


    driver = PySCFDriver(atom=geometry,
                         unit=UnitsType.ANGSTROM,
                         basis=basis,
                         spin=spin,
                         charge=charge,
                         method=MethodType.RHF)

    problem = ElectronicStructureProblem(driver)
    main_op = problem.second_q_ops()[0]

    driver_result = driver.run()
    particle_number = driver_result.get_property('ParticleNumber')

    alpha, beta = particle_number.num_alpha, particle_number.num_beta
    num_particles = (alpha, beta)
    num_spin_orbitals = particle_number.num_spin_orbitals

    qubit_op = converter.convert(main_op, num_particles=num_particles)

    init_state = HartreeFock(num_spin_orbitals,
                             num_particles,
                             converter)

    num_qubits = qubit_op.num_qubits

    vqe_solver = create_VQE_from_ansatz_type(var_form_type,
                                         num_qubits,
                                         init_state,
                                         quantum_instance,
                                         optimizer,
                                         converter,
                                         num_particles,
                                         num_spin_orbitals,
                                         init_point)

    myLogger.info('Fine di prepare_base_VQE')

    return converter, vqe_solver, problem, qubit_op

def store_intermediate_result(count, par, energy, std):
    global parameters
    parameters.append(par)
    log_string = str(count) + ' ' + str(energy) + ' ' + str(std)
    myLogger.info(log_string)

def create_VQE_from_ansatz_type(var_form_type,
                            num_qubits,
                            init_state,
                            quantum_instance,
                            optimizer,
                            converter,
                            num_particles,
                            num_spin_orbitals,
                            initial_point):
    myLogger.info('Inizio create_VQE_from_ansatz_type')

    if var_form_type == 'TwoLocal':
        ansatz = TwoLocal(num_qubits=num_qubits,
                          rotation_blocks='ry',
                          entanglement_blocks='cx',
                          initial_state=init_state,
                          entanglement='linear')
        if None in initial_point:
            initial_point = np.random.rand(ansatz.num_parameters)
        vqe_solver = VQE(ansatz=ansatz,
                         optimizer=optimizer,
                         initial_point=initial_point,
                         callback=store_intermediate_result,
                         quantum_instance=quantum_instance)
    elif var_form_type == 'EfficientSU(2)':
        ansatz = EfficientSU2(num_qubits=num_qubits,
                              entanglement='linear',
                              initial_state=init_state)
        if None in initial_point:
            initial_point = np.random.rand(ansatz.num_parameters)
        vqe_solver = VQE(ansatz=ansatz,
                         optimizer=optimizer,
                         initial_point=initial_point,
                         callback=store_intermediate_result,
                         quantum_instance=quantum_instance)

    elif var_form_type == 'UCCSD':
        ansatz = UCCSD(qubit_converter=converter,
                       initial_state=init_state,
                       num_particles=num_particles,
                       num_spin_orbitals=num_spin_orbitals)._build()

        if None in initial_point:
            initial_point = np.random.rand(ansatz.num_parameters)
        vqe_solver = VQE(quantum_instance=quantum_instance,
                         ansatz=ansatz,
                         optimizer=optimizer,
                         callback=store_intermediate_result,
                         initial_point=initial_point)

    elif var_form_type == 'SO(4)':
        ansatz = construct_SO4_ansatz(num_qubits,
                                    init=init_state)
        if None in initial_point:
            initial_point = np.random.rand(6*(num_qubits-1))
        vqe_solver = VQE(ansatz=ansatz,
                         optimizer=optimizer,
                         initial_point=initial_point,
                         callback=store_intermediate_result,
                         quantum_instance=quantum_instance)

    else:
        raise Exception("VAR_FORM_TYPE NOT EXISTS")

    myLogger.info('Fine  create_VQE_from_ansatz_type')

    return vqe_solver

def solve_hamiltonian_VQE(options):
    myLogger.info('Inizio solve_hamiltonian_VQE')
    myLogger.info('OPTIONS')
    myLogger.info(options)

    converter, vqe_solver, problem, qubit_op = prepare_base_VQE(options)

    calc = GroundStateEigensolver(converter, vqe_solver)
    result = calc.solve(problem)

    myLogger.info('Fine solve_hamiltonian_VQE')
    myLogger.info('RESULT')
    myLogger.info(result)

    return result

def solve_lagrangian_VQE(options):
    myLogger.info('Inizio solve_lagrangian_VQE')
    myLogger.info('OPTIONS')
    myLogger.info(options)

    converter, vqe_solver, problem, qubit_op = prepare_base_VQE(options)

    aux_ops_not_converted = problem.second_q_ops()[1:4]
    aux_ops = convert_list_fermOp_to_qubitOp(aux_ops_not_converted,
                                         converter,
                                         problem.num_particles)

    lagrange_op = qubit_op
    for operatore in options['lagrange']['operators']:
        operator = operatore[0]
        value = operatore[1]
        multiplier = operatore[2]
        lagrange_op = create_lagrange_operator_PS(lagrange_op,
                                               aux_ops,
                                               multiplier=multiplier,
                                               operator=operator,
                                               value=value)

    old_result = vqe_solver.compute_minimum_eigenvalue(operator=lagrange_op,
                                                 aux_operators=aux_ops)

    new_result = problem.interpret(old_result)

    myLogger.info('Fine solve_lagrangian_VQE')
    myLogger.info('RESULT')
    myLogger.info(new_result)

    return new_result

def solve_aug_lagrangian_VQE(options, lamb):
    myLogger.info('Inizio solve_aug_lagrangian_VQE')
    myLogger.info('OPTIONS')
    myLogger.info(options)

    converter, vqe_solver, problem, qubit_op = prepare_base_VQE(options)

    if options['var_form_type'] == 'UCCSD':
        vqe_solver = vqe_solver.get_solver(problem, converter)

    aux_ops_not_converted = problem.second_q_ops()[1:4]
    aux_ops = convert_list_fermOp_to_qubitOp(aux_ops_not_converted,
                                         converter,
                                         problem.num_particles)

    lagrange_op = qubit_op
    for operatore in options['lagrange']['operators']:
        operator = operatore[0]
        value = operatore[1]
        multiplier = operatore[2]
        lagrange_op = create_lagrange_operator_aug(lagrange_op,
                                                aux_ops,
                                                multiplier_square=multiplier,
                                                multiplier_simple=lamb,
                                                operator=operator,
                                                value=value)

    old_result = vqe_solver.compute_minimum_eigenvalue(operator=lagrange_op,
                                                      aux_operators=aux_ops)

    new_result = problem.interpret(old_result)

    myLogger.info('Fine solve_aug_lagrangian_VQE')
    myLogger.info('RESULT')
    myLogger.info(new_result)

    return new_result

def convert_list_fermOp_to_qubitOp(old_aux_ops, converter, num_particles):
    myLogger.info('Inizio convert_list_fermOp_to_qubitOp')

    new_aux_ops = []
    for op in old_aux_ops:
        op_new = converter.convert(op, num_particles)
        new_aux_ops.append(op_new)

    myLogger.info('Fine convert_list_fermOp_to_qubitOp')

    return new_aux_ops

def find_best_result(partial_results):
    energy_min = 100
    tmp_result = 5
    for result in partial_results:
        if result.total_energies[0] < energy_min:
            energy_min = result.total_energies[0]
            tmp_result = result

    return tmp_result

def calc_penalty(lag_op_list, result, threshold, tmp_mult):
    penalty = 0
    accectable_result = True

    for operatore in lag_op_list:
        if operatore[0] == 'number':
            penalty += tmp_mult*((result.num_particles[0] - operatore[1])**2)
            if abs(result.num_particles[0] - operatore[1]) > threshold:
                accectable_result = False
        if operatore[0] == 'spin-squared':
            penalty += tmp_mult*((result.total_angular_momentum[0] - operatore[1])**2)
            if abs(result.total_angular_momentum[0] - operatore[1]) > threshold:
                accectable_result = False
        if operatore[0] == 'spin-z':
            penalty += tmp_mult*((result.spin[0] - operatore[1])**2)
            if abs(result.spin[0] - operatore[1]) > threshold:
                    accectable_result = False

    return penalty, accectable_result

def solve_lag_series_VQE(options):
    iter_max = options['series']['itermax']
    step = options['series']['step']
    par = np.zeros(get_num_par(options['var_form_type'], options['molecule']['molecule'][0]))
    mult = 0.01
    threshold = 0.5

    global parameters
    parameters = [par]

    partial_results = []

    for i in range(iter_max):
        tmp_mult = mult + step * i

        lag_op_list = []

        for single_op in options['lagrange']['operators']:
            operatore = (single_op[0],
                         single_op[1],
                         float(tmp_mult))
            lag_op_list.append(operatore)

        options['lagrange']['operators'] = lag_op_list

        options['init_point'] = par
        result = solve_lagrangian_VQE(options)
        par = parameters[len(parameters) - 1]

        penalty, accectable_result = calc_penalty(lag_op_list,
                                                result,
                                                threshold,
                                                tmp_mult)

        log_str = "Iter " + str(i)
        log_str += " mult " + str(np.round(tmp_mult, 2))
        log_str += "\tE = " + str(np.round(result.total_energies[0], 7))
        log_str += "\tP = " + str(penalty)
        log_str += "\tE-P = " + str(np.round(result.total_energies[0] - penalty, 7))

        myLogger.info(log_str)
        #print('\t\t', result.total_energies[0], '\t', penalty)

        if accectable_result:
            result.total_energies[0] = result.total_energies[0] - penalty
            partial_results.append(result)

    result = find_best_result(partial_results)

    return result

def get_num_par(varform, mol_type):
    if mol_type == 'H3+':
        if varform == 'TwoLocal':
            return 16
        elif varform == 'SO(4)':
            return 18
        elif varform == 'UCCSD':
            return 16
        elif varform == 'EfficientSU(2)':
            return 32
    elif 'H2' in mol_type:
        if varform == 'TwoLocal':
            return 8
        elif varform == 'SO(4)':
            return 6
        elif varform == 'UCCSD':
            return 8
        elif varform == 'EfficientSU(2)':
            return 16
    elif 'H4' in mol_type:
        if varform == 'TwoLocal':
            return 24
        elif varform == 'SO(4)':
            return 30
        elif varform == 'UCCSD':
            return 24
        elif varform == 'EfficientSU(2)':
            return 48
    else:
        raise Exception('mol_type not totally implemented')

def solve_lag_aug_series_VQE(options):
    iter_max = options['series']['itermax']
    par = np.zeros(get_num_par(options['var_form_type'], options['molecule']['molecule'][0]))
    mult = 0.01
    step = options['series']['step']

    lamb = options['series']['lamb']

    global parameters
    parameters = [par]
    for i in range(iter_max):
        tmp_mult = mult + step * i

        lag_op_list = []

        for single_op in options['lagrange']['operators']:
            operatore = (single_op[0],
                         single_op[1],
                         float(tmp_mult))
            lag_op_list.append(operatore)

        options['lagrange']['operators'] = lag_op_list

        options['init_point'] = par

        result = solve_aug_lagrangian_VQE(options, lamb)

        penalty = tmp_mult*((result.num_particles[0] - operatore[1])**2)
        penalty -= lamb * (result.num_particles[0] - operatore[1])

        log_str = "Iter " + str(i)
        log_str += " mult " + str(np.round(tmp_mult, 2))
        log_str += " lamb " + str(lamb)
        log_str += "\tE = " + str(np.round(result.total_energies[0], 7))
        log_str += "\tE-P = " + str(np.round(result.total_energies[0] - penalty, 7))
        myLogger.info(log_str)

        par = parameters[len(parameters) - 1]

        for operatore in lag_op_list:
            if operatore[0] == 'number':
                lamb = lamb - tmp_mult*2*(result.num_particles[0] - operatore[1])
            if operatore[0] == 'spin-squared':
                lamb = lamb - tmp_mult*2*(result.total_angular_momentum[0] - operatore[1])
            if operatore[0] == 'spin-z':
                lamb = lamb - tmp_mult*2*(result.spin[0] - operatore[1])

        #print(result.total_energies[0] - penalty, " ", penalty)

    return result

def solve_VQE(options):
    global parameters
    parameters = []

    if not options['lagrange']['active']:
        return solve_hamiltonian_VQE(options)
    if not options['lagrange']['series']:
        return solve_lagrangian_VQE(options)
    else:
        if options['lagrange']['augmented']:
            return solve_lag_aug_series_VQE(options)
        else:
            return solve_lag_series_VQE(options)

