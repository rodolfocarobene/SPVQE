import numpy as np

from qiskit_nature.drivers import PySCFDriver, UnitsType, HFMethodType, Molecule
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.mappers.second_quantization import ParityMapper, BravyiKitaevMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.circuit.library import HartreeFock

from qiskit_nature.algorithms import VQEUCCFactory, GroundStateEigensolver

from qiskit_nature.results import EigenstateResult

from qiskit.algorithms.optimizers import L_BFGS_B, CG
from qiskit.algorithms import VQE

from qiskit.circuit import QuantumRegister, Parameter
from qiskit.circuit.library import TwoLocal

from qiskit.utils import QuantumInstance

from qiskit import IBMQ
from qiskit import Aer
from qiskit import QuantumCircuit

from qiskit.opflow.primitive_ops import PauliOp
from qiskit.quantum_info import Pauli

def add_unitary_gate(circuit,
                     qubit1,
                     qubit2,
                     params,
                     p0):
    circuit.s(qubit1)
    circuit.s(qubit2)
    circuit.h(qubit2)
    circuit.cx(qubit2,qubit1)
    circuit.u(params[p0],params[p0+1],params[p0+2],qubit1); p0 += 3
    circuit.u(params[p0],params[p0+1],params[p0+2],qubit2); p0 += 3
    circuit.cx(qubit2,qubit1)
    circuit.h(qubit2)
    circuit.sdg(qubit1)
    circuit.sdg(qubit2)

#definition of SO(4) variational form for arbitrary nuber of qubits

def SO_04(numqubits, init = None):
    num_parameters = 6*(numqubits-1)
    parameters = []
    for i in range(num_parameters):
        name = "par" + str(i)
        new = Parameter(name)
        parameters.append(new)
    q = QuantumRegister(numqubits, name='q')
    if init != None:
        circ = init
    else:
        circ = QuantumCircuit(q)
    i = 0
    n = 0
    while i + 1 < numqubits:
        add_unitary_gate(circ , i, i+1, parameters , 6*n)
        n = n + 1
        i = i + 2
    i = 1
    while i + 1 < numqubits:
        add_unitary_gate(circ , i, i+1, parameters , 6*n)
        n = n +1
        i = i + 2
    return circ

def CreateLagrangeOperator(h_op,
                           a_op,
                           multiplier,
                           operator = "number",
                           value=2):

    # TODO: this switch sucks
    if operator == "number":
        idx = 0
    elif operator == "spin-squared":
        idx = 1
    elif operator == "spin-z":
        idx = 2

    x = np.zeros(h_op.num_qubits)
    z = np.zeros(h_op.num_qubits)

    o_opt = PauliOp(Pauli((z,x)), -value)

    mult = a_op[idx].add(o_opt).mul(multiplier)
    L_op  =  h_op.add(mult)

    return L_op

def PrepareBaseVQE(options):

    spin = options['molecule']['spin']
    charge = options['molecule']['charge']
    basis = options['molecule']['basis']
    geometry = options['molecule']['geometry']

    var_form_type = options['var_form_type']
    quantum_instance = options['quantum_instance']
    optimizer = options['optimizer']
    converter = options['converter']


    #setting up molecule driver

    driver = PySCFDriver(atom = geometry,
                         unit = UnitsType.ANGSTROM,
                         basis = basis,
                         spin = spin,
                         charge = charge,
                         hf_method = HFMethodType.RHF)

    problem = ElectronicStructureProblem(driver)
    main_op = problem.second_q_ops()[0]

    molecule = driver.run()

    alpha, beta = molecule.num_alpha, molecule.num_beta
    num_particles = (alpha, beta)
    num_spin_orbitals = molecule.num_molecular_orbitals*2

    qubit_op = converter.convert(main_op, num_particles=num_particles)

    init_state = HartreeFock(num_spin_orbitals,
                             num_particles,
                             converter)

    #ansatz

    num_qubits = qubit_op.num_qubits

    vqe_solver = CreateVQEFromAnsatzType(var_form_type,
                                         num_qubits,
                                         init_state,
                                         quantum_instance,
                                         optimizer)


    return converter, vqe_solver, problem, qubit_op

def CreateVQEFromAnsatzType(var_form_type,
                            num_qubits,
                            init_state,
                            quantum_instance,
                            optimizer):

    if var_form_type == 'TwoLocal':
        ansatz = TwoLocal(num_qubits = num_qubits,
                          rotation_blocks = 'ry',
                          entanglement_blocks = 'cx',
                          initial_state = init_state,
                          entanglement = 'linear')
        vqe_solver = VQE(ansatz = ansatz,
                         optimizer = optimizer,
                         quantum_instance = quantum_instance)
    elif var_form_type == 'UCCSD':
        vqe_solver = VQEUCCFactory(quantum_instance,
                                   initial_state = init_state)

    elif var_form_type == 'SO(4)':
        ansatz = SO_04(num_qubits,
                       init = init_state)
        vqe_solver = VQE(ansatz = ansatz,
                         optimizer = optimizer,
                         quantum_instance = quantum_instance)

    else:
        raise Exception("VAR_FORM_TYPE NOT EXISTS")
    return vqe_solver

def SolveHamiltonianVQE(options):
    converter, vqe_solver, problem, qubit_op = PrepareBaseVQE(options)

    calc = GroundStateEigensolver(converter, vqe_solver)
    result = calc.solve(problem)

    return result

def SolveLagrangianVQE(options):
    converter, vqe_solver, problem, qubit_op = PrepareBaseVQE(options)

    auxOpsNotConverted = problem.second_q_ops()[1:4]
    aux_ops = ConvertListFermOpToQubitOp(auxOpsNotConverted,
                                         converter,
                                         problem.num_particles)

    multiplier = options['lagrange']['multiplier']
    operator = options['lagrange']['operator']
    value = options['lagrange']['value']

    lagrange_op = CreateLagrangeOperator(qubit_op,
                                         aux_ops,
                                         multiplier = multiplier,
                                         operator = operator,
                                         value = value)

    oldResult = vqe_solver.compute_minimum_eigenvalue(operator=lagrange_op,
                                                 aux_operators=aux_ops)

    newResult = problem.interpret(oldResult)
    return newResult

def ConvertListFermOpToQubitOp(old_aux_ops, converter, num_particles):
    new_aux_ops = []
    for op in old_aux_ops:
        opNew = converter.convert(op, num_particles)
        new_aux_ops.append(opNew)
    return new_aux_ops


#---------------------------------------------------------
import json
import git

def WriteJson(JsonOptions, filename):
    repo = git.Repo(search_parent_directories=True)
    commit = repo.head.object.hexsha
    JsonOptions['commit'] = commit


    description = input("Inserisci una minima descrizione: ")

    json_obj = json.dumps(dictionary, indent = 4)

    if os.path.exists(filename):
        raise Exeption('Esiste già un file con questo nome!!')
    else:
        with open(filename, "w") as out:
            out.write(json_obj)
