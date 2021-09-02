import itertools

import numpy as np
import PySimpleGUI as psg

import warnings
#warnings.filterwarnings('ignore', category=SystemTimeWarning)
warnings.simplefilter("ignore")

from qiskit.algorithms.optimizers import CG, COBYLA
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit.providers.aer.noise import NoiseModel
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit.utils import QuantumInstance
from qiskit.ignis.mitigation import CompleteMeasFitter
from qiskit import Aer
from qiskit.providers.aer import QasmSimulator
from qiskit import IBMQ

def getDefaultOpt():
    print('Loading default values')
    values = {
        'spin' : 0,
        'charge' : 1,
        'basis' : ['sto-6g'],
        'varforms' : ['TwoLocal'],
        'backend' : ['statevector_simulator'],
        'noise' : ['None'],
        'shots' : 1024,
        'correction' : ['False'],
        'optimizer' : ['CG'],
        'dist_min' : 0.3,
        'dist_max' : 3.5,
        'dist_delta' : 11,
        'lagrangiana' : ['False'],
        'lag_op' : ['number'],
        'lag_value_num' : 2,
        'lag_mult_num_min' : 0.2,
        'lag_mult_num_max' : 0.2,
        'lag_mult_num_delta' : 0.1,
        'lag_value_spin2' : 0,
        'lag_mult_spin2_min' : 0.2,
        'lag_mult_spin2_max' : 0.2,
        'lag_mult_spin2_delta' : 0.1,
        'lag_value_spinz' : 0,
        'lag_mult_spinz_min' : 0.2,
        'lag_mult_spinz_max' : 0.2,
        'lag_mult_spinz_delta' : 0.1
    }
    return values

def retriveVQEOptions(argv):
    possibleForms = ['TwoLocal', 'SO(4)', 'UCCSD', 'EfficientSU(2)']
    possibleBasis = ['sto-3g', 'sto-6g']
    possibleNoise = ['None', 'ibmq_santiago']
    possibleBool  = ['True', 'False']
    possibleOptim = ['CG', 'COBYLA']
    possibleLagop = ['number','spin-squared','spin-z']
    possibleBack  = ['statevector_simulator','qasm_simulator','hardware']

    layout=[[psg.Text('Molecola')],
            [psg.Text('Spin'), psg.Input(default_text=0,size=(4,10),                                                key='spin'),
             psg.Text('Charge'), psg.Input(default_text=1,size=(4,10),                                              key='charge'),
             psg.Text('Basis'),
             psg.Listbox(possibleBasis, default_values=['sto-6g'],select_mode='extended',size=(7,2),                key='basis')],
            [psg.Text('Scegli forma variazionale')],
            [psg.Listbox(possibleForms, default_values=['TwoLocal'],select_mode='extended',size=(12,4),             key='varforms')],
            [psg.Text('Scegli il tipo di backend')],
            [psg.Listbox(possibleBack,default_values=['statevector_simulator'], select_mode='extended',size=(17,3), key='backend'),
             psg.Text('shots'), psg.Input(default_text=1024,size=(4,10),                                            key='shots')],
            [psg.Text('Scegli l\'eventuale rumore'),
             psg.Listbox(possibleNoise, default_values=['None'], select_mode='extended',size=(14,2),                key='noise')],
            [psg.Text('Correction'),
             psg.Listbox(possibleBool, default_values=['False'], select_mode='extended',size=(5,2),                 key='correction')],
            [psg.Text('Optimizer'),
             psg.Listbox(possibleOptim, default_values=['CG'], select_mode='extended',size=(8,2),                   key='optimizer')],
            [psg.Text('Distanze')],
            [psg.Text('Min'), psg.Input(default_text=0.3,size=(4,10),                                               key='dist_min'),
             psg.Text('Max'), psg.Input(default_text=3.5,size=(4,10),                                               key='dist_max'),
             psg.Text('Delta'), psg.Input(default_text=0.1,size=(4,10),                                             key='dist_delta')],
            [psg.Text('Lagrangiana'),
             psg.Listbox(possibleBool,default_values=['False'], select_mode='extended',size=(5,2),                  key='lagrangiana'),
             psg.Text('Operatore'),
             psg.Listbox(possibleLagop,default_values=['number'], select_mode='extended',size=(14,3),               key='lag_op')],

            [psg.Text('NUMBER: Value'), psg.Input(default_text=2,size=(4,10),                                       key='lag_value_num'),
             psg.Text('Mult: min'),  psg.Input(default_text=0.2,size=(4,10),                                        key='lag_mult_num_min'),
             psg.Text('max'), psg.Input(default_text=1,size=(4,10),                                                 key='lag_mult_num_max'),
             psg.Text('delta'), psg.Input(default_text=0.1,size=(4,10),                                             key='lag_mult_num_delta')],

            [psg.Text('SPIN-2: Value'), psg.Input(default_text=0,size=(4,10),                                       key='lag_value_spin2'),
             psg.Text('Mult: min'),  psg.Input(default_text=0.2,size=(4,10),                                        key='lag_mult_spin2_min'),
             psg.Text('max'), psg.Input(default_text=1,size=(4,10),                                                 key='lag_mult_spin2_max'),
             psg.Text('delta'), psg.Input(default_text=0.1,size=(4,10),                                             key='lag_mult_spin2_delta')],

            [psg.Text('SPIN-Z: Value'), psg.Input(default_text=0,size=(4,10),                                       key='lag_value_spinz'),
             psg.Text('Mult: min'),  psg.Input(default_text=0.2,size=(4,10),                                        key='lag_mult_spinz_min'),
             psg.Text('max'), psg.Input(default_text=1,size=(4,10),                                                 key='lag_mult_spinz_max'),
             psg.Text('delta'), psg.Input(default_text=0.1,size=(4,10),                                             key='lag_mult_spinz_delta')],

            [psg.Button('Inizia calcolo', font=('Times New Roman',12))]]

    if len(argv) > 1:
        if argv[1] == 'fast':
            values = getDefaultOpt()
    else:
        win =psg.Window('Definisci opzioni per VQE',
                        layout,
                        resizable=True,
                        font='Lucida',
                        text_justification='left')
        e,values=win.read()
        win.close()

    setOptimizers(values)
    setLagrangian(values)
    setBackendAndNoise(values)
    lagops = setLagrangeOps(values)

    options = {
        'dist' : {
            'min' : values['dist_min'],
            'max' : values['dist_max'],
            'delta' : values['dist_delta']
        },
        'molecule' : {
            'spin' : int(values['spin']),
            'charge' : int(values['charge']),
            'basis' : values['basis']
        },
        'varforms' : values['varforms'],
        'quantum_instance' : values['backend'],
        'optimizer' : values['optimizer'],
        'converter' : QubitConverter(mapper = ParityMapper(),
                                   two_qubit_reduction = True),
        'lagrange' : {
            'active' : values['lagrangiana'],
            'operators' : lagops,
        }
    }

    options = setDistAndGeometry(options)

    printChoseOption(options)

    return options

def setLagrangeOps(values):
    lagops_list = [('dummy', 0, 0)]
    for op in values['lag_op']:
        if op == 'number':
            mults = np.arange(float(values['lag_mult_num_min']), float(values['lag_mult_num_max']), float(values['lag_mult_num_delta']))
            for mult in mults:
                lagops_list.append((op, values['lag_value_num'], mult))
        if op == 'spin-squared':
            mults = np.arange(float(values['lag_mult_spin2_min']), float(values['lag_mult_spin2_max']), float(values['lag_mult_spin2_delta']))
            for mult in mults:
                lagops_list.append((op, values['lag_value_spin2'], mult))
        if op == 'spin-z':
            mults = np.arange(float(values['lag_mult_spinz_min']), float(values['lag_mult_spinz_max']), float(values['lag_mult_spinz_delta']))
            for mult in mults:
                lagops_list.append((op, values['lag_value_spinz'], mult))
    return lagops_list

def printChoseOption(options):
    print('OPZIONI SCELTE')
    print('dist: ', options['dists'])
    print('basis: ', options['molecule']['basis'])
    print('varforms: ', options['varforms'])
    print('instances: ', options['quantum_instance'])
    print('optimizer: ', options['optimizer'])
    print('lagrange: ',
          '\n\tactive: ', options['lagrange']['active'],
          '\n\toperator: ', options['lagrange']['operators'])

def setOptimizers(values):
    optimizers = []
    for opt in values['optimizer']:
        if opt == 'CG':
            optimizers.append((CG(), 'CG'))
        if opt == 'COBYLA':
            optimizers.append((COBYLA(), 'COBYLA'))
    values['optimizer'] = optimizers

def setLagrangian(values):
    lagrangian_list = []
    for lag in values['lagrangiana']:
        if lag == 'True':
            lagrangian_list.append(True)
        else:
            lagrangian_list.append(False)
    values['lagrangiana'] = lagrangian_list

def setBackendAndNoise(values):
    quantum_instance_list = []

    provider = IBMQ.load_account()

    iteratore = list(itertools.product(values['backend'],values['noise'],values['correction']))

    for backend,noise,corr in iteratore:

        if backend == 'statevector_simulator':
            quantum_instance = QuantumInstance(Aer.get_backend('statevector_simulator'))
            noise = 'None'
        elif backend == 'qasm_simulator':
            quantum_instance = QuantumInstance(backend=QasmSimulator(method='statevector'),
                                               skip_qobj_validation=True,
                                               shots=int(values['shots']))
        if backend == 'qasm_simulator' and noise != 'None':
            device = provider.get_backend(noise)
            coupling_map = device.configuration().coupling_map
            noise_model = NoiseModel.from_backend(device)
            basis_gates = noise_model.basis_gates
            seed = 150
            quantum_instance.seed_simulator = seed
            quantum_instance.seed_transpiler = seed
            quantum_instance.coupling_map = coupling_map
            quantum_instance.noise_model = noise_model

        name = backend + '_' + noise

        if backend == 'qasm_simulator' and noise != 'None' and corr == 'True':
            quantum_instance.measurement_error_mitigation_cls = CompleteMeasFitter
            quantum_instance.measurement_error_mitigation_shots = 1000
            name += '_corrected'

        quantum_instance_list.append((quantum_instance,name))

    values['backend'] = quantum_instance_list

def setDistAndGeometry(options):
    minimo = float(options['dist']['min'])
    massimo = float(options['dist']['max'])
    delta = float(options['dist']['delta'])

    dist = np.arange(minimo,massimo,delta)
    options['dists'] = dist
    alt = np.sqrt(dist**2 - (dist/2)**2, dtype='float64')

    geometries = []

    for i, d in enumerate(dist):
        geom = "H .0 .0 .0; H .0 .0 " + str(d) + "; H .0 " + str(alt[i]) + " " + str(d/2)
        geometries.append(geom)

    options['geometries'] = geometries

    return options
