import itertools

import numpy as np
import PySimpleGUI as psg

import warnings
#warnings.filterwarnings('ignore', category=SystemTimeWarning)
warnings.simplefilter("ignore")

from qiskit.algorithms.optimizers import CG, COBYLA, SPSA, L_BFGS_B, ADAM
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
        'optimizer' : ['COBYLA'],
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
        'lag_mult_spinz_delta' : 0.1,
        'series_itermax' : 10,
        'series_step' : 0.2
    }
    return values

def retriveVQEOptions(argv):
    possibleForms = ['TwoLocal', 'SO(4)', 'UCCSD', 'EfficientSU(2)']
    possibleBasis = ['sto-3g', 'sto-6g']
    possibleNoise = ['None', 'ibmq_santiago']
    possibleBool  = ['True', 'False']
    possibleLag   = ['True', 'False', 'Series']
    possibleOptim = ['COBYLA', 'CG', 'SPSA', 'L_BFGS_B', 'ADAM']
    possibleLagop = ['number','spin-squared','spin-z', 'num+spin2', 'spin2+spinz', 'num+spinz', 'num+spin2+spinz']
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
             psg.Listbox(possibleOptim, default_values=['COBYLA'], select_mode='extended',size=(8,5),               key='optimizer')],
            [psg.Text('Distanze')],
            [psg.Text('Min'), psg.Input(default_text=0.3,size=(4,10),                                               key='dist_min'),
             psg.Text('Max'), psg.Input(default_text=3.5,size=(4,10),                                               key='dist_max'),
             psg.Text('Delta'), psg.Input(default_text=0.1,size=(4,10),                                             key='dist_delta')],
            [psg.Text('Lagrangiana'),
             psg.Listbox(possibleLag,default_values=['False'], select_mode='extended',size=(5,3),                   key='lagrangiana')],

            [psg.Text('LagSeries: itermax'), psg.Input(default_text=10,size=(4,10),                                 key='series_itermax'),
             psg.Text('step: '), psg.Input(default_text=0.2,size=(4,10),                                            key='series_step')],

            [psg.Text('Operatore'),
             psg.Listbox(possibleLagop,default_values=['number'], select_mode='extended',size=(14,7),               key='lag_op')],

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
        },
        'series' : {
            'itermax' : int(values['series_itermax']),
            'step' : float(values['series_step'])
        }
    }

    options = setDistAndGeometry(options)

    printChoseOption(options)

    return options

def setLagrangeOps(values):
    if values['lagrangiana'] == ['False']:
        lagops_list = [[('dummy', 0, 0)]]
    else:
        lagops_list = []

    for op in values['lag_op']:
        if op == 'number':
            mults = np.arange(float(values['lag_mult_num_min']), float(values['lag_mult_num_max']), float(values['lag_mult_num_delta']))
            for mult in mults:
                lagops_list.append([(op, int(values['lag_value_num']), mult)])
        if op == 'spin-squared':
            mults = np.arange(float(values['lag_mult_spin2_min']), float(values['lag_mult_spin2_max']), float(values['lag_mult_spin2_delta']))
            for mult in mults:
                lagops_list.append([(op, int(values['lag_value_spin2']), mult)])
        if op == 'spin-z':
            mults = np.arange(float(values['lag_mult_spinz_min']), float(values['lag_mult_spinz_max']), float(values['lag_mult_spinz_delta']))
            for mult in mults:
                lagops_list.append([(op, int(values['lag_value_spinz']), mult)])
        #-----
        if op == 'num+spin2':
            mults_num = np.arange(float(values['lag_mult_num_min']), float(values['lag_mult_num_max']), float(values['lag_mult_num_delta']))
            mults_spin2 = np.arange(float(values['lag_mult_spin2_min']), float(values['lag_mult_spin2_max']), float(values['lag_mult_spin2_delta']))
            for mult_num in mults_num:
                for mult_spin2 in mults_spin2:
                    templist = []
                    templist.append(('number', int(values['lag_value_num']), mult_num))
                    templist.append(('spin-squared', int(values['lag_value_spin2']), mult_spin2))
                    lagops_list.append(templist)

        #-----
        if op == 'num+spinz':
            mults_num = np.arange(float(values['lag_mult_num_min']), float(values['lag_mult_num_max']), float(values['lag_mult_num_delta']))
            mults_spinz = np.arange(float(values['lag_mult_spinz_min']), float(values['lag_mult_spinz_max']), float(values['lag_mult_spinz_delta']))
            for mult_num in mults_num:
                for mult_spinz in mults_spinz:
                    templist = []
                    templist.append(('number', int(values['lag_value_num']), mult_num))
                    templist.append(('spin-z', int(values['lag_value_spinz']), mult_spinz))
                    lagops_list.append(templist)

        #-----
        if op == 'spin2+spinz':
            mults_spin2 = np.arange(float(values['lag_mult_spin2_min']), float(values['lag_mult_spin2_max']), float(values['lag_mult_spin2_delta']))
            mults_spinz = np.arange(float(values['lag_mult_spinz_min']), float(values['lag_mult_spinz_max']), float(values['lag_mult_spinz_delta']))
            for mult_spin2 in mults_spin2:
                for mult_spinz in mults_spinz:
                    templist = []
                    templist.append(('spin-squared', int(values['lag_value_spin2']), mult_spin2))
                    templist.append(('spin-z', int(values['lag_value_spinz']), mult_spinz))
                    lagops_list.append(templist)

        #-----
        if op == 'num+spin2+spinz':
            mults_spin2 = np.arange(float(values['lag_mult_spin2_min']), float(values['lag_mult_spin2_max']), float(values['lag_mult_spin2_delta']))
            mults_num = np.arange(float(values['lag_mult_num_min']), float(values['lag_mult_num_max']), float(values['lag_mult_num_delta']))
            mults_spinz = np.arange(float(values['lag_mult_spinz_min']), float(values['lag_mult_spinz_max']), float(values['lag_mult_spinz_delta']))
            for mult_spin2 in mults_spin2:
                for mult_spinz in mults_spinz:
                    for mult_num in mults_num:
                        templist = []
                        templist.append(('spin-squared', int(values['lag_value_spin2']), mult_spin2))
                        templist.append(('spin-z', int(values['lag_value_spinz']), mult_spinz))
                        templist.append(('number', int(values['lag_value_num']), mult_num))
                        lagops_list.append(templist)

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
            optimizers.append((CG(maxiter=1000, eps=1e-5), 'CG'))
        elif opt == 'COBYLA':
            optimizers.append((COBYLA(), 'COBYLA'))
        elif opt == 'ADAM':
            optimizers.append((ADAM(maxiter=1000, eps=1e-5), 'ADAM'))
        elif opt == 'L_BFGS_B':
            optimizers.append((L_BFGS_B(maxiter=1000, epsilon=1e-5), 'LBFGSB'))
        elif opt == 'SPSA':
            optimizers.append((SPSA(maxiter=2000), 'SPSA'))
    values['optimizer'] = optimizers

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
