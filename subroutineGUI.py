import itertools
import warnings

import numpy as np
import PySimpleGUI as psg

from qiskit import Aer
from qiskit import IBMQ

from qiskit.algorithms.optimizers import CG, COBYLA, SPSA, L_BFGS_B, ADAM, NFT
from qiskit.providers.aer.noise import NoiseModel
from qiskit.utils import QuantumInstance
from qiskit.ignis.mitigation import CompleteMeasFitter
from qiskit.providers.aer import QasmSimulator

from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import ParityMapper

warnings.simplefilter("ignore")

def get_default_opt():
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

        'series_itermax_min' : 10,
        'series_itermax_max' : 20,
        'series_itermax_step' : 15,

        'series_step_min' : 0.1,
        'series_step_max' : 1,
        'series_step_step' : 10,

        'series_lamb_min' : 2,
        'series_lamb_max' : 6,
        'series_lamb_step' : 10
    }
    return values

def get_layout():
    possible_molecules = ['H3+', 'H2', 'H2+', 'H2*', 'H4', 'H4*', 'Li2', 'Li2+',
                          'LiH', 'H2O']
    possible_forms = ['TwoLocal', 'SO(4)', 'UCCSD', 'EfficientSU(2)']
    possible_basis = ['sto-3g', 'sto-6g']
    possible_noise = ['None', 'ibmq_santiago']
    possible_bool = ['True', 'False']
    possible_lag = ['True', 'False', 'Series', 'AUGSeries']
    possible_optim = ['COBYLA', 'CG', 'SPSA', 'L_BFGS_B', 'ADAM', 'NFT']
    possible_lag_op = ['number', 'spin-squared', 'spin-z',
                     'num+spin2', 'spin2+spinz', 'num+spinz', 'num+spin2+spinz']
    possible_backend = ['statevector_simulator', 'qasm_simulator',
                        'qasm_simulator_online', 'hardware',
                        'qasm_gpu']

    layout = [[psg.Text('Molecola'),
               psg.Combo(possible_molecules, default_value='H3+',
                         size=(5,1), key='molecule'),
               psg.Text('Basis'),
               psg.Listbox(possible_basis, default_values=['sto-6g'],
                           select_mode='extended', size=(7, 2), key='basis')],

            [psg.Text('Scegli forma variazionale'),
             psg.Listbox(possible_forms, default_values=['TwoLocal'],
                         select_mode='extended', size=(12, 4), key='varforms')],

            [psg.Text('Scegli il tipo di backend'),
             psg.Listbox(possible_backend, default_values=['statevector_simulator'],
                         select_mode='extended', size=(17, 4), key='backend'),
             psg.Text('shots'),
             psg.Input(default_text=1024, size=(4, 10), key='shots')],

            [psg.Text('Scegli l\'eventuale rumore'),
             psg.Listbox(possible_noise, default_values=['None'],
                         select_mode='extended', size=(14, 2), key='noise')],

            [psg.Text('Correction'),
             psg.Listbox(possible_bool, default_values=['False'],
                         select_mode='extended', size=(5, 2), key='correction')],

            [psg.Text('Optimizer'),
             psg.Listbox(possible_optim, default_values=['COBYLA'],
                         select_mode='extended', size=(8, 6), key='optimizer')],

            [psg.Text('Distanze: Min'),
             psg.Input(default_text=0.3, size=(4, 10), key='dist_min'),
             psg.Text('          Max'),
             psg.Input(default_text=3.5, size=(4, 10), key='dist_max'),
             psg.Text('          Delta'),
             psg.Input(default_text=0.1, size=(4, 10), key='dist_delta')],

            [psg.Text('Lagrangiana'),
             psg.Listbox(possible_lag, default_values=['Series'],
                         select_mode='extended', size=(10, 4), key='lagrangiana')],

            [psg.Text('LagSeries(itermax) MIN:'),
             psg.Input(default_text=10, size=(4, 10), key='series_itermax_min'),
             psg.Text('MAX: '),
             psg.Input(default_text=20, size=(4, 10), key='series_itermax_max'),
             psg.Text('DELTA: '),
             psg.Input(default_text=20, size=(4, 10), key='series_itermax_step')],

            [psg.Text('LagSeries(step) MIN:'),
             psg.Input(default_text=0.2, size=(4, 10), key='series_step_min'),
             psg.Text('MAX: '),
             psg.Input(default_text=1, size=(4, 10), key='series_step_max'),
             psg.Text('DELTA: '),
             psg.Input(default_text=10, size=(4, 10), key='series_step_step')],

            [psg.Text('LagSeries(lamb) MIN:'),
             psg.Input(default_text=2, size=(4, 10), key='series_lamb_min'),
             psg.Text('MAX: '),
             psg.Input(default_text=6, size=(4, 10), key='series_lamb_max'),
             psg.Text('DELTA: '),
             psg.Input(default_text=10, size=(4, 10), key='series_lamb_step')],

            [psg.Text('Operatore'),
             psg.Listbox(possible_lag_op, default_values=['number'],
                         select_mode='extended', size=(18, 7), key='lag_op')],

            [psg.Text('NUMBER: Value'),
             psg.Input(default_text=2, size=(4, 10), key='lag_value_num'),
             psg.Text('Mult: min'),
             psg.Input(default_text=0.2, size=(4, 10), key='lag_mult_num_min'),
             psg.Text('max'),
             psg.Input(default_text=1, size=(4, 10), key='lag_mult_num_max'),
             psg.Text('delta'),
             psg.Input(default_text=10, size=(4, 10), key='lag_mult_num_delta')],

            [psg.Text('SPIN-2: Value'),
             psg.Input(default_text=0, size=(4, 10), key='lag_value_spin2'),
             psg.Text('Mult: min'),
             psg.Input(default_text=0.2, size=(4, 10), key='lag_mult_spin2_min'),
             psg.Text('max'),
             psg.Input(default_text=1, size=(4, 10), key='lag_mult_spin2_max'),
             psg.Text('delta'),
             psg.Input(default_text=10, size=(4, 10), key='lag_mult_spin2_delta')],

            [psg.Text('SPIN-Z: Value'),
             psg.Input(default_text=0, size=(4, 10), key='lag_value_spinz'),
             psg.Text('Mult: min'),
             psg.Input(default_text=0.2, size=(4, 10), key='lag_mult_spinz_min'),
             psg.Text('max'),
             psg.Input(default_text=1, size=(4, 10), key='lag_mult_spinz_max'),
             psg.Text('delta'),
             psg.Input(default_text=10, size=(4, 10), key='lag_mult_spinz_delta')],

            [psg.Button('Inizia calcolo', font=('Times New Roman', 12))]]

    return layout

def retrive_VQE_options(argv):

    if len(argv) > 1:
        if argv[1] == 'fast':
            values = get_default_opt()
    else:
        layout = get_layout()
        win = psg.Window('Definisci opzioni per VQE',
                         layout,
                         resizable=True,
                         font='Lucida',
                         text_justification='left')
        e, values = win.read()
        win.close()

    set_optimizers(values)
    set_backend_and_noise(values)
    lagops = set_lagrange_ops(values)

    options = {
        'dist' : {
            'min' : values['dist_min'],
            'max' : values['dist_max'],
            'delta' : values['dist_delta']
        },
        'molecule' : {
            'molecule' : values['molecule'],
            'spin' : get_spin(values['molecule']),
            'charge' : get_charge(values['molecule']),
            'basis' : values['basis']
        },
        'varforms' : values['varforms'],
        'quantum_instance' : values['backend'],
        'shots': int(values['shots']),
        'optimizer' : values['optimizer'],
        'converter' : QubitConverter(mapper=ParityMapper(),
                                     two_qubit_reduction=True),
        'lagrange' : {
            'active' : values['lagrangiana'],
            'operators' : lagops,
        },
        'series' : {
            'itermax' : np.arange(int(values['series_itermax_min']),
                                  int(values['series_itermax_max']),
                                  int(values['series_itermax_step'])),
            'step' : np.arange(float(values['series_step_min']),
                               float(values['series_step_max']),
                               float(values['series_step_step'])),
            'lamb' : np.arange(float(values['series_lamb_min']),
                               float(values['series_lamb_max']),
                               float(values['series_lamb_step']))
        }
    }

    options = set_dist_and_geometry(options)

    print_chose_options(options)

    return options

def set_ops_num_spin2(values, lagops_list):
    mults_num = np.arange(float(values['lag_mult_num_min']),
                          float(values['lag_mult_num_max']),
                          float(values['lag_mult_num_delta']))
    mults_spin2 = np.arange(float(values['lag_mult_spin2_min']),
                            float(values['lag_mult_spin2_max']),
                            float(values['lag_mult_spin2_delta']))
    for mult_num in mults_num:
        for mult_spin2 in mults_spin2:
            mult_num = np.round(mult_num, 2)
            mult_spin2 = np.round(mult_spin2, 2)
            templist = []
            templist.append(('number',
                             int(values['lag_value_num']),
                             mult_num))
            templist.append(('spin-squared',
                             int(values['lag_value_spin2']),
                             mult_spin2))
            lagops_list.append(templist)

def set_ops_num_spinz(values, lagops_list):
    mults_num = np.arange(float(values['lag_mult_num_min']),
                          float(values['lag_mult_num_max']),
                          float(values['lag_mult_num_delta']))
    mults_spinz = np.arange(float(values['lag_mult_spinz_min']),
                            float(values['lag_mult_spinz_max']),
                            float(values['lag_mult_spinz_delta']))
    for mult_num in mults_num:
        for mult_spinz in mults_spinz:
            mult_num = np.round(mult_num, 2)
            mult_spinz = np.round(mult_spinz, 2)
            templist = []
            templist.append(('number',
                             int(values['lag_value_num']),
                             mult_num))
            templist.append(('spin-z',
                             int(values['lag_value_spinz']),
                             mult_spinz))
            lagops_list.append(templist)

def set_ops_spin2_spinz(values, lagops_list):
    mults_spin2 = np.arange(float(values['lag_mult_spin2_min']),
                            float(values['lag_mult_spin2_max']),
                            float(values['lag_mult_spin2_delta']))
    mults_spinz = np.arange(float(values['lag_mult_spinz_min']),
                            float(values['lag_mult_spinz_max']),
                            float(values['lag_mult_spinz_delta']))
    for mult_spin2 in mults_spin2:
        for mult_spinz in mults_spinz:
            mult_spin2 = np.round(mult_spin2, 2)
            mult_spinz = np.round(mult_spinz, 2)
            templist = []
            templist.append(('spin-squared',
                             int(values['lag_value_spin2']),
                             mult_spin2))
            templist.append(('spin-z',
                             int(values['lag_value_spinz']),
                             mult_spinz))
            lagops_list.append(templist)

def set_ops_num_spin2_spinz(values, lagops_list):
    mults_spin2 = np.arange(float(values['lag_mult_spin2_min']),
                            float(values['lag_mult_spin2_max']),
                            float(values['lag_mult_spin2_delta']))
    mults_num = np.arange(float(values['lag_mult_num_min']),
                          float(values['lag_mult_num_max']),
                          float(values['lag_mult_num_delta']))
    mults_spinz = np.arange(float(values['lag_mult_spinz_min']),
                            float(values['lag_mult_spinz_max']),
                            float(values['lag_mult_spinz_delta']))
    for mult_spin2 in mults_spin2:
        for mult_spinz in mults_spinz:
            for mult_num in mults_num:
                mult_spin2 = np.round(mult_spin2, 2)
                mult_spinz = np.round(mult_spinz, 2)
                mult_num = np.round(mult_num, 2)
                templist = []
                templist.append(('spin-squared',
                                 int(values['lag_value_spin2']),
                                 mult_spin2))
                templist.append(('spin-z',
                                 int(values['lag_value_spinz']),
                                 mult_spinz))
                templist.append(('number',
                                 int(values['lag_value_num']),
                                 mult_num))
                lagops_list.append(templist)

def set_ops_num(values, lagops_list, op):
    mults = np.arange(float(values['lag_mult_num_min']),
                      float(values['lag_mult_num_max']),
                      float(values['lag_mult_num_delta']))
    for mult in mults:
        mult = np.round(mult, 2)
        lagops_list.append([(op,
                             int(values['lag_value_num']),
                             mult)])

def set_ops_spin2(values, lagops_list, op):
    mults = np.arange(float(values['lag_mult_spin2_min']),
                      float(values['lag_mult_spin2_max']),
                      float(values['lag_mult_spin2_delta']))
    for mult in mults:
        mult = np.round(mult, 2)
        lagops_list.append([(op,
                             int(values['lag_value_spin2']),
                             mult)])

def set_ops_spinz(values, lagops_list, op):
    mults = np.arange(float(values['lag_mult_spinz_min']),
                      float(values['lag_mult_spinz_max']),
                      float(values['lag_mult_spinz_delta']))
    for mult in mults:
        mult = np.round(mult, 2)
        lagops_list.append([(op,
                             int(values['lag_value_spinz']),
                             mult)])

def set_lagrange_ops(values):
    if values['lagrangiana'] == ['False']:
        lagops_list = [[('dummy', 0, 0)]]
    else:
        lagops_list = []

    for op in values['lag_op']:
        if op == 'number':
            set_ops_num(values, lagops_list, op)

        elif op == 'spin-squared':
            set_ops_spin2(values, lagops_list, op)

        elif op == 'spin-z':
            set_ops_spinz(values, lagops_list, op)

        elif op == 'num+spin2':
            set_ops_num_spin2(values, lagops_list)

        elif op == 'num+spinz':
            set_ops_num_spinz(values, lagops_list)

        elif op == 'spin2+spinz':
            set_ops_spin2_spinz(values, lagops_list)

        elif op == 'num+spin2+spinz':
            set_ops_num_spin2_spinz(values, lagops_list)

    return lagops_list

def print_chose_options(options):
    quantum_instance_name = []
    for ist in options['quantum_instance']:
        quantum_instance_name.append(ist[1])
    optimizers_name = []
    for opt in options['optimizer']:
        optimizers_name.append(opt[1])

    print('OPZIONI SCELTE')
    print('mol_type: ', options['molecule']['molecule'])
    print('charge: ', options['molecule']['charge'])
    print('spin: ', options['molecule']['spin'])
    print('dist: ', options['dists'])
    print('basis: ', options['molecule']['basis'])
    print('varforms: ', options['varforms'])
    print('instances: ', quantum_instance_name)
    print('optimizer: ', optimizers_name)
    if options['lagrange']['active'][0] != 'False':
        print('lagrange: ',
              '\n\tactive: ', options['lagrange']['active'],
              '\n\toperator: ', options['lagrange']['operators'])
        print('series: ',
              '\n\titermax: ', options['series']['itermax'],
              '\n\tstep: ', options['series']['step'],
              '\n\tlamb: ', options['series']['lamb'])

def set_optimizers(values):
    optimizers = []
    for opt in values['optimizer']:
        if opt == 'CG':
            optimizers.append((CG(maxiter=100), 'CG'))
        elif opt == 'COBYLA':
            optimizers.append((COBYLA(), 'COBYLA'))
        elif opt == 'ADAM':
            optimizers.append((ADAM(maxiter=100), 'ADAM'))
        elif opt == 'L_BFGS_B':
            optimizers.append((L_BFGS_B(maxiter=100), 'LBFGSB'))
        elif opt == 'SPSA':
            optimizers.append((SPSA(maxiter=100), 'SPSA'))
        elif opt == 'NFT':
            optimizers.append((NFT(maxiter=60), 'NFT'))
    values['optimizer'] = optimizers

def set_backend_and_noise(values):
    quantum_instance_list = []

    provider = IBMQ.load_account()


    iteratore = list(itertools.product(values['backend'],
                                       values['noise'],
                                       values['correction']))

    for backend, noise, corr in iteratore:

        if backend == 'statevector_simulator':
            quantum_instance = QuantumInstance(Aer.get_backend('statevector_simulator'))
            noise = 'None'
        elif backend == 'qasm_simulator':
            quantum_instance = QuantumInstance(backend=QasmSimulator(), #Aer.get_backend('qasm_simulator'),
                                               shots=int(values['shots']))
        elif backend == 'qasm_gpu':
            quantum_instance = QuantumInstance(backend=Aer.get_backend('qasm_simulator'),
                                               backend_options={'method': 'statevector_gpu'},
                                               shots=int(values['shots']))
        elif backend == 'qasm_simulator_online':
            quantum_instance = QuantumInstance(backend=provider.backend.ibmq_qasm_simulator,
                                               #provider.get_backend('ibmq_qasm_simulator'), #QasmSimulator(method='statevector'),
                                               shots=int(values['shots']))

        elif backend == 'hardware':
            mybackend = provider.get_backend('ibmq_bogota')
            quantum_instance = mybackend

        if backend == 'qasm_simulator' and noise != 'None':
            device = provider.get_backend(noise)
            coupling_map = device.configuration().coupling_map
            noise_model = NoiseModel.from_backend(device)
            quantum_instance.coupling_map = coupling_map
            quantum_instance.noise_model = noise_model


        name = backend + '_' + noise

        if backend == 'qasm_simulator' and noise != 'None' and corr == 'True':
            quantum_instance.measurement_error_mitigation_cls = CompleteMeasFitter
            quantum_instance.measurement_error_mitigation_shots = 1000
            name += '_corrected'

        quantum_instance_list.append((quantum_instance, name))

    values['backend'] = quantum_instance_list

def set_dist_and_geometry(options):
    minimo = float(options['dist']['min'])
    massimo = float(options['dist']['max'])
    delta = float(options['dist']['delta'])

    dist = np.arange(minimo, massimo, delta)
    options['dists'] = dist
    geometries = []

    mol_type = options['molecule']['molecule']
    if mol_type == 'H3+':
        alt = np.sqrt(dist**2 - (dist/2)**2, dtype='float64')
        for i, single_dist in enumerate(dist):
            geom = "H .0 .0 .0; H .0 .0 " + str(single_dist) + "; H .0 " + str(alt[i]) + " " + str(single_dist/2)
            geometries.append(geom)
    elif 'H2O' in mol_type:
        for single_dist in dist:
            alt = single_dist * np.cos(0.25)
            lung = single_dist * np.sin(0.25) + 0.9584
            geom = "H .0 .0 .0; O .0 .0 0.9584; H .0 " + str(alt) + " " + str(lung)
            geometries.append(geom)
    elif 'H2' in mol_type:
        for single_dist in dist:
            geom = "H .0 .0 .0; H .0 .0 " + str(single_dist)
            geometries.append(geom)
    elif 'H4' in mol_type:
        for single_dist in dist:
            geom = "H .0 .0 .0; H .0 .0 " + str(single_dist) + "; H .0 .0 " + str(2*single_dist) + "; H .0 .0 " + str(3*single_dist)
            geometries.append(geom)
    elif 'Li2' in mol_type:
        for single_dist in dist:
            geom = "Li .0 .0 .0; Li .0 .0 " + str(single_dist)
            geometries.append(geom)
    elif 'LiH' in mol_type:
        for single_dist in dist:
            geom = "Li .0 .0 .0; H .0 .0 " + str(single_dist)
            geometries.append(geom)
    options['geometries'] = geometries

    return options

def get_spin(mol_opt):
    mol_type = mol_opt
    if mol_type == 'H3+':
        return 0
    elif mol_type == 'H2':
        return 0
    elif mol_type == 'H2*':
        return 2
    elif mol_type == 'H2+':
        return 1
    elif mol_type == 'H4':
        return 0
    elif mol_type == 'H4*':
        return 2
    elif mol_type == 'Li2':
        return 0
    elif mol_type == 'Li2+':
        return 1
    elif mol_type == 'H2O':
        return 0
    elif mol_type == 'LiH':
        return 0

def get_charge(mol_opt):
    mol_type = mol_opt
    if mol_type == 'H3+':
        return 1
    elif mol_type == 'H2':
        return 0
    elif mol_type == 'H2*':
        return 0
    elif mol_type == 'H2+':
        return 1
    elif mol_type == 'H4':
        return 0
    elif mol_type == 'H4*':
        return 0
    elif mol_type == 'Li2':
        return 0
    elif mol_type == 'Li2+':
        return 1
    elif mol_type == 'LiH':
        return 0
    elif mol_type == 'H2O':
        return 0
