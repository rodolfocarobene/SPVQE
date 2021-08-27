import numpy as np
import PySimpleGUI as psg

import warnings
#warnings.filterwarnings('ignore', category=SystemTimeWarning)
warnings.simplefilter("ignore")

from qiskit.algorithms.optimizers import CG
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit.providers.aer.noise import NoiseModel
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit.utils import QuantumInstance
from qiskit.ignis.mitigation import CompleteMeasFitter
from qiskit import Aer
from qiskit import IBMQ

def getDefaultOpt():
    print('Loading default values')
    values = {
        'spin' : 0,
        'charge' : 1,
        'basis' : 'sto-6g',
        'varforms' : ['TwoLocal'],
        'backend' : 'statevector_simulator',
        'noise' : 'None',
        'shots' : 1024,
        'correction' : 'False',
        'optimizer' : 'CG',
        'dist_min' : 0.3,
        'dist_max' : 3.5,
        'dist_delta' : 0.1,
        'lagrangiana' : 'False',
        'lag_op' : 'number',
        'lag_value' : 2,
        'lag_mult' : 0.2
    }
    return values

def retriveVQEOptions(argv):
    layout=[[psg.Text('Molecola')],
            [psg.Text('Spin'), psg.Input(default_text=0,size=(4,10),key='spin'),
             psg.Text('Charge'), psg.Input(default_text=1,size=(4,10),key='charge'),
             psg.Text('Basis'),
             psg.Combo(['sto-3g','sto-6g'],default_value='sto-6g', key='basis')],
            [psg.Text('Scegli forma variazionale')],
            [psg.Listbox(values=['TwoLocal', 'SO(4)', 'UCCSD'],default_values=['TwoLocal'],select_mode='extended',key='varforms',size=(10, 5))],
            [psg.Text('Scegli il tipo di backend')],
            [psg.Combo(['statevector_simulator','qasm_simulator','hardware'],default_value='statevector_simulator',key='backend'),
             psg.Text('shots'), psg.Input(default_text=1024,size=(4,10),key='shots')],
            [psg.Text('Scegli l\'eventuale rumore'),
             psg.Combo(['None', 'ibmq_santiago'],default_value='None', key='noise')],
            [psg.Text('Correction'),
             psg.Combo(['False','True'], default_value='False',key='correction')],
            [psg.Text('Optimizer'),
             psg.Combo(['CG','COBYLA'], default_value='CG',key='optimizer')],
            [psg.Text('Distanze')],
            [psg.Text('Min'), psg.Input(default_text=0.3,size=(4,10),key='dist_min'),
             psg.Text('Max'), psg.Input(default_text=3.5,size=(4,10),key='dist_max'),
             psg.Text('Delta'), psg.Input(default_text=0.1,size=(4,10),key='dist_delta')],
            [psg.Text('Lagrangiana'),
             psg.Combo(['True','False'],default_value='False',key='lagrangiana')],
            [psg.Text('Operatore'),
             psg.Combo(['number','spin_squared','spin_z'],default_value='number',key='lag_op'),
             psg.Text('Value'), psg.Input(default_text=2,size=(4,10),key='lag_value'),
             psg.Text('Multiplier'), psg.Input(default_text=0.2,size=(4,10),key='lag_mult')],
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

    if values['optimizer'] == 'CG':
        optimizer = CG()
    elif values['optimizer'] == 'COBYLA':
        optimizer = COBYLA()

    if values['lagrangiana'] == 'True':
        lagrangianActivity = True
    elif values['lagrangiana'] == 'False':
        lagrangianActivity = False

    quantum_instance = setBackendAndNoise(values)

    options = {
        'dist' : {
            'min' : values['dist_min'],
            'max' : values['dist_max'],
            'delta' : values['dist_delta']
        },
        'molecule' : {
            'spin' : int(values['spin']),
            'charge' : int(values['charge']),
            'basis' : str(values['basis']),
        },
        'varforms' : values['varforms'],
        'quantum_instance' : quantum_instance,
        'optimizer' : optimizer,
        'converter' : QubitConverter(mapper = ParityMapper(),
                                   two_qubit_reduction = True),
        'lagrange' : {
            'active' : lagrangianActivity,
            'operator' : values['lag_op'],
            'value' : int(values['lag_value']),
            'multiplier': float(values['lag_mult'])
        }
    }

    options = setDistAndGeometry(options)

    return options

def setBackendAndNoise(values):
    provider = IBMQ.load_account()

    if values['backend'] == 'statevector_simulator':
        quantum_instance = QuantumInstance(Aer.get_backend('statevector_simulator'))
    elif values['backend'] == 'qasm_simulator':
        quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'),
                                           shots=int(values['shots']))

    if values['backend'] == 'qasm_simulator' and values['noise'] != 'None':
        device = provider.get_backend(values['noise'])
        coupling_map = device.configuration().coupling_map
        noise_model = NoiseModel.from_backend(device)
        basis_gates = noise_model.basis_gates
        seed = 150
        quantum_instance.seed_simulator = seed
        quantum_instance.seed_transpiler = seed
        quantum_instance.coupling_map = coupling_map
        quantum_instance.noise_model = noise_model

    if values['correction'] == 'True':
        quantum_instance.measurement_error_mitigation_cls = CompleteMeasFitter
        quantum_instance.measurement_error_mitigation_shots = 1000
    return quantum_instance

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
