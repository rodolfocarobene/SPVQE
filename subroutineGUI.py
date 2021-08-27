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
from qiskit import Aer
from qiskit import IBMQ

def retriveVQEOptions():
    layout=[[psg.Text('Molecola', size=(20,1),font='Lucida',justification='left')],
            [psg.Text('Spin', font='Lucida', justification='left'), psg.Input(default_text=0,size=(4,10),key='spin'),
             psg.Text('Charge', font='Lucida', justification='left'), psg.Input(default_text=1,size=(4,10),key='charge'),
             psg.Text('Basis', font='Lucida', justification='left'),
             psg.Combo(['sto-3g','sto-6g'],default_value='sto-6g', key='basis')],
            [psg.Text('Scegli forma variazionale',size=(20, 1), font='Lucida',justification='left')],
            [psg.Listbox(values=['TwoLocal', 'SO(4)', 'UCCSD'],default_values=['TwoLocal'],select_mode='extended',key='varforms',size=(10, 5))],
            [psg.Text('Scegli il tipo di backend',size=(30, 1), font='Lucida',justification='left')],
            [psg.Combo(['statevector_simulator','qasm_simulator','hardware'],default_value='statevector_simulator',key='backend'),
             psg.Text('shots',font='Lucida',justification='left'), psg.Input(default_text=1024,size=(4,10),key='shots')],
            [psg.Text('Scegli l\'eventuale rumore',size=(30, 1), font='Lucida',justification='left')],
            [psg.Combo(['None', 'ibmq_santiago'],default_value='None', key='noise')],
            [psg.Text('Optimizer', font='Lucida',justification='left'),
             psg.Combo(['CG','COBYLA'], default_value='CG',key='optimizer')],
            [psg.Text('Distanze', font='Lucida', justification='left')],
            [psg.Text('Min', font='Lucida', justification='left'), psg.Input(default_text=0.3,size=(4,10),key='dist_min'),
             psg.Text('Max', font='Lucida', justification='left'), psg.Input(default_text=3.5,size=(4,10),key='dist_max'),
             psg.Text('Delta', font='Lucida', justification='left'), psg.Input(default_text=0.1,size=(4,10),key='dist_delta')],
            [psg.Text('Lagrangiana',font='Lucida',justification='left'),
             psg.Combo(['True','False'],default_value='False',key='lagrangiana')],
            [psg.Text('Operatore',font='Lucida',justification='left'),
             psg.Combo(['number','spin_squared','spin_z'],default_value='number',key='lag_op'),
             psg.Text('Value',font='Lucida',justification='left'), psg.Input(default_text=2,size=(4,10),key='lag_value'),
             psg.Text('Multiplier',font='Lucida',justification='left'), psg.Input(default_text=0.2,size=(4,10),key='lag_mult')],
            [psg.Button('Inizia calcolo', font=('Times New Roman',12))]]

    win =psg.Window('Definisci opzioni per VQE',layout,resizable=True)
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
