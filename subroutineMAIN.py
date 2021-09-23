import copy

import numpy as np

def iterator_item_to_string(item):
    name = item[0]
    name += '_' + item[1]
    name += '_' + item[2][1]
    name += '_' + item[3][1]
    if item[4] == 'True':
        if item[5][0][0] != 'dummy':
            name += '_Lag_'
            for operator in item[5]:
                name += operator[0]+'('+str(np.round(operator[2], 2))+')'
    elif item[4] == 'Series':
        if item[5][0][0] != 'dummy':
            name += '_LagSeries_'
            for operator in item[5]:
                name += operator[0]+'_'+str(item[6])+'x'+str(np.round(item[7], 2))+'_'
    elif item[4] == 'AUGSeries':
        if item[5][0][0] != 'dummy':
            name += '_LagAUGSeries_'
            for operator in item[5]:
                name += operator[0]+'_'+str(item[6])+'x'+str(np.round(item[7], 2))
                name += '_lamb'+str(np.round(item[8], 2))+'_'

    return name

def from_item_iter_to_option(options, item):
    option = copy.deepcopy(options)
    option['molecule']['basis'] = item[0]
    option['var_form_type'] = item[1]
    option['quantum_instance'] = item[2][0]
    option['optimizer'] = item[3][0]

    if item[4] == 'True' or item[4] == 'False':
        option['lagrange']['active'] = item[4]
        option['lagrange']['series'] = False
        option['lagrange']['augmented'] = False
        option['init_point'] = [None]
    elif item[4] == 'Series':
        option['lagrange']['active'] = True
        option['lagrange']['series'] = True
        option['lagrange']['augmented'] = False
    elif item[4] == 'AUGSeries':
        option['lagrange']['active'] = True
        option['lagrange']['series'] = True
        option['lagrange']['augmented'] = True

    option['lagrange']['operators'] = item[5]

    if item[5][0][0] == 'dummy':
        option['lagrange']['active'] = False

    option['series']['itermax'] = item[6]
    option['series']['step'] = item[7]
    option['series']['lamb'] = item[8]

    return option

def return_classical_results(mol_type):
    if mol_type == 'H2*':
        return [0.8746663352974902, 0.28351783858747615,
                -0.07984820956276151, -0.3201959251257661,
                -0.48669833531774864, -0.6055310045065269,
                -0.6919856041885428, -0.7556938311464096,
                -0.8030523503481138, -0.8384549893848645,
                -0.8649961002491231, -0.8849048747158421,
                -0.8998201809754169, -0.9109662272337082,
                -0.9192672637162707, -0.925425076771708,
                -0.9299734637128507, -0.9333180090970401,
                -0.9357660130931245, -0.9375494300790719,
                -0.9388425886811848, -0.9397758952469387,
                -0.9404464113188682, -0.9409259963579195,
                -0.9412675574323823, -0.9415098285613728,
                -0.9416810086568391, -0.9418015147949865,
                -0.9418860522452043, -0.9419451598079606,
                -0.9419863551299065, -0.942014977507915]
    else:
        return [0.7930249944681114, -0.23748138680009934,
                -0.7527805858030638, -1.0251057217763133,
                -1.1699429398436911, -1.2433092266100079,
                -1.2748268385778982, -1.2811879994251272,
                -1.2722647705355448, -1.2541321018359217,
                -1.2306719129030712, -1.2044585772543404,
                -1.1772556560041034, -1.1502958922569917,
                -1.1244398972305523, -1.1002689580616154,
                -1.0781444800233184, -1.0582520399914663,
                -1.0406387409179958, -1.0252471289480645,
                -1.0119461452363505, -1.000558408005646,
                -0.990882909320383, -0.9827125641869094,
                -0.9758466366305577, -0.9700986094113239,
                -0.965300395780803, -0.9613038824951535,
                -0.9579807036602958, -0.9552209634214398,
                -0.9529314255228293, -0.9510335121611919]
