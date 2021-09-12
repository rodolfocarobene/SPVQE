import numpy as np
import copy

def iteratorItemToString(item):
    name = item[0]
    name += '_' + item[1]
    name += '_' + item[2][1]
    name += '_' + item[3][1]
    if item[4] == 'True':
        if item[5][0][0] != 'dummy':
            name += '_Lag_'
            for operator in item[5]:
                name += operator[0]+'('+str(np.round(operator[2],2))+')'
    elif item[4] == 'Series':
        if item[5][0][0] != 'dummy':
            name += '_LagSeries_'
            for operator in item[5]:
                name += operator[0]+'('+str(np.round(operator[2],2))+')'

    return name

def fromItemIterToOption(options, item):
    option = copy.deepcopy(options)
    option['molecule']['basis'] = item[0]
    option['var_form_type'] = item[1]
    option['quantum_instance'] = item[2][0]
    option['optimizer'] = item[3][0]

    if item[4] == 'True' or item[4] == 'False':
        option['lagrange']['active'] = item[4]
        option['lagrange']['series'] = False
        option['init_point'] = [None]
    else:
        option['lagrange']['active'] = True
        option['lagrange']['series'] = True

    option['lagrange']['operators'] = item[5]

    if item[5][0][0] == 'dummy':
        option['lagrange']['active'] = False

    return option


