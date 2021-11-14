import sys
import itertools

import numpy as np
import copy

from subroutineVQE import solve_VQE
from subroutineGUI import retrive_VQE_options
from subroutineJSON import retrive_json_options, write_json
from subroutineMAIN import iterator_item_to_string, from_item_iter_to_option

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

    name += '_spin_' + str(item[9])

    return name

def from_item_iter_to_option(options, item):
    option = copy.deepcopy(options)
    option['molecule']['basis'] = item[0]
    option['molecule']['spin'] = item[9]
    option['var_form_type'] = item[1]
    option['quantum_instance'] = item[2][0]
    option['optimizer'] = item[3][0]

    if item[4] == 'True' or item[4] == 'False':
        if item[4] == 'True':
            lag_bool = True
        else:
            lag_bool = False
        option['lagrange']['active'] = lag_bool
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

    newop = []
    for op in option['lagrange']['operators']:
        if op[0] == 'spin-squared':
            op = (op[0], item[9], op[2])
            newop.append(op)
    option['lagrange']['operators'] = newop

    if item[5][0][0] == 'dummy':
        option['lagrange']['active'] = False

    option['series']['itermax'] = item[6]
    option['series']['step'] = item[7]
    option['series']['lamb'] = item[8]

    return option
if __name__ == '__main__':
    options = retrive_VQE_options(sys.argv)

    spins = [0, 2, 4]

    iteratore = list(itertools.product(options['molecule']['basis'],
                                       options['varforms'],
                                       options['quantum_instance'],
                                       options['optimizer'],
                                       options['lagrange']['active'],
                                       options['lagrange']['operators'],
                                       options['series']['itermax'],
                                       options['series']['step'],
                                       options['series']['lamb'],
                                       spins
                                       )
                     )

    results = {}
    parameters = {}
    names = []
    newiteratore = []

    for idx, item in enumerate(iteratore):
        name = iterator_item_to_string(item)
        if name not in names:
            names.append(name)
            results[name] = []
            parameters[name] = []
            newiteratore.append(item)
    iteratore = newiteratore


    for i, geometry in enumerate(options['geometries']):
        options['molecule']['geometry'] = geometry
        print("D = ", np.round(options['dists'][i], 2))
        for item in iteratore:
            name = iterator_item_to_string(item)
            option = from_item_iter_to_option(options, item)

            if len(parameters[name]) > 0:
                option['init_point'] = parameters[name]

            result_tot, parameters[name] = solve_VQE(option)
            results[name].append(result_tot)

            energy = result_tot.total_energies[0]
            spin2 = result_tot.total_angular_momentum[0]
            number = result_tot.num_particles[0]

            print("\t", name, "\tE = ", energy, '\tS2 = ', spin2, '\tN = ', number)

        #STO BARANDO ?!?
       #for kind in parameters:
       #       if len(parameters[kind]) == 0:
       #           parameters[kind] = parameters[name]
        #--------------------

    JsonOptions = retrive_json_options(__file__,
                                     options,
                                     results)

    write_json(JsonOptions)
