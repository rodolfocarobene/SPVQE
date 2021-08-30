import numpy as np
import sys
import itertools

from subroutineVQE import solveVQE
from subroutineGUI import retriveVQEOptions
from subroutineJSON import retriveJSONOptions, writeJson

def iteratorItemToString(item):
    name = item[0]
    name += '_' + item[1]
    name += '_' + item[2][1]
    name += '_' + item[3][1]
    if item[4] == True:
        name += '_Lag_' + item[5][0]

    return name

if __name__ == '__main__':
    options = retriveVQEOptions(sys.argv)

    iteratore = list(itertools.product(options['molecule']['basis'],
                                       options['varforms'],
                                       options['quantum_instance'],
                                       options['optimizer'],
                                       options['lagrange']['active'],
                                       options['lagrange']['operators']))

    energies = {}
    for item in iteratore:
        name = iteratorItemToString(item)
        energies[name] = []

    for i, geometry in enumerate(options['geometries']):
        options['molecule']['geometry'] = geometry
        print("D = ", np.round(options['dists'][i], 2))
        for item in iteratore:
            name = iteratorItemToString(item)

            option = options.copy()
            option['molecule']['basis'] = item[0]
            option['var_form_type'] = item[1]
            option['quantum_instance'] = item[2][0]
            option['optimizer'] = item[3][0]
            option['lagrange']['active'] = item[4]
            option['lagrange']['operator'] = item[5][0]
            option['lagrange']['value'] = int(item[5][1])
            option['lagrange']['multiplier'] = float(item[5][2])

            result_tot = solveVQE(option)
            result = result_tot.total_energies[0]
            energies[name].append(result)

            print("\t", name, "\tE = ", result)


    JsonOptions = retriveJSONOptions(__file__,
                                     options['dists'].tolist(),
                                     energies)

    writeJson(JsonOptions)






