import sys
import itertools

import numpy as np

from subroutineVQE import solveVQE
from subroutineGUI import retriveVQEOptions
from subroutineJSON import retriveJSONOptions, writeJson
from subroutineMAIN import iteratorItemToString, fromItemIterToOption

if __name__ == '__main__':
    options = retriveVQEOptions(sys.argv)

    iteratore = list(itertools.product(options['molecule']['basis'],
                                       options['varforms'],
                                       options['quantum_instance'],
                                       options['optimizer'],
                                       options['lagrange']['active'],
                                       options['lagrange']['operators'],
                                       options['series']['itermax'],
                                       options['series']['step'],
                                       options['series']['lamb']
                                       )
                     )

    results = {}
    names = []
    newiteratore = []

    for idx, item in enumerate(iteratore):
        name = iteratorItemToString(item)
        if name not in names:
            names.append(name)
            results[name] = []
            newiteratore.append(item)
    iteratore = newiteratore

    for i, geometry in enumerate(options['geometries']):
        options['molecule']['geometry'] = geometry
        print("D = ", np.round(options['dists'][i], 2))
        for item in iteratore:
            name = iteratorItemToString(item)
            option = fromItemIterToOption(options, item)

            result_tot = solveVQE(option)
            results[name].append(result_tot)

            energy = result_tot.total_energies[0]
            print("\t", name, "\tE = ", energy)

    JsonOptions = retriveJSONOptions(__file__,
                                     options,
                                     results)

    writeJson(JsonOptions)
