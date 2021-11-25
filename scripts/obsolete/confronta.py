import pprint
import json
import sys

import numpy as np

from subroutineMAIN import return_classical_results

def import_valori_veri(dists, mol_type):
    all_dists = np.arange(0.3, 3.5, 0.1)
    riferimento_import = []

    vals_import = return_classical_results(mol_type)

    list_tupla = zip(all_dists, vals_import)

    for distanza, energy in list_tupla:
        if np.round(distanza, 2) in [np.round(num, 2) for num in dists]:
            riferimento_import.append(energy)

    return riferimento_import

def valori_da_confrontare(filename):
    file = open(filename)
    data = json.load(file)
    file.close()

    vals_final = {}
    for res in data['results_tot']:
        ax_y = []
        for single_result in data['results_tot'][res]:
            ax_y.append(single_result['energy'])
        vals_final[res] = ax_y

    return vals_final, data['options']['dists'], data['options']['molecule']['molecule']

def sort_dict(dict1):
    newdict = {k: v for k, v in sorted(dict1.items(),
                                       key=lambda item: item[1])}
    newlist = []
    for key in newdict:
        newlist.append((key, newdict[key]))
    return newlist

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('missing args')
        sys.exit()

    vals, dists, mol_type = valori_da_confrontare(sys.argv[1])
    riferimento = import_valori_veri(dists, mol_type)
    differenze = {}
    for item in vals:
        tmp_difference = []
        zip_obj = zip(riferimento, vals[item])
        for riferimento_i, vals_i in zip_obj:
            tmp_difference.append(abs(riferimento_i - vals_i))
        differenze[item] = sum(tmp_difference)

    sortedDict = sort_dict(differenze)
    for a, b in sortedDict:
        print(a, " : ", b)

