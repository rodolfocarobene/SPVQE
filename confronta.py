import pprint
import json
import sys

import numpy as np

def import_valori_veri(dists, mol_type):
    all_dists = np.arange(0.3, 3.5, 0.1)
    riferimento_import = []

    if mol_type == 'H2*':
        vals_import = [0.8746663352974902, 0.28351783858747615,
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
        vals_import = [0.7930249944681114, -0.23748138680009934,
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

    list_tupla = zip(all_dists, vals_import)

    for dist, energy in list_tupla:
        if np.round(dist, 2) in [np.round(num, 2) for num in dists]:
            riferimento_import.append(energy)

    return riferimento_import

def valori_da_confrontare(filename):
    file = open(filename)
    data = json.load(file)
    file.close()

    vals = {}
    for res in data['results_tot']:
        y = []
        for singleResult in data['results_tot'][res]:
            y.append(singleResult['energy'])
        vals[res] = y

    return vals, data['options']['dists'], data['options']['molecule']['molecule']

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
        exit()

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

