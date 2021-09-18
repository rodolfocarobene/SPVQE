import pprint
import json
import sys

import numpy as np

def importValoriVeri(dists):
    all_dists = np.arange(0.3, 3.5, 0.1)
    riferimento_import = []

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

def valoriDaConfrontare(filename):
    file = open(filename)
    data = json.load(file)
    file.close()

    vals = {}
    for item in data['results_tot']:
        y = []
        for singleResult in data['results_tot'][item]:
            y.append(singleResult['energy'])
        vals[item] = y

    return vals, data['options']['dists']

def sortDict(dict1):
    newdict = {k: v for k, v in sorted(dict1.items(),
                                       key=lambda item: item[1])}
    newlist = []
    for item in newdict:
        newlist.append((item, newdict[item]))
    return newlist

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('missing args')
        exit()

    vals, dists = valoriDaConfrontare(sys.argv[1])
    riferimento = importValoriVeri(dists)
    differenze = {}
    for item in vals:
        tmp_difference = []
        zip_obj = zip(riferimento, vals[item])
        for riferimento_i, vals_i in zip_obj:
            tmp_difference.append(abs(riferimento_i - vals_i))
        differenze[item] = sum(tmp_difference)

    sortedDict = sortDict(differenze)
    for a,b in sortedDict:
        print(a," : ", b)

