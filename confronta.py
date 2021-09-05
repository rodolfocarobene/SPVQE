import pprint
import json
import sys

def importValoriVeri():
    vals = []
    file = open('data/exp_02_09_09_40',)
    data = json.load(file)
    file.close()

    for singleResult in data['results_tot']['sto-6g_UCCSD_statevector_simulator_None_CG']:
        vals.append(singleResult['energy'])

    return vals

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
    return vals

def sortDict(dict1):
    sorted_val = sorted(dict1.values())
    dict2 = []
    for i in sorted_val:
        for k in dict1.keys():
            if dict1[k] == i:
                dict2.append((k,dict1[k]))
                break
    return dict2

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('missing args')
        exit()

    riferimento = importValoriVeri()
    vals = valoriDaConfrontare(sys.argv[1])

    differenze = {}
    for item in vals:
        tmp_difference = []
        zip_obj = zip(riferimento, vals[item])
        for riferimento_i, vals_i in zip_obj:
            tmp_difference.append(abs(riferimento_i - vals_i))
        differenze[item] = sum(tmp_difference)

    sortedDict = sortDict(differenze)
    for a,b in sortedDict:
        print(a,' : ',b)
    #pprint.pprint(differenze)

