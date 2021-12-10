import json
import sys

from pprint import pprint

import numpy as np

if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.exit()

    print('Leggo l\'esperimento: ', sys.argv[1])
    file = open(sys.argv[1], encoding='utf-8')
    data = json.load(file)
    file.close()

    results = data['results_tot']
    x = data['options']['dists']

    if len(sys.argv) == 2:
        sys.argv.append('energy')

    print("dist=")
    pprint(x)

    myLegend = []

    for  item in results:
        y = []

        for single_result in results[item]:
            if sys.argv[2] == 'energy':
                y.append(single_result['energy'])
            elif sys.argv[2] == 'number':
                y.append(single_result['auxiliary']['particles'][0])
            elif sys.argv[2] == 'spin-z':
                y.append(single_result['auxiliary']['spin-z'][0])
            elif sys.argv[2] == 'spin2':
                y.append(single_result['auxiliary']['spin-sq'][0])

        print(item,"=")
        pprint(y)
