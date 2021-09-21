import json
import sys

import numpy as np
from pprint import pprint


if __name__ == '__main__':
    if len(sys.argv) == 1:
        exit()

    print('Leggo l\'esperimento: ', sys.argv[1])
    file = open(sys.argv[1],)
    data = json.load(file)
    file.close()

    results = data['results_tot']
    x = data['options']['dists']

    print("dist=")
    pprint(x)

    myLegend = []

    for  item in results:
        y = []

        for singleResult in results[item]:
            y.append(singleResult['energy'])

        print(item,"=")
        pprint(y)

