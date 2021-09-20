import json
import sys

import matplotlib.pyplot as plt
import numpy as np

def return_val_riferimento():
    vals = [0.7930249944681114, -0.23748138680009934,
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
    return vals

if __name__ == '__main__':
    if len(sys.argv) == 1:
        exit()

    print('Leggo l\'esperimento: ', sys.argv[1])
    file = open(sys.argv[1],)
    data = json.load(file)
    file.close()

    results = data['results_tot']
    x = data['options']['dists']

    fig = plt.figure(figsize=(8, 5))
    ax = plt.subplot(111)

    myLegend = []

    for  item in results:
        y = []

        for singleResult in results[item]:
            y.append(singleResult['energy'])

        if 'statevector_simulator' in item:
            linestyle = 'dashed'
            markerstyle = 'o'
        elif 'qasm_simulator' in item:
            linestyle = 'dashdot'
            markerstyle = 'o'
        else:
            linestyle = 'None'
            markerstyle = 'o'

        ax.plot(x, y, label=item, linestyle=linestyle, marker=markerstyle)
        myLegend.append(item)

    fig.subplots_adjust(right=0.936, top=0.8)

    ax.plot(np.arange(0.3,3.5,0.1), return_val_riferimento(),
            label='riferimento', linestyle='dashed', marker='None')

    plt.xlabel(r"$d$ $[\AA]$", fontsize='x-large')
    plt.ylabel(r"Energia $[E_H]$", fontsize='x-large')

    if len(sys.argv) > 2:
        if sys.argv[2] == 'nolegend':
            print(myLegend)
        else:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
                      ncol=1, fancybox=True, shadow=True)
    else:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5,1.3),
                  ncol = 1, fancybox = True, shadow = True)



    plt.show()
