import json
import sys
import matplotlib.pyplot as plt

if __name__ == '__main__':
    if len(sys.argv) == 1:
        exit()

    print('Leggo l\'esperimento: ', sys.argv[1])
    file = open(sys.argv[1],)
    data = json.load(file)
    file.close()

    results = data['results_tot']
    x = data['options']['dists']

    for  item in results:
        y = []
        for singleResult in results[item]:
            y.append(singleResult['energy'])

        if 'statevector_simulator' in item:
            linestyle = 'dashed'
            markerstyle = 'None'
        elif 'qasm_simulator' in item:
            linestyle = 'dashdot'
            markerstyle = 'None'
        else:
            linestyle = 'None'
            markerstyle = 'o'


        plt.plot(x,y,label=item, linestyle=linestyle, marker=markerstyle)

    plt.xlabel(r"$d$ $[\AA]$", fontsize = 'x-large')
    plt.ylabel(r"Energia $[E_H]$", fontsize = 'x-large')
    plt.legend(fontsize = 'x-large',loc='center left', bbox_to_anchor=(1,0.5))



    plt.show()
