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

    fig = plt.figure(figsize=(8,5))
    ax = plt.subplot(111)

    myLegend = []

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

        ax.plot(x,y,label=item, linestyle=linestyle, marker=markerstyle)
        myLegend.append(item)

    fig.subplots_adjust(right=0.936, top=0.8)

    plt.xlabel(r"$d$ $[\AA]$", fontsize = 'x-large')
    plt.ylabel(r"Energia $[E_H]$", fontsize = 'x-large')

    if len(sys.argv) > 2:
        if sys.argv[2] == 'nolegend':
            print(myLegend)
        else:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
                      ncol = 1, fancybox = True, shadow = True)
    else:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5,1.3),
                  ncol = 1, fancybox = True, shadow = True)



    plt.show()
