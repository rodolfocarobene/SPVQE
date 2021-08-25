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

    energies = data['energies']
    x = data['dist']
    tipo = data['backend']['type']

    if tipo == 'statevector_simulator':
        linestyle = 'dashed'
        markerstyle = 'None'
    elif tipo == 'quasm_simulator':
        linestyle = 'dashdot'
        markerstyle = 'None'
    else:
        linestyle = 'None'
        markerstyle = 'o'


    for (method,y) in energies.items():
        plt.plot(x,y,label=method, linestyle=linestyle, marker=markerstyle)

    plt.xlabel(r"$d$ $[\AA]$", fontsize = 'x-large')
    plt.ylabel(r"Energia $[E_H]$", fontsize = 'x-large')
    plt.legend(fontsize = 'x-large')



    plt.show()
