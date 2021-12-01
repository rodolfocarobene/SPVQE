import json
import sys
import re

import matplotlib.pyplot as plt
import numpy as np
sys.path.append('/home/rodolfo/qiskit_nature_0.2.0/main/C-VQE/')
from spvqe.classical_values import return_classical_results

def get_results_dists_moltype(file_name):
    file = open(file_name, encoding='UTF-8')
    data = json.load(file)
    file.close()

    results = data['results_tot']
    x = data['options']['dists']
    mol_type = data['options']['molecule']['molecule']

    print('DESCRIZIONE: ', data['description'])

    return results, x, mol_type

def from_item_to_label(item):
    '''
    name = re.sub(r'sto-6g_', '', item)
    name = re.sub(r'EfficientSU\(2\)_|TwoLocal_|UCCSD_|SO(4)_', '', name)
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_\d+x.+_$', '', name)
    name = re.sub(r'None_', '', name)
    '''

    if 'Series' in item:
        name = 'Series of penalties'
    elif 'Lag' in item:
        name = 'Simple penalty'
    else:
        name = 'Standard VQE'

    return name

def from_item_to_linestyle(item):
    '''
    if 'statevector_simulator' in item:
        linestyle = 'dashed'
    elif 'qasm_simulator' in item:
        linestyle = 'dashdot'
    else:
        linestyle = None
    '''

    if 'Series' in item:
        linestyle = '-'
    elif 'Lag' in item:
        linestyle = '--'
    else:
        linestyle = '-'

    return linestyle

def from_item_to_marker(item):
    if 'Series' in item:
        markerstyle = 'x'
        size = 10
    elif 'Lag' in item:
        markerstyle = '+'
        size = 10
    else:
        markerstyle = 'o'
        size = 5

    return markerstyle, size

def from_item_to_color(item):
    if 'Series' in item:
        color = 'C2'
    elif 'Lag' in item:
        color = 'C0'
    else:
        color = 'C1'

    return color

def get_title(argv):
    index_title = argv.index('title')

    if len(argv) > index_title+1:
        return argv[index_title+1]
    else:
        return "none"

def get_types(argv):
    tmp_types = []
    sizes = [4]
    if 'number' in argv:
        tmp_types.append('number')
        sizes.append(1)
    if 'spin2' in argv:
        tmp_types.append('spin2')
        sizes.append(1)
    if 'spinz' in argv:
        tmp_types.append('spinz')
        sizes.append(1)
    if len(tmp_types) == 0:
        tmp_types.append('number')
        sizes.append(1)

    return tmp_types, sizes

if __name__ == '__main__':
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '-.'
    if len(sys.argv) == 1:
        sys.exit()

    results, x, mol_type = get_results_dists_moltype(sys.argv[1])

    types_graph, sizes = get_types(sys.argv)

    fig, axes = plt.subplots(len(types_graph)+1, 1, sharex=True,
                             gridspec_kw={'height_ratios': sizes})

    myLegend = []

    for  item in results:
        y_energy = []
        y_aux = {}
        y_aux['number'] = []
        y_aux['spinz'] = []
        y_aux['spin2'] = []

        for singleResult in results[item]:
            y_energy.append(singleResult['energy'])

            if 'number' in types_graph:
                y_aux['number'].append(singleResult['auxiliary']['particles'])
            if 'spinz' in types_graph:
                y_aux['spinz'].append(singleResult['auxiliary']['spin-z'])
            if 'spin2' in types_graph:
                y_aux['spin2'].append(singleResult['auxiliary']['spin-sq'])

        linestyle = from_item_to_linestyle(item)
        markerstyle, markersize = from_item_to_marker(item)
        color = from_item_to_color(item)

        axes[0].plot(x, y_energy, label=from_item_to_label(item),
                     linestyle=linestyle, marker=markerstyle, markersize=markersize,
                     mew=2, color=color, linewidth=0.5)

        for i, type_graph in enumerate(types_graph):
            axes[i+1].plot(x, y_aux[type_graph], label=from_item_to_label(item),
                           linestyle=linestyle, marker=markerstyle, markersize=markersize,
                           mew=2, color=color, linewidth=0.5)

        myLegend.append(item)

    fig.subplots_adjust(right=0.936, top=0.8)

    for i, type_graph in enumerate(types_graph):
        if type_graph == 'number':
            axes[i+1].set_ylabel("Particles\nnumber", fontsize='x-large')
        elif type_graph == 'spinz':
            axes[i+1].set_ylabel(r"Spin-z", fontsize='x-large')
        elif type_graph == 'spin2':
            axes[i+1].set_ylabel(r"${S}^2$", fontsize='x-large')

    fig.suptitle('LiH statevector simulation for different interatomic distances', fontsize='x-large')
    plt.xlabel(r"$d$ $[\AA]$", fontsize='large')
    axes[0].set_ylabel(r"Energy $[E_H]$", fontsize='large')

    plt.xticks(np.arange(0.1, 6, 0.2))
    axes[0].set_yticks(np.arange(-9, -6 , 0.1))
    axes[1].set_yticks(np.arange(2, 2.002 , 0.0005))
    axes[2].set_yticks(np.arange(0, 1 , 0.2))

    if 'nolegend' in sys.argv:
        print(myLegend)
    else:
        axes[0].legend(loc='best',
                       ncol=1, fancybox=True, shadow=True)
    if 'title' in sys.argv:
        fig.suptitle(get_title(sys.argv), fontsize=20)

    if 'no-ref' not in sys.argv:
        axes[0].plot(np.arange(0.3, 5, 0.2),
                     return_classical_results(mol_type).fci,
                     label='Full Configuration Interaction (FCI)',
                     zorder=0, color='r',
                     linestyle=':', marker='None', linewidth=1.5)

        axes[0].plot(np.arange(0.3, 5, 0.2),
                     return_classical_results(mol_type).hf,
                     label='Hartree Fock (HF)',
                     zorder=0, color='m',
                     linestyle=':', marker='None', linewidth=1.5)

        plot2 = plt.figure(2)
        for  item in results:
            y_energy = []
            y_aux = {}

            for singleResult in results[item]:
                y_energy.append(singleResult['energy'])

            linestyle = from_item_to_linestyle(item)
            markerstyle, markersize = from_item_to_marker(item)
            color = from_item_to_color(item)

            y = []
            fci = return_classical_results(mol_type).fci
            for i, el in enumerate(y_energy):
                y.append(el - fci[len(fci) - i-1])

            plt.plot(x,y,
                         label=from_item_to_label(item),
                         linestyle='None', marker=markerstyle, markersize=markersize,
                         mew=2, color=color)


    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    fig.tight_layout()
    plt.show()
