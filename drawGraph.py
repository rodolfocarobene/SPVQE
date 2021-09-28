import json
import sys
import re

import matplotlib.pyplot as plt
import numpy as np

from subroutineMAIN import return_classical_results

def get_results_dists_moltype(file_name):
    file = open(file_name)
    data = json.load(file)
    file.close()

    results = data['results_tot']
    x = data['options']['dists']
    mol_type = data['options']['molecule']['molecule']

    print('DESCRIZIONE: ', data['description'])

    return results, x, mol_type

def from_item_to_label(item):
    name = re.sub(r'sto-6g_', '', item)
    name = re.sub(r'EfficientSU\(2\)_|TwoLocal_|UCCSD_|SO(4)_', '', name)
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_\d+x.+_$', '', name)
    name = re.sub(r'None_', '', name)
    return name

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

        linestyle = 'None'
        markerstyle = 'o'

        if 'statevector_simulator' in item:
            linestyle = 'dashed'
            markerstyle = 'o'
        elif 'qasm_simulator' in item:
            linestyle = 'dashdot'
            markerstyle = 'o'

        axes[0].plot(x, y_energy, label=from_item_to_label(item),
                     linestyle=linestyle, marker=markerstyle)

        for i, type_graph in enumerate(types_graph):
            axes[i+1].plot(x, y_aux[type_graph], label=from_item_to_label(item),
                           linestyle=linestyle, marker=markerstyle)

        myLegend.append(item)

    fig.subplots_adjust(right=0.936, top=0.8)

    if 'no-ref' not in sys.argv:
        axes[0].plot(np.arange(0.3, 3.5, 0.1),
                     return_classical_results(mol_type).fci,
                     label='Full Configuration Interaction (FCI)',
                     linestyle='dotted', marker='None')

        axes[0].plot(np.arange(0.3, 3.5, 0.1),
                     return_classical_results(mol_type).hf,
                     label='Hartree Fock (HF)',
                     linestyle='dotted', marker='None')

    for i, type_graph in enumerate(types_graph):
        if type_graph == 'number':
            axes[i+1].set_ylabel("Particles\nnumber", fontsize='x-large')
        elif type_graph == 'spinz':
            axes[i+1].set_ylabel(r"Spin-z", fontsize='x-large')
        elif type_graph == 'spin2':
            axes[i+1].set_ylabel(r"${S}^2$", fontsize='x-large')

    plt.xlabel(r"$d$ $[\AA]$", fontsize='x-large')
    axes[0].set_ylabel(r"Energy $[E_H]$", fontsize='x-large')

    if 'nolegend' in sys.argv:
        print(myLegend)
    else:
        axes[0].legend(loc='best',
                       ncol=1, fancybox=True, shadow=True)
    if 'title' in sys.argv:
        fig.suptitle(get_title(sys.argv), fontsize=20)

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    fig.tight_layout()
    plt.show()
