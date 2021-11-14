import json
import sys
import re

import matplotlib.pyplot as plt
import numpy as np

ITEMS = []

def get_results_dists_moltype(file_name):
    file = open(file_name)
    data = json.load(file)
    file.close()

    results = data['results_tot']
    x = data['options']['dists']
    mol_type = data['options']['molecule']['molecule']

    print('DESCRIZIONE: ', data['description'])

    return results, x, mol_type

def from_item_to_label(name):
    if name not in ITEMS:
        ITEMS.append(name)

    if "Series" in name:
        mult = 0.5 / float(re.sub(r'.*(\d\.\d*)_$', r'\1', name))
        name = "Series (" + str(int(mult)) + " iters)"
        #name = re.sub(r'.*(\d\.\d*)_$', r'Series (\1)', name)
    else:
        name = re.sub(r'.*\((\d\.\d*)\)$', r'Simple penalty (1 iter)', name)
        #name = re.sub(r'.*\((\d\.\d*)\)$', r'Simple penalty (\1)', name)


    return name

def from_item_to_linestyle(item):

    if 'Series' in item:
        linestyle = '--'
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
    idx = ITEMS.index(item)
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    return colors[idx]

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

    types_graph, sizes = get_types(['number','spin2'])

    plot1 = plt.figure(1)
    plot2 = plt.figure(2)

    axes1 = plot1.subplots(len(types_graph)+1, 1, sharex=True,
                                 gridspec_kw={'height_ratios': sizes})

    axes2 = plot2.subplots(len(types_graph)+1, 1, sharex=True,
                                 gridspec_kw={'height_ratios': sizes})


    axes1 = axes2

    myLegend = []

    for  item in results:
        label=from_item_to_label(item)

        if "Series" in item:
            axes = axes2
        else:
            axes = axes1

        y_energy = []
        y_aux = {}
        y_aux['number'] = []
        y_aux['spin2'] = []

        for singleResult in results[item]:
            y_energy.append(singleResult['energy'])

            if 'number' in types_graph:
                y_aux['number'].append(singleResult['auxiliary']['particles'])
            if 'spin2' in types_graph:
                y_aux['spin2'].append(singleResult['auxiliary']['spin-sq'])

        linestyle = from_item_to_linestyle(item)
        markerstyle, markersize = from_item_to_marker(item)
        color = from_item_to_color(item)

        axes[0].plot(x, y_energy, label=from_item_to_label(item),
                     linestyle=linestyle, marker=markerstyle, markersize=markersize,
                     mew=2, color=color)

        for i, type_graph in enumerate(types_graph):
            if "Series" in item:
                axes = axes2
            else:
                axes = axes1
            axes[i+1].plot(x, y_aux[type_graph], label=from_item_to_label(item),
                           linestyle=linestyle, marker=markerstyle, markersize=markersize,
                           mew=2, color=color)

        myLegend.append(item)

    plot1.subplots_adjust(right=0.936, top=0.8)
    plot2.subplots_adjust(right=0.936, top=0.8)

    for i, type_graph in enumerate(types_graph):
        if type_graph == 'number':
            axes1[i+1].set_ylabel("Particles\nnumber", fontsize='x-large')
            axes2[i+1].set_ylabel("Particles\nnumber", fontsize='x-large')
        elif type_graph == 'spinz':
            axes1[i+1].set_ylabel(r"Spin-z", fontsize='x-large')
            axes2[i+1].set_ylabel(r"Spin-z", fontsize='x-large')
        elif type_graph == 'spin2':
            axes1[i+1].set_ylabel(r"${S}^2$", fontsize='x-large')
            axes2[i+1].set_ylabel(r"${S}^2$", fontsize='x-large')

    plt.xlabel(r"$d$ $[\AA]$", fontsize='x-large')
    axes1[0].set_ylabel(r"Energy $[E_H]$", fontsize='x-large')
    axes2[0].set_ylabel(r"Energy $[E_H]$", fontsize='x-large')

    if 'nolegend' in sys.argv:
        print(myLegend)
    else:
        axes1[0].legend(loc='best',
                       ncol=1, fancybox=True, shadow=True)
        axes2[0].legend(loc='best',
                       ncol=1, fancybox=True, shadow=True)
    if 'title' in sys.argv:
        fig.suptitle(get_title(sys.argv), fontsize=20)

    fci = [-1.20445857725434, -1.1772556560041039, -1.150295892256992, -1.1244398972305527, -1.100268958061616, -1.0781444800233178, -1.0582520399914663, -1.0406387409179958, -1.0252471289480645, -1.0119461452363505, -1.0005584080056462]

    hf = [-1.1577140541119595, -1.1242888776973972, -1.0903826899967253, -1.0568801388727014, -1.0244247396564674, -0.9934786770804, -0.9643624774246218, -0.9372833074829038, -0.9123563277190744, -0.8896216436058548, -0.8690585548963081]

    axes1[0].plot(np.arange(1.4, 2.5, 0.1), fci,
                 label='Full Configuration Interaction (FCI)',
                 zorder=0, color='r',
                 linestyle=(0, (1, 1)), marker='None', linewidth=1.5)

    axes1[0].plot(np.arange(1.4, 2.5, 0.1), hf,
                 label='Hartree Fock (HF)',
                 zorder=0, color='m',
                 linestyle=(0, (1, 1)), marker='None', linewidth=1.5)

    '''
    axes2[0].plot(np.arange(1.4, 2.5, 0.1), fci,
                 label='Full Configuration Interaction (FCI)',
                 zorder=0, color='r',
                 linestyle=(0, (1, 1)), marker='None', linewidth=1.5)

    axes2[0].plot(np.arange(1.4, 2.5, 0.1), hf,
                 label='Hartree Fock (HF)',
                 zorder=0, color='m',
                 linestyle=(0, (1, 1)), marker='None', linewidth=1.5)
    '''

    for  item in results:

        if "Series" in item:
            axes = axes2
        else:
            axes = axes1

        y_energy = []
        y_aux = {}

        for singleResult in results[item]:
            y_energy.append(singleResult['energy'])

        linestyle = from_item_to_linestyle(item)
        markerstyle, markersize = from_item_to_marker(item)
        color = from_item_to_color(item)

        '''
        y = []
        fci = [-1.20445857725434, -1.1772556560041039, -1.150295892256992, -1.1244398972305527, -1.100268958061616, -1.0781444800233178, -1.0582520399914663, -1.0406387409179958, -1.0252471289480645, -1.0119461452363505, -1.0005584080056462]

        for i, el in enumerate(y_energy):
            y.append(el - fci[len(fci) - i-1])

        plt.plot(x,y,
                     label=from_item_to_label(item),
                     linestyle='None', marker=markerstyle, markersize=markersize,
                     mew=2, color=color)
        '''

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plot1.tight_layout()
    plot2.tight_layout()
    plt.show()
