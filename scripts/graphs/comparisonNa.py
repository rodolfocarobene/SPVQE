import json
import sys
import re

import matplotlib.pyplot as plt
import numpy as np

ITEMS = []

fci = [ -160.57325765896988 ]
hf = [ -160.49567937064683 ]

distances = np.arange(1, 2, 11)

full_mult = 1

def calc_diff_e(y_energy):
    diff = 0
    for idx, f in enumerate(fci):
        if idx != 9:
            diff += pow(abs(f - y_energy[idx]), 2)
    diff = np.sqrt(diff / (len(fci)))
    return diff

def calc_diff_s(y_energy):
    diff = 0
    for idx, f in enumerate(y_energy):
        for k in f:
            if idx != 9:
                diff += pow(abs(k),2)

    diff = np.sqrt(diff / (len(fci)))
    return diff


def get_results_dists_moltype(file_name):
    file = open(file_name)
    data = json.load(file)
    file.close()

    results = data['results_tot']
    x = data['options']['dists']
    mol_type = data['options']['molecule']['molecule']

    print('DESCRIZIONE: ', data['description'])

    return results, x, mol_type

def get_second_graph(diff_E, diff_S):

    mults = []

    for item in diff_E:
        mult = from_item_to_mult(item)
        mults.append(mult)

    y_e = [diff_E[x] for _, x in sorted(zip(mults, diff_E))]
    y_s = [diff_S[x] for _, x in sorted(zip(mults, diff_S))]
    mults.sort()

    return mults, y_e, y_s

def from_item_to_mult(name):
    if "Series" in name:
        mult = np.rint(full_mult / float(re.sub(r'.*(\d\.\d*)_$', r'\1', name)))
    else:
        mult = 1
    return mult

def from_item_to_label(name):
    if name not in ITEMS:
        ITEMS.append(name)

    if "Series" in name:
        mult = np.rint(full_mult / float(re.sub(r'.*(\d\.\d*)_$', r'\1', name)))
        name = "Series (" + str(int(mult)) + " iters)"
    else:
        name = re.sub(r'.*\((\d\.\d*)\)$', r'Simple penalty (1 iter)', name)

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
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13']
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

    axes2 = plot2.subplots()

    myLegend = []

    diff_E = {}
    diff_S = {}

    for  item in results:
        diff_E[item] = []
        diff_S[item] = []

        label=from_item_to_label(item)

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

        axes1[0].plot(x, y_energy, label=from_item_to_label(item),
                     linestyle=linestyle, marker=markerstyle, markersize=markersize,
                     mew=2, color=color)

        for i, type_graph in enumerate(types_graph):
            axes1[i+1].plot(x, y_aux[type_graph], label=from_item_to_label(item),
                           linestyle=linestyle, marker=markerstyle, markersize=markersize,
                           mew=2, color=color)

        myLegend.append(item)

        #second graph

        diff_E[item] = calc_diff_e(y_energy)
        diff_S[item] = calc_diff_s(y_aux['spin2'])
        print(item, ' ', diff_E[item], ' ', diff_S[item])

    axes2.set_xlabel('Step number', fontsize=35)
    axes2.set_ylabel('Standard deviation from FCI energy', fontsize=35, labelpad=15)
    plt.grid(axis='both', linestyle='-.')
    #plt.title('Decreasing errors with step number', fontsize='x-large')
    axes_s = axes2.twinx()
    axes_s.set_ylabel('Standard deviation from spin value', fontsize=35, labelpad=15)

    x, y_e, y_s = get_second_graph(diff_E, diff_S)

    axes2.plot(x, y_e, color='red', label='energy', markersize=15, marker='o', linewidth=1.5)
    axes2.yaxis.label.set_color('red')
    axes2.tick_params(axis='y', colors='red', labelsize=20)
    axes2.tick_params(axis='x', labelsize=20)

    axes_s.plot(x, y_s, color='green', label='spin2', markersize=15, marker='o', linewidth=1.5)
    axes_s.yaxis.label.set_color('green')
    axes_s.tick_params(axis='y', colors='green', labelsize=20)

    axes_s.spines['left'].set_linewidth(2)
    axes_s.spines['right'].set_linewidth(2)
    axes_s.spines['left'].set_color('red')
    axes_s.spines['right'].set_color('green')
    axes_s.spines['top'].set_linewidth(2)
    axes_s.spines['bottom'].set_linewidth(2)

    axes2.spines['left'].set_linewidth(2)
    axes2.spines['right'].set_linewidth(2)
    axes2.spines['left'].set_color('red')
    axes2.spines['right'].set_color('green')
    axes2.spines['top'].set_linewidth(2)
    axes2.spines['bottom'].set_linewidth(2)

    plot1.subplots_adjust(right=0.936, top=0.8)
    plot2.subplots_adjust(right=0.936, top=0.8)

    for i, type_graph in enumerate(types_graph):
        if type_graph == 'number':
            axes1[i+1].set_ylabel("Particles\nnumber", fontsize='x-large')
        elif type_graph == 'spinz':
            axes1[i+1].set_ylabel(r"Spin-z", fontsize='x-large')
        elif type_graph == 'spin2':
            axes1[i+1].set_ylabel(r"${S}^2$", fontsize='x-large')

    plt.xlabel(r"$d$ $[\AA]$", fontsize='x-large')
    axes1[0].set_ylabel(r"Energy $[E_H]$", fontsize='x-large')

    if 'nolegend' in sys.argv:
        print(myLegend)
    else:
        axes1[0].legend(loc='best',
                       ncol=1, fancybox=True, shadow=True)
    if 'title' in sys.argv:
        fig.suptitle(get_title(sys.argv), fontsize=20)


    axes1[0].plot(distances, fci,
                 label='Full Configuration Interaction (FCI)',
                 zorder=0, color='r',
                 linestyle=(0, (1, 1)), marker='None', linewidth=1.5)

    axes1[0].plot(distances, hf,
                 label='Hartree Fock (HF)',
                 zorder=0, color='m',
                 linestyle=(0, (1, 1)), marker='None', linewidth=1.5)

    for  item in results:

        y_energy = []
        y_aux = {}

        for singleResult in results[item]:
            y_energy.append(singleResult['energy'])

        linestyle = from_item_to_linestyle(item)
        markerstyle, markersize = from_item_to_marker(item)
        color = from_item_to_color(item)

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
