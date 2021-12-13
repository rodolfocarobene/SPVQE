import sys
import json
import itertools

import numpy as np

from spvqe.vqes import solve_VQE
from spvqe.gui import retrive_VQE_options
from spvqe.json_handler import retrive_json_options, write_json
from spvqe.iterator import iterator_item_to_string, from_item_iter_to_option

if __name__ == '__main__':
    options = retrive_VQE_options(sys.argv)

    iteratore = list(itertools.product(options['molecule']['basis'],
                                       options['varforms'],
                                       options['quantum_instance'],
                                       options['optimizer'],
                                       options['lagrange']['active'],
                                       options['lagrange']['operators'],
                                       options['series']['itermax'],
                                       options['series']['step'],
                                       options['series']['lamb']
                                       )
                     )

    results = {}
    parameters = {}
    energies = {}
    numbers = {}
    spin = {}
    names = []
    newiteratore = []

    for idx, item in enumerate(iteratore):
        name = iterator_item_to_string(item)
        if name not in names:
            names.append(name)
            results[name] = []
            energies[name] = []
            numbers[name] = []
            spin[name] = []

            for k, geometry in enumerate(options['geometries']):
                results[name].append([])
                energies[name].append([])
                numbers[name].append([])
                spin[name].append([])

            parameters[name] = []
            newiteratore.append(item)
    iteratore = newiteratore


    for k, geometry in enumerate(options['geometries']):
        options['molecule']['geometry'] = geometry
        print("D = ", np.round(options['dists'][k], 2))
        for item in iteratore:
            name = iterator_item_to_string(item)
            option = from_item_iter_to_option(options, item)

            for i in range(20):

                result_tot, parameters[name] = solve_VQE(option)
                results[name][k].append(result_tot)

                energy = result_tot.total_energies[0]
                spin2 = result_tot.total_angular_momentum[0]
                number = result_tot.num_particles[0]

                energies[name][k].append(energy)
                numbers[name][k].append(number)
                spin[name][k].append(spin2)

                print(i, "\t", name, "  E = ", energy, '  S2 = ', spin2, '  N = ', number)


    print('Energies=', energies)
    print('S2=', spin)
    print('N=', numbers)

    with open("starting-points.txt", "w") as f:
        f.write(json.dumps(energies))
        f.write(json.dumps(numbers))
        f.write(json.dumps(spin))
    f.close()
