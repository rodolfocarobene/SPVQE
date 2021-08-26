import numpy as np

from subroutineVQE import solveVQE
from subroutineGUI import retriveVQEOptions
from subroutineJSON import retriveJSONOptions, writeJson

if __name__ == '__main__':
    options = retriveVQEOptions()
    energies = {}
    for var in options['varforms']:
        energies[var] = []

    for i, geometry in enumerate(options['geometries']):
        options['molecule']['geometry'] = geometry
        for form in options['varforms']:
            options['var_form_type'] = form

            result_tot = solveVQE(options)
            result = result_tot.total_energies[0]
            energies[form].append(result)
            print("D = ", np.round(options['dists'][i], 2),
                  "\tVar = ", form,
                  "\tE = ", result)


    JsonOptions = retriveJSONOptions(__file__,
                                     options['dists'].tolist(),
                                     energies)

    writeJson(JsonOptions)






