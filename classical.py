import numpy as np
from pyscf import gto, scf, ao2mo, mp, cc, fci, tools
import matplotlib.pyplot as plt

def get_geometry(dist):
    alt = np.sqrt(dist**2 - (dist/2)**2, dtype='float64')
    return "H .0 .0 .0; H .0 .0 " + str(dist) + "; H .0 " + str(alt) + " " + str(dist/2)

def get_energy(geometry, charge, spin, basis = '6-31G'):
    mol = gto.M(atom=geometry, charge=charge, spin=spin, basis=basis, symmetry=True, verbose=0)
    mf = scf.RHF(mol)
    Ehf = mf.kernel()
    fci_h3 = fci.FCI(mf)
    e_fci = fci_h3.kernel()[0]
    return e_fci

if __name__ == '__main__':
    dists = np.arange(0.3, 3.5, .1)

    energies = {}
    energies['neutra'] = []
    energies['ione+'] = []
    energies['ione-'] = []
    energies['weird'] = []

    for dist in dists:

        geometry = get_geometry(dist)

        e_fci = get_energy(geometry, 1, 0)
        energies['ione+'].append(e_fci)

        e_fci = get_energy(geometry, 0, 1)
        energies['neutra'].append(e_fci)

        e_fci = get_energy(geometry, -1, 0)
        energies['ione-'].append(e_fci)

        e_fci = get_energy(geometry, 1, 2)
        energies['weird'].append(e_fci)

    for method in energies:
        plt.plot(dists, energies[method], label=method)


    plt.xlabel(r"$d$ $[\AA]$")
    plt.ylabel(r"Energy $[Ha]$")
    plt.legend()
    plt.show()
