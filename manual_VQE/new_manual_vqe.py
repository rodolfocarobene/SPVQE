import ....

def ei(i, n):
    vi = np.zeros(n)
    vi[i] = 1.0
    return vi[:]

def funzioneprincipaledire():

    driver = PySCFDriver(...)
    molecule = driver.run()

    core = Hamiltonian(...)

    H_op, A_op = core.run(molecule)
    dE = core._energy_shift + core._ph_energy_shift + core._nuclear_repulsion_energy
    A_op = A_op[:3]
    dA = [0]*len(A_op)

    init_state = HartreeFock(...)
    num_qubits = H_op.num_qubits
    num_parameters = 6*(num_qubits-1)

    ansatz = createSO4Ansatz(...)
    quantum_instance = createQuantumInstance(...)

    algo = createVQE(...)
    algo.run(...)


class manualVQE:
    def __init__(...):
        .
        .
        .
    def readFromFile(...):
        .
        .
        .
    def measure(...):
        .
        .
        .
    def measureAncillas(...):
        .
        .
        .
    def evaluateEnergyGradient(...):
        .
        .
        .
    def performLineSearch(...):
        .
        .
        .
    def dumpOnFile(...):
        .
        .
        .
    def run(...):
        .
        .
        .
