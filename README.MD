[![Code Quality Score](https://api.codiga.io/project/29093/score/svg)](https://frontend.code-inspector.com/public/project/29093/C-VQE/dashboard)  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Sequence of penalties method to study excited states using VQE

## Abstract

We propose an extension of the Variational Quantum Eigensolver (VQE) that leads to more accurate energy estimations and can be used to study excited states.
The method is based on the introduction of a sequence of increasing penalties in the cost function.
This approach does not require circuit modifications and thus can be applied with no additional depth cost.

Through numerical simulations, we show that we are able to produce variational states with desired physical properties, such as total spin and charge.
We assess its performance both on classical simulators and on currently available quantum devices, calculating the potential energy curves of small molecular systems in different physical configurations.
Finally, we compare our method to the original VQE and to another extension, obtaining a better agreement with exact simulations for both energy and targeted physical quantities.


## Installation and usage
Download the repository and install the requirements (better to do so in a python [virtual environment](https://docs.python.org/3/library/venv.html).
```console
foo@bar:~$ python -m venv spvqe && cd spvqe
foo@bar:~$ source bin/activate
foo@bar:~$ git clone https://github.com/rodolfocarobene/SPVQE.git && cd SPVQE
foo@bar:~$ pip install .  # eventually, for developer mode, `pip install -e .`
foo@bar:~$ python script/main.py  # execute main program
```
## GUI

The main program controls the simulation using a graphical interface.

<img src="https://github.com/rodolfocarobene/C-VQE/blob/master/images/GUI.png"  width="300" height="450">

It's possible to chose different options at the same time, the program will automatically simulate the current system with all the different options permutations. The options include:
* Molecule to simulate
* Basis to use in simulation
* Variational Ansatz
* Backend
* Number of shots
* Noise model
* Error correction mechanism
* Optimizer
* Bond distances to simulate
* Addition of penalty terms or series of penalties
* Details on the penalty terms to add, constrained operator, steps

Each option will be given a unique 'name'.
It's possible to bypass the GUI by calling the main program with `python script/main.py fast`: the default values (written in src/spvqe/gui.py) will be loaded.

## Files

    . project folder
    ├── src/spvqe             # Source files (alternatively `lib` or `app`)
    ├── scripts
    | ├── main                # Main script to launch gui and simulations
    | └── graphs              # Routines to draw results of simulation
    ├── notebooks             # Some Jupyter-Notebooks to use as examples
    ├── data
    ├── LICENSE
    ├── README.md
    ├── requirements.txt
    └── setup.py

## Citation policy

If you found SPVQE useful, please consider citing it in your publications, we appreciate it greatly:
* [arxiv link](https://arxiv.org/abs/2304.05262)
* [journal link](https://iopscience.iop.org/article/10.1088/2058-9565/acd1a9)

R. Carobene, S. Barison, A. Giachero,  Sequence of penalties method to study excited states using VQE, 2023 Quantum Sci. Technol. 8 035014,  https://dx.doi.org/10.1088/2058-9565/acd1a9 , arXiv:2304.05262 [quant-ph]

```
@article{spvqe,
  author    = "Rodolfo Carobene and Stefano Barison and Andrea Giachero",
  title     = "Sequence of penalties method to study excited states using VQE",
  journal   = "Quantum Science and Technology",
  year      = 2023,
  doi       = {10.1088/2058-9565/acd1a9},
  url       = {https://doi.org/10.1088/2058-9565/acd1a9},
  publisher = {{IOP} Publishing},
  volume    = {8},
  number    = {3},
  pages     = {035014},
}
```
