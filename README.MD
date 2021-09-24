[![Code Quality Score](https://www.code-inspector.com/project/29093/score/svg)](https://frontend.code-inspector.com/public/project/29093/C-VQE/dashboard)  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Constrained Variational Quantum Eigensolver

## Description

This repository contains all the code written while looking for a better constrained VQE implementation

## Usage

* Download the repository;
* Meet the requirements: the best way to do so is by [creating a python virtual environment](https://docs.python.org/3/library/venv.html). After sourcing it, you can use `pip install -r requirements.txt`;
* Create logs directory: `mkdir logs`;
* Run test.py: `python test.py` (follow section GUI for more info);
* Write a simulation description: at the end of the simulation all the results are stored in a special JSON file that ends with this description. The JSON file also write the current git commit 'name', remove that option in subroutineJSON.py if you are not in a git repository. 
* Draw your result: simply with `python drawGraph.py data/exp...`

## GUI

The main program controls the simulation using a graphical interface. 
![GUI](https://github.com/rodolfocarobene/C-VQE/blob/master/images/GUI.png)
It's possible to chose different options at the same time, the program will automatically simulate the current system with all the different options permutations. 
Each option will be given a unique 'name'.
It's possible to bypass the GUI by calling the main program with `python test.py fast`: the default values (written in subroutineGUI.py) are loaded.

## Files

* __classical.py__ : exact SCF simulation for different H3 configuration (H3, H3+, H3-, H3+*)
* __confronta.py__ : takes a data file as input (Es: python confronta.py data/exp_21_10_10_54). Returns a list of all methods whose results are stored in the data file, ordered by the total squared distance from classical obtained values.
* __drawGraph.py__ : takes a data file as input. Draws a mpl graph with all stored results + classical values. Can take another argoument "nolegend" (Es: python drawGraph.py data/exp_21_21_21_21 nolegend) to instruct the script to not draw the legend (in large data files can cover the actual graph)
* __example.py__ : just a simple standalone VQE implementation with qiskit_nature 0.2.0.
* __requirements.txt__ : every installed package.
* __subroutineVQE.py__ : contains all the functions relevant to a VQE implementation (or variants).
* __subroutineGUI.py__ : contains all the functions relevant to the GUI.
* __subroutineMAIN.py__ : contains all the functions relevant to the main program (test.py). It's needed to pass from subroutineGUI to subroutineVQE.
* __subroutineJSON.py__ : contains all the functions relevant to the Json dumping process (data files are json files).
* __test.py__ : the main program.
* __README.MD__ : ths file. Hi loop!
* __data/__ : data folder.

## Changes to downloaded libs

* /lib/python3.8/site-packages/pyscf/lib/misc.py -> raw 46 -> default mode 'r'
* some changes (I don't remember where, ups) to solve some bugs related to qiskit nature 0.2.0 property submodule
  
