[![Code Quality Score](https://www.code-inspector.com/project/29093/score/svg)](https://frontend.code-inspector.com/public/project/29093/C-VQE/dashboard)  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Constrained Variational Quantum Eigensolver

## Description

This repository contains all the code written while looking for a better constrained VQE implementation

## Usage

* Meet the requirements
* Create logs directory
* Run test.py
* Enjoy

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
  
