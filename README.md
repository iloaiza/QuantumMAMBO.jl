# QuantumMAMBO.jl: Efficient many-body routines in Julia

## Using QuantumMAMBO.jl
`QuantumMAMBO.jl` includes efficient implementations for fermionic and qubit operators. To obtain results shown in Ref.(1), run on a terminal (e.g. for LiH):

julia L1.jl lih

All options and tolerances can be seen in config.jl. See installation for more information.

## Installation
Using Julia's package manager, add the "QuantumMAMBO" package. Can either be done by typing `]` on the Julia REPL, followed by `add QuantumMAMBO`, or by running the following lines:
```
import Pkg
Pkg.add("QuantumMAMBO")
```

By default, `QuantumMAMBO.jl` will use the package [`PythonCall`](https://github.com/cjdoris/PythonCall.jl) for calling python, installing all necessary python packages on a fresh conda environment using [`MicroMamba`](https://github.com/cjdoris/MicroMamba.jl). Uncommenting the lines in `src/UTILS/py_utils.jl` for `ENV["JULIA_CONDAPKG_BACKEND"]` and using a local installation of the `QuantumMAMBO.jl` package can be done to instead use the local python installation. Note that calling `QuantumMAMBO.jl` from python using the [`juliacall`](https://github.com/cjdoris/PythonCall.jl) package will crash if functions that use python are called. Such functions should be called directly from python, and are mostly used for transforming `QuantumMAMBO.jl` structures to/from [`Openfermion`](https://github.com/quantumlib/OpenFermion).

## Module overview

### UTILS folder
	- bliss.jl: functions for BLISS routine (see Ref. 2)
	- cost.jl: functions for calculating different norms of operators. Mainly 1- and 2-norms of fermionic operators.
	- decompose.jl: CSA, DF, and related decompositions of fermionic operators
	- ferm_utils.py: utilities for fermionic operators in python, interfaces with openfermion
	- fermionic.jl: utilities for QuantumMAMBO fermionic operators class (F_OP, defined in structures.jl)
	- guesses.jl: initial guesses for decomposition routines
	- ham_utils.py: python utilities for the electronic structure Hamiltonian, interfaces with openfermion
	- lcu.jl: calculation of lcu 1-norms for different decompositions
	- linprog.jl: linear programming routines for symmetry-shift optimization (see Refs. 1 and 2, corresponds to "partial" routine in Ref. 2)
	- majorana.jl: utilities for QuantumMAMBO Majorana operators class (M_OP, defined in structures.jl)
	- orbitals.jl: orbital optimization routine for 1-norm minimization (see Koridon et al., Phys. Rev. Res. 3 (3), 2021. Material is also covered in Refs. 1 and 2)
	- parallel.jl: code with parallel capabilities, mainly for trotter bounds (under construction)
	- projectors.jl: builds projectors of Fock space into constant number of electrons subspace, useful for Trotter bounds (under progress)
	- py_qubits.jl: python utilities for qubit operators, interfaces with openfermion
	- py_utils.jl: julia interface to all python modules and openfermion
	- qubit.jl: utilities for QuantumMAMBO qubit operators class (Q_OP, defined in structures.jl)
	- saving.jl: save-load utilities for decompositions and optimization results, uses HDF5
	- structures.jl: definition of classes for many-body operators
	- symmetries.jl: building of symmetry operators, e.g. Sz, Ne
	- symplectic.jl: utilities for representing and manipulating qubit space as symplectic vectors
	- trotter.jl: Trotterization implementation, errors and bounds (under construction)
	- unitaries.jl: unitary transformations related to fermionic QuantumMAMBO operators (i.e. F_OP)
	- wrappers.jl: runner functions which run workflows for obtaining all necessary quantities for e.g. tables in Refs. 1 and 2

### Main folder
	- L1.jl: full workflow for obtaining LCU 1-norms for all decompositions/methods
	- build.jl: single module for loading all utilities
	- config.jl: general parameters and settings for all functions
	- install.sh: installer file, see Installation section for more information


## References
This code was developped and used for all results in the publications:

[1] - I. Loaiza, A. Marefat Khah, N. Wiebe, and A. F. Izmaylov, Reducing molecular electronic Hamiltonian simulation cost for Linear Combination of Unitaries approaches. arXiv:2208.08272 (2022).

[2] - I. Loaiza, A. F. Izmaylov, Reducing the molecular electronic Hamiltonian encoding costs on quantum computers by symmetry shifts. arXiv:2304.13772 (2023).
