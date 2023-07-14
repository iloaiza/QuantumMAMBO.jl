# QuantumMAMBO.jl: Efficient many-body routines in Julia

QuantumMAMBO provides structures for many-body objects in quantum computing. They can all be seen in `src/UTILS/structures.jl`. In particular, it provides the fermionic operators `F_OP` specially geared towards two-electron number-conserving operators, `Q_OP` for qubit operators, and `M_OP`for Majorana operators coming from fermionic operators. All operators and unitaries have structured classes, which in the future will be used for efficient compilation of quantum circuits and resource estimates.


## Using QuantumMAMBO.jl
`QuantumMAMBO.jl` includes efficient implementations for fermionic and qubit operators. To obtain results shown in Ref.(1), run on a terminal (e.g. for LiH):

`julia L1.jl lih`

All options and tolerances can be seen in `src/config.jl`. By default, `QuantumMAMBO.jl` will use the package [`PythonCall`](https://github.com/cjdoris/PythonCall.jl) for calling Python, installing all necessary Python packages on a fresh conda environment using [`MicroMamba`](https://github.com/cjdoris/MicroMamba.jl). Change the `PY_ENV` variable in `src/config.jl` to `"Null"` for using local Python enviornment instead. There are three ways in which QuantumMAMBO can be used, which are now decribed.

### (1) Native Julia package
Using Julia's package manager, add the `QuantumMAMBO` package. Can either be done by typing `]` on the Julia REPL, followed by `add QuantumMAMBO`, or by running the following lines:
```
import Pkg
Pkg.add("QuantumMAMBO")
```
### (2) Git cloning and using local installation
For development, this repository can be cloned and called from a Julia session in the directory with the commands:
```
import Pkg
Pkg.activate("./")
Pkg.instantiate()
using QuantumMAMBO
```
This allows for changes to be done in the package and tried out before creating a pull request for uploading a new version of the package.

### (3) Interface with Python
For using QuantumMAMBO from a Python session, the `juliacall` Python package is required, which can be installed with `pip install juliacall`. Once installed, QuantumMAMBO can be installed on Python as:
```
import juliacall
from juliacall import Main as jl

jl.seval('import Pkg')
jl.seval('Pkg.add("QuantumMAMBO")')
```
Once installed, it can be imported as a Python module by running:
```
import juliacall
from juliacall import Main as jl

jl.seval("using QuantumMAMBO")

mambo = jl.QuantumMAMBO
```
See `pyMAMBO.py` for an example script of interfacing Python with QuantumMAMBO. QuantumMAMBO can also be called from a local installation for Python by instead using the following script:
```
import juliacall
from juliacall import Main as jl

jl.seval('include("src/QuantumMAMBO.jl")')
jl.seval("using .QuantumMAMBO")

mambo = jl.QuantumMAMBO
```


## Module overview

### Main folder
	- L1.jl: full workflow for obtaining LCU 1-norms for all decompositions/methods
	- Project.toml: package metadata and dependencies
	- CondaPkg.toml: python installation dependencies for micromamba environment
	- pyMAMBO.py: example of python script using QuantumMAMBO


### src foler
	- config.jl: general parameters and settings for all functions
	- QuantumMAMBO.jl: wrapper for the module, includes all necessary files
	- UTILS folder: contains all functions

### UTILS folder
	- bliss.jl: functions for BLISS routine (see Ref. 2)
	- cost.jl: functions for calculating different norms of operators. Mainly 1- and 2-norms of fermionic operators.
	- decompose.jl: CSA, DF, and related decompositions of fermionic operators
	- ferm_utils.py: utilities for fermionic operators in python, interfaces with openfermion
	- fermionic.jl: utilities for QuantumMAMBO fermionic operators class (F_OP, defined in structures.jl)
	- gradient.jl: gradients for CSA and CSA_SD optimizations
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


## References
This code was developped and used for all results in the publications:

[1] - I. Loaiza, A. Marefat Khah, N. Wiebe, and A. F. Izmaylov, Reducing molecular electronic Hamiltonian simulation cost for Linear Combination of Unitaries approaches. Quantum Sci. Technol. 8 (3) 035019, 2023.

[2] - I. Loaiza, A. F. Izmaylov, Reducing the molecular electronic Hamiltonian encoding costs on quantum computers by symmetry shifts. arXiv:2304.13772, 2023.

## Code collaborators
	- Ignacio Loaiza (@iloaiza)
	- Aritra Brahmachari (@AritraBrahmachari)

