# QuantumMAMBO.jl: Efficient many-body routines in Julia
## Version 1.3.0
QuantumMAMBO provides ways of evaluating and improving the costs of ground state energy estimation quantum algorithms.

## Linear Programming Block Invariant Symmetry Shift (LPBLISS)
LPBLISS uses the Block Invariant Symmetry Shift to reduce the L1 norm of a given Hamiltonian. The required optimization is 
formulated as a linear program.

An example of how to use LPBLISS to take in an FCIDUMP file and then return an FCIDUMP file of the LPBLISS-treated Hamiltonian is found in `examples/lpbliss_fcidump_run.jl`. 

Using a unix-flavour OS (Mac OS X, Linux, Windows Subsystem for Linux), the example can be run with the following steps.

1. If you do not already have Julia installed, run `curl -fsSL https://install.julialang.org | sh` in the terminal to download and begin the installation process. After installation, close and reopen the terminal.

2. Create and enter a folder for the code with
```
$ mkdir QuantumMAMBO_deltaE
$ cd QuantumMAMBO_deltaE/
```

3. Download this branch of QuantumMAMBO and unzip with
```
$ curl -LO https://github.com/iloaiza/QuantumMAMBO.jl/archive/refs/heads/deltaE_pyscf_upgrades.zip
$ unzip deltaE_pyscf_upgrades.zip
```
4. Enter the unzipped folder and run the example with
```
$ cd QuantumMAMBO.jl-deltaE_pyscf_upgrades
$ julia examples/lpbliss_fcidump_run.jl
```
Initial compilation and running of the code can take around 10 minutes, while subsequent runs are faster. After much prior output, you should see the following after the run:
```
-------------------------Hamiltonian Info-------------------------------------
FCIDUMP file path: examples/data/fcidump.36_1ru_II_2pl
Number of orbitals: 7
Number of spin orbitals: 14
Number of electrons: 12
Two S: 0
Two Sz: 0
Orbital symmetry: [0, 0, 0, 0, 0, 0, 0]
Extra attributes: Dict("ISYM" => 1)
-------------------------Delta E / 2, whole Fock space-------------------------------------
Original Hamiltonian, whole Fock space:
E_max, orig: -2881.779326309789
E_min, orig: -2982.401506717449
ΔE/2, orig: 50.31109020382996
LPBLISS-modified Hamiltonian, whole Fock space:
E_max, LPBLISS: -2968.3953798905172
E_min, LPBLISS: -2988.732425049674
ΔE/2, LPBLISS: 10.168522579578394
-------------------------Delta E / 2, Subspace------------------------------
Original Hamiltonian, 12 electrons:
E_max, orig, subspace: -2968.537986371558
E_min, orig, subspace: -2982.401515039751
ΔE/2, orig, subspace: 6.931764334096442
LPBLISS-modified Hamiltonian, 12 electrons:
E_max, LPBLISS, subspace: -2968.537986371558
E_min, LPBLISS, subspace: -2982.401515039752
ΔE/2, LPBLISS, subspace: 6.9317643340971244
------------------------L1 NORMS-------------------------------
Pauli L1 Norm, original Hamiltonian: 62.86697375040692
Pauli L1 Norm, LPBLISS-treated Hamiltonian: 13.325570371408116
```
See the last two lines of the output for the L1 norms of the Pauli version of LCU before and after LPBLISS.

After the run, you will see an FCIDUMP file `examples/data/fcidump.36_1ru_II_2pl_BLISS` and an HDF5 file `examples/data/36_1ru_II_2pl_BLISS.h5` containing the LPBLISS-treated Hamiltonian.

# Many-body Structures
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

QM = jl.QuantumMAMBO
```
See `pyMAMBO.py` for an example script of interfacing Python with QuantumMAMBO. QuantumMAMBO can also be called from a local installation for Python by instead using the following script:
```
import juliacall
from juliacall import Main as jl

jl.seval('import Pkg; Pkg.activate("./")')
jl.seval('Pkg.instantiate()')
jl.seval("using .QuantumMAMBO")

QM = jl.QuantumMAMBO
```


## Module overview

### Main folder
- `L1.jl` : full workflow for obtaining LCU 1-norms for all decompositions/methods
- `LCU.jl` : workflow for obtaining circuits of LCUs using cirq-ft
- `Project.toml` : package metadata and dependencies
- `CondaPkg.toml` : python installation dependencies for micromamba environment
- `pyMAMBO.py` : example of python script using QuantumMAMBO


### src foler
- `config.jl` : general parameters and settings for all functions
- `QuantumMAMBO.jl` : wrapper for the module, includes all necessary files
- `UTILS folder` : contains all functions

### UTILS folder
- `bliss.jl` : functions for BLISS routine (see Ref. 2)
- `circuits.jl` : julia wrappers for interfacing QuantumMAMBO LCU decompositions with cirq circuit building
- `circuits.py` : python wrapper which includes all cirq building tools in cirq folder
- `cirq` : folder containing all python functions for building cirq LCU oracles
- `cost.jl` : functions for calculating different norms of operators. Mainly 1- and 2-norms of fermionic operators.
- `decompose.jl` : CSA, DF, and related decompositions of fermionic operators
- `ferm_utils.py` : utilities for fermionic operators in python, interfaces with openfermion
- `fermionic.jl` : utilities for QuantumMAMBO fermionic operators class (F_OP, defined in structures.jl)
- `gradient.jl` : gradients for CSA and CSA_SD optimizations
- `guesses.jl` : initial guesses for decomposition routines
- `ham_utils.py` : python utilities for the electronic structure Hamiltonian, interfaces with openfermion
- `lcu.jl` : calculation of lcu 1-norms for different decompositions
- `linprog.jl` : linear programming routines for symmetry-shift optimization (see Refs. 1 and 2, corresponds to "partial" routine in Ref. 2)
- `majorana.jl` : utilities for QuantumMAMBO Majorana operators class (M_OP, defined in structures.jl)
- `orbitals.jl` : orbital optimization routine for 1-norm minimization (see Koridon et al., Phys. Rev. Res. 3 (3), 2021. Material is also covered in Refs. 1 and 2)
- `parallel.jl` : code with parallel capabilities, mainly for trotter bounds (under construction)
- `planted.jl` : routines for obtaining planted solutions for a given Hamiltonian
- `projectors.jl` : builds projectors of Fock space into constant number of electrons subspace, useful for Trotter bounds (under progress)
- `py_qubits.jl` : python utilities for qubit operators, interfaces with openfermion
- `py_utils.jl` : julia interface to all python modules and openfermion
- `qubit.jl` : utilities for QuantumMAMBO qubit operators class (Q_OP, defined in structures.jl)
- `saving.jl` : save-load utilities for decompositions and optimization results, uses HDF5
- `structures.jl` : definition of classes for many-body operators
- `symmetries.jl` : building of symmetry operators, e.g. Sz, Ne
- `symplectic.jl` : utilities for representing and manipulating qubit space as symplectic vectors
- `trotter.jl` : Trotterization implementation, errors and bounds (under construction)
- `unitaries.jl` : unitary transformations related to fermionic QuantumMAMBO operators (i.e. F_OP)
- `wrappers.jl` : runner functions which run workflows for obtaining all necessary quantities for e.g. tables in Refs. 1 and 2


## References
This code was developped and used for all results in the publications:

[1] I. Loaiza, A. Marefat Khah, N. Wiebe, and A. F. Izmaylov, Reducing molecular electronic Hamiltonian simulation cost for Linear Combination of Unitaries approaches. [Quantum Sci. Technol. 8 (3) 035019, 2023.](https://www.doi.org/10.1088/2058-9565/acd577)

[2] I. Loaiza, A. F. Izmaylov, Block-Invariant Symmetry Shift: Preprocessing technique for second-quantized Hamiltonians to improve their decompositions to Linear Combination of Unitaries. [arXiv:2304.13772, 2023.](https://arxiv.org/abs/2304.13772)

## Code collaborators
- Ignacio Loaiza (@iloaiza)
- Aritra Brahmachari (@AritraBrahmachari)
- Joshua T. Cantin (@jtcantin)
- Linjun Wang (@Zephrous5747): author of [module_sdstate](https://github.com/Zephrous5747/sdstate)
