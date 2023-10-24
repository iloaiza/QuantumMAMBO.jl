mol_name = ARGS[1]

import Pkg

Pkg.activate("./") # uncomment for using local QuantumMAMBO installation
import QuantumMAMBO as QM


###### SAVELOAD ROUTINES FOR MOLECULAR HAMILTONIAN #######
FILENAME = QM.DATAFOLDER*mol_name
H,η = QM.SAVELOAD_HAM(mol_name, FILENAME)

println("Running Pauli (Sparse) routine...")
@time t_count, tot_qubits = QM.Pauli_circuit(H)
@show t_count
@show tot_qubits

println("\nRunning DF routine...")
@time t_count, tot_qubits = QM.DF_circuit(H)
@show t_count
@show tot_qubits

println("\nRunning AC routine...")
@time t_count, tot_qubits = QM.AC_circuit(H)
@show t_count
@show tot_qubits