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

println("Running PREPARE-based circuits...")
println("DF")
prep_op, sel_op = QM.MTD_circuit(H, flavour = "DF")

println("MPS")
prep_op, sel_op = QM.MTD_circuit(H, flavour = "MPS")

println("CP4")
prep_op, sel_op = QM.MTD_circuit(H, flavour = "CP4")

println("\nRunning SELECT-based circuits")

println("\nRunning DF routine...")
@time t_count, tot_qubits = QM.DF_circuit(H)
@show t_count
@show tot_qubits

println("\nRunning AC routine...")
@time t_count, tot_qubits = QM.AC_circuit(H)
@show t_count
@show tot_qubits