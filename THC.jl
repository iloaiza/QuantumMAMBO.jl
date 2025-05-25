import Pkg

Pkg.activate("./") # uncomment for using local QuantumMAMBO installation
#Pkg.resolve()
#Pkg.instantiate()
using QuantumMAMBO: DATAFOLDER, SAVELOAD_HAM, RUN_L1, symmetry_treatment, INTERACTION, bliss_optimizer, PAULI_L1, quadratic_bliss, Ne_block_diagonalizer, matrix_symmetry_block, to_matrix, bliss_linprog, quadratic_bliss_optimizer, THC_grad, THC_fixed_uni_step, THC_tb_x_to_F_OP, THC_fixed_uni_step_lsq, THC_tb_lsq, F_OP, Fock_bound,ob_correction, F_OP_space_to_spin, SQRT_L1, lanczos_total_range, lanczos_range, TB_extract, THC_cost, one_body_L1


mol_name = ARGS[1]
step_size=parse(Int64, ARGS[2])


###### SAVELOAD ROUTINES FOR MOLECULAR HAMILTONIAN #######
FILENAME = DATAFOLDER*mol_name
H,η = SAVELOAD_HAM(mol_name, FILENAME)

println("\n\nCalculating 1-norms...")
println("1-body:")
@time λ1 = one_body_L1(H, count=false)
#@time λ1 = SQRT_L1(OB_extract(H,γ4_contribution=true), count=COUNT)
@show λ1


println("\n\nTHC routine...")
		
@time _, λ2_THC, iterations= THC_tb_lsq(H, step_size)
@show λ_THC = λ1+λ2_THC
@show iterations
@show λ2_THC

step_cost, toffoli_cost, ancilla_cost= THC_cost(2*H.N, λ_THC, step_size, iterations)
@show step_cost, toffoli_cost, ancilla_cost 


