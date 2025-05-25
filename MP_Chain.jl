using Distributed

addprocs(39)
@everywhere import Pkg
@everywhere Pkg.activate("./")
#Pkg.resolve()
#Pkg.instantiate()

@everywhere using QuantumMAMBO:DATAFOLDER, SAVELOAD_HAM, RUN_L1, symmetry_treatment, INTERACTION, bliss_optimizer, PAULI_L1, quadratic_bliss, Ne_block_diagonalizer, matrix_symmetry_block, to_matrix, bliss_linprog, quadratic_bliss_optimizer, THC_grad, THC_fixed_uni_step, THC_tb_x_to_F_OP, THC_fixed_uni_step_lsq, THC_tb_lsq, F_OP, Fock_bound,ob_correction, F_OP_space_to_spin, SQRT_L1, lanczos_total_range, lanczos_range, TB_extract, THC_cost, one_body_L1, LOCALIZED_XYZ_HAM

@everywhere begin
N=parse(Int64, ARGS[1])
M=parse(Int64, ARGS[2])


function H_CHAIN(n, r = 1.4)
	xyz = String[]
	for i in 1:n
		my_r = i*r
		push!(xyz,"H 0.0 0.0 $my_r\n")
	end

	return LOCALIZED_XYZ_HAM(xyz, DATAFOLDER * "chain_H$n", true)
end

Hhf, Hfb, η = H_CHAIN(N) #we are using Foster-Boys localization for the H-chains

println("\n\nCalculating 1-norms...")
println("1-body:")
@time λ1 = one_body_L1(Hfb, count=false)

@show λ1


println("\n\nTHC routine...")
		
@time _, λ2_THC, iterations= THC_tb_lsq(Hfb, M, "H_"*ARGS[1]*"_"*ARGS[2]*".h5")
@show λ_THC = λ1+λ2_THC
@show iterations
@show λ2_THC

step_cost, toffoli_cost, ancilla_cost= THC_cost(2*Hfb.N, λ2_THC, M, iterations)
@show step_cost, toffoli_cost, ancilla_cost 
end
