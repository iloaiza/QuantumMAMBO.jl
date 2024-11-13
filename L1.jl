# OPTIONS FOR WHAT ROUTINES SHOULD BE RAN
DO_CSA = false #perform Cartan Sub-Algebra (CSA) decomposition of Hamiltonian
DO_DF = false #perform Double-Factorization (DF) decomposition of Hamiltonian
DO_ΔE = false #obtain lower bound ΔE/2 of Hamiltonian, only for small systems!
FOCK_BOUND = false #obtain lower bound for ΔE/2 using Fock matrix, can be done when ΔE is not accessible!
DO_AC = false #do anticommuting grouping technique
DO_OO = false #do orbital optimization routine
DO_SQRT = false #obtain square-root factorization 1-norms
DO_MHC = false #do Majorana Hyper-Contraction routine (SVD-based MTD-1ˆ4)
DO_MTD_CP4 = false #MTD-1ˆ4
DO_THC = true #tensor hypercontraction
SYM_SHIFT = false #do symmetry-shift optimization routine (i.e. partial shift)
INT = false #do interaction picture optimization routines
verbose = true #verbose for sub-routines
COUNT = false #whether total number of unitaries should be counted
BLISS = false #whether block-invariant symmetry shift routine is done
DO_LANCZOS=false #whether ΔE/2 is estimated using Lanczos iteration
DO_TROTTER = false #whether Trotter α is calculated, requires parallel routines
DO_FC = false #whether fully-commuting routine is done
TABLE_PRINT = true #whether final 1-norms are printed for copy-pasting in LaTeX table
SYM_SUBSPACE = false #whether spectral range is calculated for Hamiltonian in reduced-symmetry subspace

######## RUNNING CODE
mol_name = ARGS[1]

import Pkg

Pkg.activate("./") # uncomment for using local QuantumMAMBO installation
Pkg.instantiate()
using QuantumMAMBO: DATAFOLDER, SAVELOAD_HAM, RUN_L1, symmetry_treatment, INTERACTION, bliss_optimizer, PAULI_L1, quadratic_bliss, Ne_block_diagonalizer, matrix_symmetry_block, to_matrix, bliss_linprog, quadratic_bliss_optimizer, THC_grad, THC_fixed_uni_step, THC_tb_x_to_F_OP, THC_fixed_uni_step_lsq, THC_tb_lsq, F_OP, Fock_bound,ob_correction, F_OP_space_to_spin, SQRT_L1, lanczos_total_range, lanczos_range, TB_extract


###### SAVELOAD ROUTINES FOR MOLECULAR HAMILTONIAN #######
FILENAME = DATAFOLDER*mol_name
H,η = SAVELOAD_HAM(mol_name, FILENAME)

#Some testing stuff

#H=TB_extract(H)

#=obt=zeros(H.N, H.N)
tbt=zeros(H.N, H.N, H.N, H.N)
F1=F_OP(([0.0],obt,H.mbts[3]))
@show Fock_bound(F1)
H=F1
@show H.N=#
#@show SQRT_L1(F1)
#H=F_OP_space_to_spin(F1)
#@show Fock_bound(H)
#@show SQRT_L1(H)

#Some more testing
#=obt=H.mbts[2]+ob_correction(H)
tbt=zeros(H.N, H.N, H.N, H.N)
F1=F_OP(([0],obt,tbt))
H=F1=#


###### END: SAVELOAD ROUTINES FOR MOLECULAR HAMILTONIAN #######

RUN_L1(H, η=η, DO_CSA = DO_CSA, DO_DF = DO_DF, DO_ΔE = DO_ΔE, LATEX_PRINT = TABLE_PRINT, 
	DO_FC = DO_FC, SYM_RED=DO_TROTTER, DO_AC = DO_AC, DO_OO = DO_OO, DO_THC = DO_THC, 
	DO_SQRT = DO_SQRT, DO_LANCZOS=DO_LANCZOS, DO_TROTTER=DO_TROTTER, DO_MHC = DO_MHC, DO_MTD_CP4 = DO_MTD_CP4, 
	COUNT = COUNT, verbose=verbose, name=FILENAME*".h5", FOCK_BOUND = FOCK_BOUND)


if SYM_SHIFT
	println("\n\nStarting symmetry-shift routine...")
	@time H_SYM, shifts = symmetry_treatment(H, verbose=verbose, SAVENAME=FILENAME*"_SYM.h5") # H = H_SYM + shifts[1]*Ne2 + shifts[2]*Ne
	println("Finished obtaining symmetry shifts, running routines for shifted Hamiltonian...")
	RUN_L1(H_SYM, η=η, DO_CSA = DO_CSA, DO_DF = DO_DF, DO_ΔE = DO_ΔE, LATEX_PRINT = TABLE_PRINT, 
		DO_FC = DO_FC, SYM_RED=DO_TROTTER, DO_AC = DO_AC, DO_OO = DO_OO, DO_THC = DO_THC, 
		DO_SQRT = DO_SQRT, DO_TROTTER=DO_TROTTER, DO_MHC = DO_MHC, DO_MTD_CP4 = DO_MTD_CP4, 
		COUNT = COUNT, verbose=verbose, name=FILENAME*"_SYM.h5", FOCK_BOUND = FOCK_BOUND)
end

if INT
	println("\n\nStarting interaction picture routine...")
	@time H_INT = INTERACTION(H, SAVENAME=FILENAME*"_INT.h5")
	println("Finished obtaining interaction picture Hamiltonian, starting post-processing...")
	RUN_L1(H_INT, η=η, DO_CSA = DO_CSA, DO_DF = DO_DF, DO_ΔE = DO_ΔE, LATEX_PRINT = TABLE_PRINT, 
		DO_FC = DO_FC, SYM_RED=DO_TROTTER, DO_AC = DO_AC, DO_OO = DO_OO, DO_THC = DO_THC, 
		DO_SQRT = DO_SQRT, DO_TROTTER=DO_TROTTER, DO_MHC = DO_MHC, DO_MTD_CP4 = DO_MTD_CP4, 
		COUNT = COUNT, verbose=verbose, name=FILENAME*"_INT.h5", FOCK_BOUND = FOCK_BOUND)
end

if BLISS
	println("\n\n Starting block-invariant symmetry shift (BLISS) routine...")
	println("BLISS optimization...")
	H_bliss,_ = bliss_linprog(H, η,verbose=true)
	H_grad_bliss=quadratic_bliss_optimizer(H, η,SAVELOAD=false)
	
	println("Running 1-norm routines...")
	#@time H_bliss = bliss_optimizer(H, η, verbose=verbose, SAVENAME=FILENAME*"_BLISS.h5")
	RUN_L1(H_bliss, η=η, DO_CSA = DO_CSA, DO_DF = DO_DF, DO_ΔE = DO_ΔE, LATEX_PRINT = TABLE_PRINT, 
		DO_FC = DO_FC, SYM_RED=DO_TROTTER, DO_AC = DO_AC, DO_OO = DO_OO, DO_THC = DO_THC, 
		DO_SQRT = DO_SQRT, DO_LANCZOS=DO_LANCZOS, DO_TROTTER=DO_TROTTER, DO_MHC = DO_MHC, DO_MTD_CP4 = DO_MTD_CP4, 
		COUNT = COUNT, verbose=verbose, name=FILENAME*"_BLISS.h5", FOCK_BOUND = FOCK_BOUND)
end

if BLISS && INT
	println("\n\n Starting interaction picture + BLISS routines...")
	println("\nRunning before routine (H -> bliss -> int)")
	@time H_before = INTERACTION(H_bliss, SAVENAME=FILENAME*"_BLISS_INT.h5")
	RUN_L1(H_before, η=η, DO_CSA = DO_CSA, DO_DF = DO_DF, DO_ΔE = DO_ΔE, LATEX_PRINT = TABLE_PRINT, 
		DO_FC = DO_FC, SYM_RED=DO_TROTTER, DO_AC = DO_AC, DO_OO = DO_OO, DO_THC = DO_THC, 
		DO_SQRT = DO_SQRT, DO_TROTTER=DO_TROTTER, DO_MHC = DO_MHC, DO_MTD_CP4 = DO_MTD_CP4, 
		COUNT = COUNT, verbose=verbose, name=FILENAME*"_BLISS_INT.h5", FOCK_BOUND = FOCK_BOUND)
end

if SYM_SUBSPACE
	using LinearAlgebra
	println("\n\n Starting symmetry-subspace routine")
	println("Building subspace-projector")
	Uη = Ne_block_diagonalizer(H.N, η, spin_orb=H.spin_orb)
	println("Building reduced H matrix...")
	@time Hη = matrix_symmetry_block(to_matrix(H), Uη)
	println("Diagonalizing symmetry-projected Hamiltonian")
	@time E,_ = eigen(Hη)
	λη_min = (maximum(real.(E))-minimum(real.(E)))/2
	@show λη_min
end

#=if DO_THC
	println("Starting Tensor Hypercontraction routine...")
	#@time THC_LSQ(H,20)
	a=8
	#@time THC_fixed_uni_step(H, a)
	#@time THC_fixed_uni_step_lsq(H,a)
	@time H_thc, λ_thc=THC_tb_lsq(H, a)
	@show λ_thc
	#@show H_thc
	#H_spin=H_thc
	#H_spin=F_OP_space_to_spin(H_thc)
	@show Fock_bound(H_thc)
	#@show SQRT_L1(H_thc)
	
	#=E_max_total,E_min_total = lanczos_total_range(one_body_tensor=H_spin.mbts[2],two_body_tensor=H_spin.mbts[3],core_energy=H_spin.mbts[1][1], steps=7)
	delta_E_total=E_max_total - E_min_total
	@show E_max_total, E_min_total
	@show delta_E_total/2
	
	for elec in 1:H_spin.N
		E_max, E_min= lanczos_range(one_body_tensor=H_spin.mbts[2],two_body_tensor=H_spin.mbts[3],core_energy=H_spin.mbts[1][1], steps=8, num_electrons=elec)
		@show elec
		@show E_max, E_min
	    	delta_E=E_max-E_min
	    	
	    	@show delta_E/2	
	end=#
	#=a=5
	N=2
	lambda_L=Int64(a*(a+1)/2)
	unitary_L=a*(N-1)
	x=zeros(lambda_L+unitary_L)
	x.=rand(0:1,length(x))
	F=THC_x_to_F_OP(x, N, a,lambda_L)
	#@time THC_fixed_uni_step(F,a)
	@time THC_fixed_uni_step_lsq(H,a)=#
end=#
	
