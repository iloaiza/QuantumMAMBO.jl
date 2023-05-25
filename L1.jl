# OPTIONS FOR WHAT ROUTINES SHOULD BE RAN
DO_CSA = true #perform Cartan Sub-Algebra (CSA) decomposition of Hamiltonian
DO_DF = true #perform Double-Factorization (DF) decomposition of Hamiltonian
DO_ΔE = true #obtain lower bound ΔE/2 of Hamiltonian, only for small systems!
DO_AC = true #do anticommuting grouping technique
DO_OO = true #do orbital optimization routine
DO_SQRT = true #obtain square-root factorization 1-norms
DO_MHC = false #do Majorana Hyper-Contraction routine
SYM_SHIFT = true #do symmetry-shift optimization routine (i.e. partial shift)
INT = true #do interaction picture optimization routines
verbose = false #verbose for sub-routines
COUNT = true #whether total number of unitaries should be counted
BLISS = true #whether block-invariant symmetry shift routine is done
DO_TROTTER = false #whether Trotter α is calculated, requires parallel routines
DO_FC = false #whether fully-commuting routine is done
TABLE_PRINT = true #whether final 1-norms are printed for copy-pasting in LaTeX table

######## RUNNING CODE
mol_name = ARGS[1]

#Load everywhere if parallel
include("src/MAMBO.jl")
using .MAMBO

###### SAVELOAD ROUTINES FOR MOLECULAR HAMILTONIAN #######
FILENAME = DATAFOLDER*mol_name
H,η = MAMBO.SAVELOAD_HAM(mol_name, FILENAME)

###### END: SAVELOAD ROUTINES FOR MOLECULAR HAMILTONIAN #######

MAMBO.RUN(H, η=η, DO_CSA = DO_CSA, DO_DF = DO_DF, DO_ΔE = DO_ΔE, LATEX_PRINT = TABLE_PRINT, 
	DO_FC = DO_FC, SYM_RED=DO_TROTTER, DO_AC = DO_AC, DO_OO = DO_OO,
	DO_SQRT = DO_SQRT, DO_TROTTER=DO_TROTTER, DO_MHC = DO_MHC, COUNT = COUNT, verbose=verbose, name=FILENAME*".h5")

if SYM_SHIFT
	println("\n\nStarting symmetry-shift routine...")
	@time H_SYM, shifts = MAMBO.symmetry_treatment(H, verbose=verbose, SAVENAME=FILENAME*"_SYM.h5") # H = H_SYM + shifts[1]*Ne2 + shifts[2]*Ne
	println("Finished obtaining symmetry shifts, running routines for shifted Hamiltonian...")
	MAMBO.RUN(H_SYM, η=η, DO_CSA = DO_CSA, DO_DF = DO_DF, DO_ΔE = DO_ΔE, LATEX_PRINT = TABLE_PRINT, 
		DO_FC = DO_FC, SYM_RED=DO_TROTTER, DO_AC = DO_AC, DO_OO = DO_OO,
		DO_SQRT = DO_SQRT, DO_TROTTER=DO_TROTTER, DO_MHC = DO_MHC, COUNT = COUNT, verbose=verbose, name=FILENAME*"_SYM.h5")
end

if INT
	println("\n\nStarting interaction picture routine...")
	@time H_INT = MAMBO.INTERACTION(H, SAVENAME=FILENAME*"_INT.h5")
	println("Finished obtaining interaction picture Hamiltonian, starting post-processing...")
	MAMBO.RUN(H_INT, η=η, DO_CSA = DO_CSA, DO_DF = DO_DF, DO_ΔE = DO_ΔE, LATEX_PRINT = TABLE_PRINT, 
		DO_FC = DO_FC, SYM_RED=DO_TROTTER, DO_AC = DO_AC, DO_OO = DO_OO,
		DO_SQRT = DO_SQRT, DO_TROTTER=DO_TROTTER, DO_MHC = DO_MHC, COUNT = COUNT, verbose=verbose, name=FILENAME*"_INT.h5")
end

if BLISS
	println("\n\n Starting block-invariant symmetry shift (BLISS) routine...")
	println("BLISS optimization...")
	H_bliss = MAMBO.bliss_optimizer(H, η, verbose=verbose, SAVENAME=FILENAME*"_BLISS.h5")
	#H_bliss = quadratic_bliss(H, η)
	println("Running 1-norm routines...")
	MAMBO.RUN(H_bliss, η=η, DO_CSA = DO_CSA, DO_DF = DO_DF, DO_ΔE = DO_ΔE, LATEX_PRINT = TABLE_PRINT, 
		DO_FC = DO_FC, SYM_RED=DO_TROTTER, DO_AC = DO_AC, DO_OO = DO_OO,
		DO_SQRT = DO_SQRT, DO_TROTTER=DO_TROTTER, DO_MHC = DO_MHC, COUNT = COUNT, verbose=verbose, name=FILENAME*"_BLISS.h5")
end

if BLISS && INT
	println("\n\n Starting interaction picture + BLISS routines...")
	println("\nRunning before routine (H -> bliss -> int)")
	@time H_before = MAMBO.INTERACTION(H_bliss, SAVENAME=FILENAME*"_BLISS_INT.h5")
	MAMBO.RUN(H_before, η=η, DO_CSA = DO_CSA, DO_DF = DO_DF, DO_ΔE = DO_ΔE, LATEX_PRINT = TABLE_PRINT, 
		DO_FC = DO_FC, SYM_RED=DO_TROTTER, DO_AC = DO_AC, DO_OO = DO_OO,
		DO_SQRT = DO_SQRT, DO_TROTTER=DO_TROTTER, DO_MHC = DO_MHC, COUNT = COUNT, verbose=verbose, name=FILENAME*"_BLISS_INT.h5")
end
