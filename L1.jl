# OPTIONS FOR WHAT ROUTINES SHOULD BE RAN
DO_CSA = false #perform Cartan Sub-Algebra (CSA) decomposition of Hamiltonian
DO_DF = true #perform Double-Factorization (DF) decomposition of Hamiltonian
DO_ΔE = true #obtain lower bound ΔE/2 of Hamiltonian, only for small systems!
FOCK_BOUND = true #obtain lower bound for ΔE/2 using Fock matrix, can be done when ΔE is not accessible!
DO_AC = true #do anticommuting grouping technique
DO_OO = false #do orbital optimization routine
DO_SQRT = false #obtain square-root factorization 1-norms
DO_MHC = false #do Majorana Hyper-Contraction routine (SVD-based MTD-1ˆ4)
DO_MTD_CP4 = false #MTD-1ˆ4
DO_THC = false #tensor hypercontraction
SYM_SHIFT = true #do symmetry-shift optimization routine (i.e. partial shift)
INT = false #do interaction picture optimization routines
verbose = true #verbose for sub-routines
COUNT = true #whether total number of unitaries should be counted
BLISS = true #whether block-invariant symmetry shift routine is done
DO_TROTTER = false #whether Trotter α is calculated, requires parallel routines
DO_FC = false #whether fully-commuting routine is done
TABLE_PRINT = true #whether final 1-norms are printed for copy-pasting in LaTeX table
SYM_SUBSPACE = false #whether spectral range is calculated for Hamiltonian in reduced-symmetry subspace

######## RUNNING CODE
mol_name = ARGS[1]

import Pkg

Pkg.activate("./") # uncomment for using local QuantumMAMBO installation
Pkg.instantiate() # uncomment for using local QuantumMAMBO installation

using QuantumMAMBO: DATAFOLDER, SAVELOAD_HAM, RUN_L1, symmetry_treatment, INTERACTION, bliss_optimizer, PAULI_L1, quadratic_bliss, Ne_block_diagonalizer, matrix_symmetry_block, to_matrix, bliss_linprog, quadratic_bliss_optimizer


###### SAVELOAD ROUTINES FOR MOLECULAR HAMILTONIAN #######
FILENAME = DATAFOLDER * mol_name
H, η = SAVELOAD_HAM(mol_name, FILENAME)

###### END: SAVELOAD ROUTINES FOR MOLECULAR HAMILTONIAN #######

RUN_L1(H, η=η, DO_CSA=DO_CSA, DO_DF=DO_DF, DO_ΔE=DO_ΔE, LATEX_PRINT=TABLE_PRINT,
    DO_FC=DO_FC, SYM_RED=DO_TROTTER, DO_AC=DO_AC, DO_OO=DO_OO, DO_THC=DO_THC,
    DO_SQRT=DO_SQRT, DO_TROTTER=DO_TROTTER, DO_MHC=DO_MHC, DO_MTD_CP4=DO_MTD_CP4,
    COUNT=COUNT, verbose=verbose, name=FILENAME * ".h5", FOCK_BOUND=FOCK_BOUND)


if SYM_SHIFT
    println("\n\nStarting symmetry-shift routine...")
    @time H_SYM, shifts = symmetry_treatment(H, verbose=verbose, SAVENAME=FILENAME * "_SYM.h5") # H = H_SYM + shifts[1]*Ne2 + shifts[2]*Ne
    println("Finished obtaining symmetry shifts, running routines for shifted Hamiltonian...")
    RUN_L1(H_SYM, η=η, DO_CSA=DO_CSA, DO_DF=DO_DF, DO_ΔE=DO_ΔE, LATEX_PRINT=TABLE_PRINT,
        DO_FC=DO_FC, SYM_RED=DO_TROTTER, DO_AC=DO_AC, DO_OO=DO_OO, DO_THC=DO_THC,
        DO_SQRT=DO_SQRT, DO_TROTTER=DO_TROTTER, DO_MHC=DO_MHC, DO_MTD_CP4=DO_MTD_CP4,
        COUNT=COUNT, verbose=verbose, name=FILENAME * "_SYM.h5", FOCK_BOUND=FOCK_BOUND)
end

if INT
    println("\n\nStarting interaction picture routine...")
    @time H_INT = INTERACTION(H, SAVENAME=FILENAME * "_INT.h5")
    println("Finished obtaining interaction picture Hamiltonian, starting post-processing...")
    RUN_L1(H_INT, η=η, DO_CSA=DO_CSA, DO_DF=DO_DF, DO_ΔE=DO_ΔE, LATEX_PRINT=TABLE_PRINT,
        DO_FC=DO_FC, SYM_RED=DO_TROTTER, DO_AC=DO_AC, DO_OO=DO_OO, DO_THC=DO_THC,
        DO_SQRT=DO_SQRT, DO_TROTTER=DO_TROTTER, DO_MHC=DO_MHC, DO_MTD_CP4=DO_MTD_CP4,
        COUNT=COUNT, verbose=verbose, name=FILENAME * "_INT.h5", FOCK_BOUND=FOCK_BOUND)
end

if BLISS
    println("\n\n Starting block-invariant symmetry shift (BLISS) routine...")
    println("BLISS optimization...")
    H_bliss, _ = bliss_linprog(H, η)

    println("Running 1-norm routines...")
    @time H_bliss = bliss_optimizer(H, η, verbose=verbose, SAVENAME=FILENAME * "_BLISS.h5")
    RUN_L1(H_bliss, η=η, DO_CSA=DO_CSA, DO_DF=DO_DF, DO_ΔE=DO_ΔE, LATEX_PRINT=TABLE_PRINT,
        DO_FC=DO_FC, SYM_RED=DO_TROTTER, DO_AC=DO_AC, DO_OO=DO_OO, DO_THC=DO_THC,
        DO_SQRT=DO_SQRT, DO_TROTTER=DO_TROTTER, DO_MHC=DO_MHC, DO_MTD_CP4=DO_MTD_CP4,
        COUNT=COUNT, verbose=verbose, name=FILENAME * "_BLISS.h5", FOCK_BOUND=FOCK_BOUND)
end

if BLISS && INT
    println("\n\n Starting interaction picture + BLISS routines...")
    println("\nRunning before routine (H -> bliss -> int)")
    @time H_before = INTERACTION(H_bliss, SAVENAME=FILENAME * "_BLISS_INT.h5")
    RUN_L1(H_before, η=η, DO_CSA=DO_CSA, DO_DF=DO_DF, DO_ΔE=DO_ΔE, LATEX_PRINT=TABLE_PRINT,
        DO_FC=DO_FC, SYM_RED=DO_TROTTER, DO_AC=DO_AC, DO_OO=DO_OO, DO_THC=DO_THC,
        DO_SQRT=DO_SQRT, DO_TROTTER=DO_TROTTER, DO_MHC=DO_MHC, DO_MTD_CP4=DO_MTD_CP4,
        COUNT=COUNT, verbose=verbose, name=FILENAME * "_BLISS_INT.h5", FOCK_BOUND=FOCK_BOUND)
end

if SYM_SUBSPACE
    using LinearAlgebra
    println("\n\n Starting symmetry-subspace routine")
    println("Building subspace-projector")
    Uη = Ne_block_diagonalizer(H.N, η, spin_orb=H.spin_orb)
    println("Building reduced H matrix...")
    @time Hη = matrix_symmetry_block(to_matrix(H), Uη)
    println("Diagonalizing symmetry-projected Hamiltonian")
    @time E, _ = eigen(Hη)
    λη_min = (maximum(real.(E)) - minimum(real.(E))) / 2
    @show λη_min
end
