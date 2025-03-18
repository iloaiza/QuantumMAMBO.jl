LOAD_HAM = false #whether to load Hamiltonian tensors from file or generate using PySCF

mol_name = ARGS[1]

import Pkg

Pkg.activate("./") # uncomment for using local QuantumMAMBO installation
Pkg.resolve()
Pkg.instantiate() # uncomment for using local QuantumMAMBO installation

import QuantumMAMBO

DATAFOLDER = "./data/"

# load one-body and two-body tensors

FILENAME = DATAFOLDER * mol_name

using HDF5
using PrettyTables


if LOAD_HAM == true
    println("Loading Hamiltonian tensors from file $FILENAME...")
    fid = h5open(FILENAME * ".h5", "r")
    obt = read(fid,"obt")
    tbt = read(fid,"tbt")
    η = read(fid,"Ne")
    num_orbs = size(obt)[1]
    if 2*num_orbs < η
        println("Warning, found $η electrons but only $(2*num_orbs) spin-orbitals, reducing number of electrons for consistency")
        η = 2*num_orbs
    end
    close(fid)
    H = QuantumMAMBO.F_OP(([0],obt,tbt))
    println("Loaded Hamiltonian with $num_orbs spacial orbitals and $η electrons")
else
    println("Generating Hamiltonian using PySCF for molecule $mol_name")
    @time H, η = QuantumMAMBO.SAVELOAD_HAM(mol_name, FILENAME)
    println("Produced $mol_name Hamiltonian with $η electrons and $(H.N) orbitals")
end

METHOD_NAME = []
ONE_NORMS = [] 
Q_COUNTS = []
T_COUNTS = []

println("\nObtaining naive Pauli decomposition estimates...")
T_Pauli, Q_Pauli = QuantumMAMBO.quantum_estimate(H, "sparse")
@time λ_Pauli = QuantumMAMBO.PAULI_L1(H)
@show λ_Pauli
push!(METHOD_NAME, "Pauli")
push!(ONE_NORMS, λ_Pauli)
push!(Q_COUNTS, Q_Pauli)
push!(T_COUNTS, T_Pauli)

println("\nStarting double factorization decomposition")
@time DF_FRAGS = QuantumMAMBO.DF_decomposition(H)
println("Finished DF decomposition, obtained $(length(DF_FRAGS)) fragments")

println("\nCalculating DF 1-norm")
@time λ1_DF = QuantumMAMBO.one_body_L1(H, count=false)
@time λ2_DF = sum(QuantumMAMBO.L1.(DF_FRAGS, count=false))
λtot_DF = λ1_DF + λ2_DF
@show λ1_DF
@show λ2_DF
@show λtot_DF

println("\nObtaining resource estimates for DF")
T_DF, Q_DF = QuantumMAMBO.df_be_cost(DF_FRAGS)
push!(METHOD_NAME, "DF")
push!(ONE_NORMS, λtot_DF)
push!(Q_COUNTS, Q_DF)
push!(T_COUNTS, T_DF)


println("\nObtaining individual BLISS shifts for each fragment")
@time bliss_frags, ϕs = QuantumMAMBO.df_bliss(DF_FRAGS)

println("\nCalculating DF+BLISS 2-body 1-norm")
@time λ2_BLISS = sum(QuantumMAMBO.L1.(bliss_frags, count=false))
@show λ2_BLISS

println("Obtaining 1-body correction")
global obt_tot = copy(H.mbts[2])

for (i,frag) in enumerate(bliss_frags)
    ob_op = QuantumMAMBO.F_OP(QuantumMAMBO.F_FRAG(1, frag.U, QuantumMAMBO.OBF(), frag.C, frag.N, frag.spin_orb, frag.coeff, frag.has_coeff))    
    global obt_tot += (2*ϕs[i]*η) .* ob_op.mbts[2]
end

F1 = QuantumMAMBO.to_OBF(obt_tot)
F1_bliss = QuantumMAMBO.ob_fragment_bliss(F1)
λ1_BLISS = QuantumMAMBO.L1(F1_bliss)
@show λ1_BLISS
λtot_BLISS = λ1_BLISS + λ2_BLISS
@show λtot_BLISS

println("\nObtaining BLISS resource estimates")
T_DF_BLISS, Q_DF_BLISS = QuantumMAMBO.df_be_cost(bliss_frags)
push!(METHOD_NAME, "DF+BLISS")
push!(ONE_NORMS, λtot_BLISS)
push!(Q_COUNTS, Q_DF_BLISS)
push!(T_COUNTS, T_DF_BLISS)


println("\n\nTL;DR")
col1 = METHOD_NAME
col2 = round.(ONE_NORMS, digits=2)
col3 = Int.(round.(Q_COUNTS, sigdigits=3))
col4 = round.(T_COUNTS, sigdigits=3)

data = hcat(col1, col2, col3, col4)
pretty_table(data, header = ["Method", "1-norm λ", "Qubits", "T-gates"])