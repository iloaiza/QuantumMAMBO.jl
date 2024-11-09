mol_name = ARGS[1]

basis="sto3g"

import Pkg

Pkg.activate("./") # uncomment for using local QuantumMAMBO installation
import QuantumMAMBO as QM


trunc = 1e-5 #value beneath which considers a coefficient 0

if length(ARGS) > 1
	cp4_jump = parse(Int64, ARGS[2])
else
	cp4_jump = 10
end

H_CHAINS = ["h$i" for i in 4:30]

###### SAVELOAD ROUTINES FOR MOLECULAR HAMILTONIAN #######
global IS_CHAIN = false
if mol_name == "h2"
	r = 0.741
	xyz = ["H 0.0 0.0 0.0", "H 0.0 0.0 $r"]
elseif mol_name == "lih"
	r = 1.595
	xyz = ["H 0.0 0.0 0.0", "Li 0.0 0.0 $r"]
elseif mol_name == "beh2"
	r = 1.326
	xyz = ["H 0.0 0.0 0.0", "Be 0.0 0.0 $r", "H 0.0 0.0 $(2*r)"]
elseif mol_name == "h2o"
	r = 0.958
	h2o_angle = 107.6 / 2
    h2o_angle = deg2rad(h2o_angle)
    xDistance = r*sin(h2o_angle)
    yDistance = r*cos(h2o_angle)
    xyz = ["O 0.0 0.0 0.0", "H -$xDistance $yDistance 0.0", "H $xDistance $yDistance 0.0"]
elseif mol_name == "nh3"	
	bondAngle = deg2rad(107)
    cosval = cos(bondAngle)
    sinval = sin(bondAngle)
	thirdyRatio = (cosval - cosval^2) / sinval
    thirdxRatio = sqrt(1 - cosval^2 - thirdyRatio^2)
    xyz = ["H 0.0 0.0 1.0", "H 0.0 $(sinval) $(cosval)", "H $(thirdxRatio) $(thirdyRatio) $(cosval)", "N 0.0 0.0 0.0"]
elseif mol_name in H_CHAINS
	global IS_CHAIN = true
	n = parse(Int, mol_name[2:end])
	r = 1.4 #1.4 Angstroms separation between hydrogens
	xyz = String[]
	for i in 1:n
		my_r = i*r
		push!(xyz,"H 0.0 0.0 $my_r\n")
	end
else
	error("Not supported mol_name = $mol_name input!")
end

function sparsify!(X)
	for (i,x) in enumerate(X)
		if abs(x) < trunc
			X[i] = 0
		end
	end
end

file_name = QM.DATAFOLDER * mol_name
Hhf, Hrot, η = QM.LOCALIZED_XYZ_HAM(xyz, file_name, true, basis = basis)
if IS_CHAIN == false
	Hrot = QM.ORBITAL_OPTIMIZATION(Hhf; verbose=true, SAVELOAD=true, SAVENAME=file_name * ".h5")
end

Hhf_full = 1 * Hhf
Hrot_full = 1 * Hrot

@show Hhf.N
sparsify!(Hhf.mbts[3])
sparsify!(Hrot.mbts[3])

QM.RUN_L1(Hhf_full, DO_CSA = false, DO_DF = false, DO_ΔE = true, LATEX_PRINT = false, 
	DO_FC = false, SYM_RED=false, DO_AC = false, DO_OO = false, DO_THC = false, 
	DO_SQRT = false, DO_TROTTER=false, DO_MHC = false, DO_CP4 = false,
	COUNT = false, verbose=false, name=file_name*".h5", FOCK_BOUND = true)

println("\n\n\n")
println("Finished obtaining Hamiltonian, starting resource estimates...")
println("\n\n\nObtaining results for Rotated Hamiltonian")
println("\nRunning Pauli routine...")
@show QM.quantum_estimate(Hrot, "sparse")

println("\nRunning AC routine...")
@show QM.quantum_estimate(Hrot, "AC")


println("\n\n\nObtaining results for HF Hamiltonian")
println("\nRunning Pauli routine...")
@show QM.quantum_estimate(Hhf, "sparse")

println("\nRunning AC routine...")
@show QM.quantum_estimate(Hhf, "AC")

println("\nRunning SVD routines...")
@show QM.quantum_estimate(Hhf_full, "SVD")

println("\nRunning CP4 routines...")
@show QM.quantum_estimate(Hhf_full, "CP4", rank_hop = cp4_jump)

println("\nRunning MPS routines...")
@show QM.quantum_estimate(Hhf_full, "MPS")

println("\nRunning DF routines...")
@show QM.quantum_estimate(Hhf_full, "DF")



