mol_name = ARGS[1]


basis="sto3g"

import Pkg

Pkg.activate("./") # uncomment for using local QuantumMAMBO installation
import QuantumMAMBO as QM


trunc = 1e-5 #value beneath which considers a coefficient 0

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

function DO_ESTIMATES(H, Hrot, ROUTINES=["DF_openfermion"]; do_orbs=true, rot_save_name="ROTATION", rank_hop = 10, cp4_ini_rank=50)
	H_sparse = 1 * H
	Hrot_sparse = 1 * Hrot

	@show H.N
	sparsify!(H_sparse.mbts[3])
	sparsify!(Hrot_sparse.mbts[3])

	println("Starting resource estimates for method list $ROUTINES")
	ESTIMATES = []
	METHODS = []


	if do_orbs == true
		if "sparse" in ROUTINES
			println("Starting orbital-optimized sparse routine...")
			@show estimate = QM.quantum_estimate(Hrot_sparse, "sparse")
			push!(ESTIMATES, estimate)
			push!(METHODS, "OO-Pauli")
		end
		if "AC" in ROUTINES
			println("Starting orbital-optimized AC routine...")
			@show estimate = QM.quantum_estimate(Hrot_sparse, "AC")
			push!(ESTIMATES, estimate)
			push!(METHODS, "OO-AC")
		end
	end

	for routine in ROUTINES
		println("Starting $routine routine...")
		if routine == "sparse" || routine == "AC"
			@show estimate = QM.quantum_estimate(H_sparse, routine)
		else
			@show estimate = QM.quantum_estimate(H, routine)
		end
		push!(ESTIMATES, estimate)
		push!(METHODS, routine)
	end

	return ESTIMATES, METHODS
end

#=
file_name = QM.DATAFOLDER * mol_name
Hhf, Hrot, η = QM.LOCALIZED_XYZ_HAM(xyz, file_name, true, basis = basis)

if IS_CHAIN == false
	Hrot = QM.ORBITAL_OPTIMIZATION(Hhf; verbose=true, SAVELOAD=true, SAVENAME=file_name * ".h5")
end


#DO_ESTIMATES(Hhf, Hrot, ["sparse", "CP4", "MPS", "DF", "DF_openfermion"])
= =#

file_name = QM.DATAFOLDER * mol_name
Hhf, Hrot, η = QM.LOCALIZED_XYZ_HAM(xyz, file_name, true, basis = basis)
A,λ = QM.quantum_estimate(Hhf, "DF_openfermion", verbose=true, SAVENAME = file_name * "_DF_OF	.h5")
T,Q = A

@show mol_name
@show λ*T
@show Q