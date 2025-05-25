import Pkg

Pkg.activate("./") # uncomment for using local QuantumMAMBO installation
import QuantumMAMBO as QM


trunc = 1e-5 #value beneath which considers a coefficient 0

H_CHAINS = ["h$i" for i in 4:2:30]

###### SAVELOAD ROUTINES FOR MOLECULAR HAMILTONIAN #######
function name_to_xyz(mol_name)
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

	return xyz
end

function sparsify!(X)
	for (i,x) in enumerate(X)
		if abs(x) < trunc
			X[i] = 0
		end
	end
end

MOL_LIST = ["h2", "lih", "h2o", "beh2", "nh3", H_CHAINS...]


for mol_name in MOL_LIST
	println("Starting molecule $mol_name")
	println("Building Hamiltonian and FB localization")
	xyz = name_to_xyz(mol_name)
	file_name = QM.DATAFOLDER * mol_name
	@time Hhf, Hfb, η = QM.LOCALIZED_XYZ_HAM(xyz, file_name, true, basis = "sto3g")
	println("Doing orbital rotation")
	@time Hrot = QM.ORBITAL_OPTIMIZATION(Hhf; verbose=true, SAVELOAD=true, SAVENAME=file_name * ".h5")
end