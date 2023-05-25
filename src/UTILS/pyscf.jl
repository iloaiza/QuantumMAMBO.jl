pyscf = pyimport("pyscf")
ofpyscf = pyimport("openfermionpyscf")

thc_py = pyimport("thc")

function mol_object(mol_name; geometry = 1, basis="sto3g")
	mol = pyscf.gto.Mole()
	mol.basis = basis
	if mol_name == "h2"
		mol.atom = "H 0 0 0; H 0 0 $geometry"
	elseif mol_name == "lih"
		mol.atom = "H 0 0 0; Li 0 0 $geometry"
	elseif mol_name == "beh2"
		mol.atom = "Be 0 0 0; H 0 0 $geometry; H 0 0 -$geometry"
	elseif mol_name == "n2"
		mol.atom = "N 0 0 0; N 0 0 $geometry"
	elseif mol_name == "h2o"
		angle = deg2rad(107.6 / 2)
        xDistance = geometries * sin(angle)
        yDistance = geometries * cos(angle)
		mol.atom = "O 0 0 0; H -$xDistance $yDistance 0; H $xDistance $yDistance 0"
	elseif mol_name == "nh3"
		bondAngle = deg2rad(107)
		cosval = cos(bondAngle)
		sinval = sin(bondAngle)
		thirdyRatio = (cosval - cosval^2) / sinval
		thirdxRatio = sqrt(1 - cosval^2 - thirdyRatio^2)
		mol.atom = "N 0 0 0; H 0 0 $geometry; H 0 $(sinval*geometry) $(cosval*geometry); H $(thirdxRatio*geometry) $(thirdyRatio*geometry) $(cosval*geometry)"
	else
		error("Molecular object builder not implemented for molecule $mol_name")
	end

	mol.build()

	return mol
end

function RHF(mol_name; geometry=1, basis="sto3g")
	mol = mol_object(mol_name, geometry=geometry, basis=basis)
	mf = pyscf.scf.ROHF(mol)
	mf.verbose=0
	mf.kernel()

	mf = of.resource_estimates.molecule.stability(mf)
	mf = of.resource_estimates.molecule.localize(mf, loc_type="pm")

	#mfcas = of.resource_estimates.molecule.pyscf_to_cas(mf)
	#_, mf = of.resource_estimates.molecule.cas_to_pyscf(mfcas)
	mf = thc_py.mf_eri(mf)

	return mf
end