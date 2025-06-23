#FUNCTIONS FOR INTERFACING WITH PYTHON
ENV["JULIA_CONDAPKG_BACKEND"] = PY_BACKEND

using PythonCall
np = pyimport("numpy")
scipy = pyimport("scipy")
sympy = pyimport("sympy")
of = pyimport("openfermion")

UTILS_DIR = @__DIR__
sys = pyimport("sys")
sys.path.append(UTILS_DIR)
ham = pyimport("ham_utils")
fermionic = pyimport("ferm_utils")
qub = pyimport("py_qubits")

of_simplify(OP) = of.reverse_jordan_wigner(of.jordan_wigner(OP))

function qubit_transform(op, transformation=F2Q_map)
	if transformation == "bravyi_kitaev" || transformation == "bk"
		op_qubit = of.bravyi_kitaev(op)
	elseif transformation == "jordan_wigner" || transformation == "jw"
		op_qubit = of.jordan_wigner(op)
	else
		error("$transformation not implemented for fermion to qubit operator maping")
	end

	return op_qubit
end

function obtain_OF_hamiltonian(mol_name; kwargs...)
	return ham.get_system(mol_name,n_elec=true; kwargs...)		
end

function xyz_OF_hamiltonian(xyz_string; kwargs...)
	return ham.system_from_xyz(xyz_string, n_elec = true, spin=0; kwargs...)
end

function localized_xyz_OF_hamiltonian(xyz_string; kwargs...)
	return ham.localized_ham_from_xyz(xyz_string, n_elec = true, spin=Spin, charge=Charge,rhf=ROHF; kwargs...)
end

function obtain_H(mol_name; kwargs...)
	#returns fermionic operator of H in orbitals and number of electrons
	h_ferm, num_elecs = obtain_OF_hamiltonian(mol_name; kwargs...)
	
	Hconst, obt, tbt = fermionic.to_tensors(h_ferm)
	obt = pyconvert(Array{Float64},obt)
	tbt = pyconvert(Array{Float64},tbt)

	mbts = ([pyconvert(Float64, Hconst)], obt, tbt)
	
	return F_OP(2,mbts,[true,true,true],false,size(obt)[1]), pyconvert(Int64, num_elecs)
end

function H_from_xyz(xyz_string; kwargs...)
	#returns fermionic operator of H in orbitals and number of electrons
	h_ferm, num_elecs = xyz_OF_hamiltonian(xyz_string; kwargs...)
	
	Hconst, obt, tbt = fermionic.to_tensors(h_ferm)
	obt = pyconvert(Array{Float64},obt)
	tbt = pyconvert(Array{Float64},tbt)

	mbts = ([pyconvert(Float64, Hconst)], obt, tbt)
	
	return F_OP(2,mbts,[true,true,true],false,size(obt)[1]), pyconvert(Int64, num_elecs)
end

function localized_H_from_xyz(xyz_string; kwargs...)
	#returns fermionic operator of H in orbitals and number of electrons
	
	h_ferm, num_elecs = localized_xyz_OF_hamiltonian(xyz_string; kwargs...)
	
	Hconst, obt, tbt = fermionic.to_tensors(h_ferm)
	obt = pyconvert(Array{Float64},obt)
	tbt = pyconvert(Array{Float64},tbt)

	mbts = ([pyconvert(Float64, Hconst)], obt, tbt)
	
	return F_OP(2,mbts,[true,true,true],false,size(obt)[1]), pyconvert(Int64, num_elecs)
end

function localized_H_from_mol_name(mol_name; kwargs...)
	#returns fermionic operator of H in orbitals and number of electrons
	xyz=ham.chooseType(mol_name, geometries=1)
	xyz=pyconvert(Array{Any}, xyz)
	Hconst, obt, tbt, num_elecs = localized_xyz_OF_hamiltonian(xyz; kwargs...)
	
	
	obt = pyconvert(Array{Float64},obt)
	tbt = 0.5*pyconvert(Array{Float64},tbt)

	mbts = ([pyconvert(Float64, Hconst)], obt, tbt)
	
	return F_OP(2,mbts,[true,true,true],false,size(obt)[end]), pyconvert(Int64, num_elecs)
end


function to_OF(OP :: F_OP)
	#returns OpenFermion Fermionic operator
	if OP.Nbods > 2
		error("Transforming fermionic operators into openfermion not implemented for more than 2-body operators!")
	end
	TOT_OP = of.FermionOperator.zero()
	TOT_OP += OP.mbts[1][1]
	
	for i in 1:OP.Nbods
		if OP.filled[i+1]
			TOT_OP += fermionic.get_ferm_op(OP.mbts[i+1], spin_orb=OP.spin_orb)
		end
	end

	return TOT_OP
end

function to_OF(P :: pauli_word)
	pauli_string = ""
	for i in 1:P.size
		bit_num = pauli_num_from_bit(P.bits[i])
		if bit_num == 1
			pauli_string = pauli_string * " X$(i-1)"
		elseif bit_num == 2
			pauli_string = pauli_string * " Y$(i-1)"
		elseif bit_num == 3
			pauli_string = pauli_string * " Z$(i-1)"
		end
	end

	return P.coeff * of.QubitOperator(pauli_string)
end

function to_OF(Q :: Q_OP)
	of_OP = of.QubitOperator.zero()
	of_OP += Q.id_coeff

	for pw in Q.paulis
		of_OP += to_OF(pw)
	end

	return of_OP
end

function to_OF(OP :: M_OP, transformation = F2Q_map)
	return to_OF(Q_OP(OP, transformation))
end

function to_OF(F :: FRAGMENT; ob_corr = false)
	#ob_corr=true removes one-body correction from fragment, make sure to collect back in one-body tensor
	#returns one-body fragment as-is
	if ob_corr == false || F.TECH == OBF()
		return to_OF(to_OP(F))
	else
		return to_OF(to_OP(F) - ob_correction(F, return_op=true))
	end
end

function from_OF(Hof)
	#returns QuantumMAMBO F_OP from Openfermion FermionOperator
	h0, obt, tbt = fermionic.to_tensors(Hof)
	h0 = pyconvert(Float64, h0)
	obt = pyconvert(Array{Float64}, obt)
	tbt = pyconvert(Array{Float64}, tbt)
	return F_OP(([h0], obt, tbt))
end

function py_sparse_import(py_sparse_mat; imag_tol=1e-14)
	#transform python sparse matrix into julia sparse matrix
	row, col, vals = scipy.sparse.find(py_sparse_mat)
	row = pyconvert(Vector{Int64}, row)
	col = pyconvert(Vector{Int64}, col)
	vals = pyconvert(Vector{Complex{Float64}}, vals)
	py_shape = py_sparse_mat.get_shape()
	n = pyconvert(Int64,py_shape[1])

	if sum(abs.(imag.(vals))) < imag_tol
		vals = real.(vals)
	end

	sparse_mat = sparse(row .+1, col .+1, vals, n, n)

	return sparse_mat
end

function mat_range(sparse_op;ncv=minimum([50,size(sparse_op)[1]]), tol=1e-3)
	#Calculates maximum and minimum eigenvalues for matrix within tolerance tol
	#uses efficient Arpack Krylov-space routine eigs
	if size(sparse_op)[1] >= 4
		E_max,_ = Arpack.eigs(sparse_op, nev=1, which=:LR, maxiter = 500, tol=tol, ncv=ncv)
		E_min,_ = Arpack.eigs(sparse_op, nev=1, which=:SR, maxiter = 500, tol=tol, ncv=ncv)
	else
		E,_ = eigen(collect(sparse_op))
		E = real.(E)
		E_max = maximum(E)
		E_min = minimum(E)
	end
	E_range = real.([E_min[1], E_max[1]])
	
	return E_range
end

function OF_qubit_op_range(op_qubit, n_qubit=pyconvert(Int64,of.count_qubits(op_qubit)); imag_tol=1e-14, ncv=minimum([50,2^n_qubit]), tol=1e-3, debug=false)
	#Calculates maximum and minimum eigenvalues for qubit operator within tolerance tol
	#uses efficient Arpack Krylov-space routine eigs
	op_py_sparse_mat = of.qubit_operator_sparse(op_qubit)
	sparse_op = py_sparse_import(op_py_sparse_mat, imag_tol=imag_tol)

	if debug
		E, _ = eigen(collect(sparse_op))
		E = real.(E)
	
		of_eigen = of.eigenspectrum(op_qubit)
		@show minimum(E), minimum(of_eigen)
		@show maximum(E), maximum(of_eigen)
	end	

	return mat_range(sparse_op, ncv=ncv, tol=tol, debug=debug)
end

function OF_to_F_OP(H, spin_orb=false)
	#H should be an openfermion FermionOperator object
	h0_py, obt_py, tbt_py = fermionic.to_tensors(H, spin_orb=spin_orb)
	h0 = pyconvert(Float64, h0_py)
	obt = pyconvert(Array{Float64}, obt_py)
	tbt = pyconvert(Array{Float64}, tbt_py)
	return F_OP(([h0], obt, tbt), spin_orb)
end
