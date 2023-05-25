#FUNCTIONS FOR INTERFACING WITH PYTHON
PY_DIR = readchomp(`which python`)
if @isdefined myid
	if myid() == 1
		println("Using python installation in $PY_DIR")
	end
else
	println("Using python installation in $PY_DIR")
end

#use python installation from which Julia session is being ran. Uses micromamba environment by default
ENV["JULIA_CONDAPKG_BACKEND"] = "MicroMamba"
#= Uncomment following lines for using default python installation instead
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = PY_DIR
# =#
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

function obtain_OF_hamiltonian(mol_name; basis="sto3g", ferm=true, geometry=1)
	return ham.get_system(mol_name,ferm=ferm,basis=basis,geometry=geometry,n_elec=true)		
end

function obtain_H(mol_name; basis="sto3g", ferm=true, geometry=1)
	#returns fermionic operator of H in orbitals and number of electrons
	h_ferm, num_elecs = obtain_OF_hamiltonian(mol_name, basis=basis, ferm=ferm, geometry=geometry)
	
	tbt = pyconvert(Array{Float64}, fermionic.get_chemist_tbt(h_ferm, spin_orb=false))
	h1b = h_ferm - fermionic.get_ferm_op(tbt, false)
    h1b = of_simplify(h1b)
	obt = pyconvert(Array{Float64}, fermionic.get_obt(h1b, spin_orb=false))

	mbts = ([pyconvert(Float64, h_ferm.constant)], obt, tbt)
	
	return F_OP(2,mbts,[true,true,true],false,size(obt)[1]), pyconvert(Int64, num_elecs)
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

function OF_qubit_op_range(op_qubit, n_qubit=pyconvert(Int64,of.count_qubits(op_qubit)); imag_tol=1e-14, ncv=minimum([50,2^n_qubit]), tol=1e-3, debug=false)
	#Calculates maximum and minimum eigenvalues for qubit operator within tolerance tol
	#uses efficient Arpack Krylov-space routine eigs
	op_py_sparse_mat = of.qubit_operator_sparse(op_qubit)
	sparse_op = py_sparse_import(op_py_sparse_mat, imag_tol=imag_tol)

	if n_qubit >= 2
		E_max,_ = eigs(sparse_op, nev=1, which=:LR, maxiter = 500, tol=tol, ncv=ncv)
		E_min,_ = eigs(sparse_op, nev=1, which=:SR, maxiter = 500, tol=tol, ncv=ncv)
	else
		E,_ = eigen(collect(sparse_op))
		E = real.(E)
		E_max = maximum(E)
		E_min = minimum(E)
	end
	E_range = real.([E_min[1], E_max[1]])
	
	if debug
		E, _ = eigen(collect(sparse_op))
		E = real.(E)
	
		of_eigen = of.eigenspectrum(op_qubit)
		@show minimum(E), minimum(of_eigen)
		@show maximum(E), maximum(of_eigen)
	end	
	
	return E_range
end