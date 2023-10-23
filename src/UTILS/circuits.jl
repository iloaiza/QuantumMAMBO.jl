#circuit construction tools, interfaces with circuits.py which uses cirq
#requires adding py_utils.jl first
circuits = pyimport("circuits")
cirq = pyimport("cirq")
cft = pyimport("cirq_ft")

function Pauli_circuit(H :: F_OP; epsilon = 1e-5, trunc_thresh = 1e-5)
	#=
	Creates Pauli (or Sparse) LCU oracles

	inputs:
		- H: fermionic operator in QuantumMAMBO format
		- epsilon: accuracy for state preparation coefficients
		- trunc_thresh: truncation threshold for zero-ing terms beneath this value
	=#

	if H.spin_orb
		error("Not implemented for spin-orbitals!")
	end

	obt = H.mbts[2]
	tbt = H.mbts[3]

	prep = circuits.Prepare_Sparse_SOTA.build(obt, tbt, probability_epsilon = epsilon, truncation_threshold = trunc_thresh)
	i_reg = prep.i_register
	j_reg = prep.j_register
	k_reg = prep.k_register
	l_reg = prep.l_register
	V_reg = prep.V_register
	s1_reg = prep.s1_register
	s2_reg = prep.s2_register

	ctl_reg = cft.Registers.build(control = 1)

	sel_circ = sparse_select(V_reg, i_reg, j_reg, k_reg, l_reg, s1_reg, s2_reg, ctl_reg)

	prep_circ = circuits.to_circuit(prep)

	println("Warning, oracle circuit does not implement inverse of PREP, good for gate count but bad for circuit")
	tot_circ = prep_circ + sel_circ + prep_circ

	t_count = cft.t_complexity(tot_circ)
	tot_qubits = length(cirq.Circuit(cirq.decompose(tot_circ)).all_qubits())

	return t_count, tot_qubits
end

function AC_circuit(H :: F_OP; epsilon = 1e-5, givens_eps = 1e-4)
	#=
	Creates anticommputing (AC) LCU oracles

	inputs:
		- H: fermionic operator in QuantumMAMBO format
		- epsilon: accuracy for state preparation coefficients
		- givens_eps: accuracy for givens rotations
	=#

	if H.spin_orb
		error("Not implemented for spin-orbitals!")
	end

	AC_coeffs, AC_ops = AC_group(H, ret_ops = true)
	@show PAULI_L1(Q_OP(H) - sum(AC_ops))
	@show PAULI_L1(Q_OP(H) - sum(AC_coeffs .* AC_ops))

	ac_groups = []
	n_qubits = AC_ops[1].N

	for i in 1:length(AC_coeffs)
		group_paulis = AC_ops[i].paulis
		group_len = length(group_paulis)
		int_arr = zeros(Int, n_qubits, group_len)
		coeffs = zeros(Complex, group_len)
		
		for (j,pw) in enumerate(group_paulis)
			coeffs[j] = pw.coeff
			int_arr[:,j] = pw_to_int_vec(pw)
		end

		push!(ac_groups, (coeffs, int_arr))
	end

	prep = circuits.Prepare_AC(ac_groups, probability_epsilon = epsilon)
	n_reg = prep.n_register

	sel = Select_AC(ac_groups, n_reg = n_reg)

	t_count = 2 * cft.t_complexity(prep) + cft.t_complexity(sel)

	println("Warning, oracle circuit does not implement inverse of PREP, good for gate count but bad for circuit")
	tot_circ = circuits.to_circuit(prep) + circuits.to_circuit(sel) + circuits.to_circuit(prep)
	tot_qubits = length(cirq.Circuit(cirq.decompose(tot_circ)).all_qubits())

	return t_count, tot_qubits
end

function U_to_Givens(U :: F_UNITARY, k :: Int64)
	#returns {θi} angles specifying Givens rotations G = Π_{i=1}^{N-1} exp(θi γ_i γ_{i+1}) for
	#G^dagger * γ_1 * G = U^dagger * γ_k * U

	coeffs = one_body_unitary(U)[:,k]

	givens = majorana_coefs_to_unitary(coeffs)

	return givens.θs
end


function DF_circuit(H :: F_OP; epsilon = 1e-5, DF_tol = SVD_tol)
	#=
	Creates DF LCU oracles

	inputs:
		- H: fermionic Hamiltonian in QuantumMAMBO format
		- epsilon: accuracy for prepare coefficients
		- DF_tol: cut-off for SVD coefficients in DF
	=#
	if H.spin_orb
		error("Not implemented for spin-orbitals!")
	end

	DF_frags = DF_decomposition(H, tol = DF_tol)
	obt = H.mbts[2] + ob_correction(H)

	num_frags = length(DF_frags)
	num_orbs = size(obt)[1]
	mus_mat = zeros(num_frags+1, num_orbs)
	thetas_tsr = zeros(num_frags+1, num_orbs, num_orbs-1)

	obt_frag = to_OBF(obt)
	mus_mat[1, :] .= obt_frag.C.λ
	for k in 1:num_orbs
		thetas_tsr[1, k, :] = U_to_Givens(obt_frag.U[1], k)
	end

	for l in 1:num_frags
		mus_mat[l+1, L] .= DF_frags[l].C.λ
		for k in 1:num_orbs
			thetas_tsr[l+1, k, :] = U_to_Givens(DF_frags[l].U[1], k)
		end
	end

	prep = circuits.DF_Prepare(mus_mat, probability_epsilon = epsilon)
	prep_inv = circuits.DF_Prepare(mus_mat, probability_epsilon = epsilon, dagger = true)
	sel = circuits.DF_Select(prep.l_register, prep.l_not_0_register, mus_mat, thetas_tsr)

	t_count = cft.t_complexity(prep) + cft.t_complexity(prep_inv) + cft.t_complexity(sel)
	tot_circ = circuits.to_circuit(prep) + circuits.to_circuit(sel) + circuits.to_circuit(prep_inv)

	tot_qubits = length(cirq.Circuit(cirq.decompose(tot_circ)).all_qubits())

	return t_count, tot_qubits
end


