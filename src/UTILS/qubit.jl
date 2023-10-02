# qubit utils
function Q_OP(F :: F_OP, transformation = F2Q_map, tol = PAULI_TOL)
	return Q_OP(M_OP(F), transformation)
end

function Q_OP(frag :: F_FRAG, transformation = F2Q_map, tol = PAULI_TOL)
	return  Q_OP(F_OP(frag), transformation, tol)
end

function majorana_pair_to_pauli(i, j, σ, n_qubits, transformation = F2Q_map)
	#transforms γiσ0*γjσ1 into Pauli word
	#σ ∈ {0,1}, corresponds to σ ∈ {false, true}
	if transformation != "jw" && transformation != "jordan-wigner"
		error("γiσ0*γjσ1 to Pauli word transformation not defined for $transformation!")
	end

	α = σ - 1
	bin_vec = zeros(Bool, 2*n_qubits)
	phase = -1im
	if i < j
		bin_vec[2i+α] = 1
		for n in 2i+α+1:2j+α-1
			bin_vec[n+n_qubits] = 1
		end
		bin_vec[2j+α] = 1
	elseif j < i
		bin_vec[2j+α] = 1
		bin_vec[2j+α + n_qubits] = 1
		for n in 2j+α+1:2i+α-1
			bin_vec[n+n_qubits] = 1
		end
		bin_vec[2i+α] = 1
		bin_vec[2i+α+n_qubits] = 1
	else
		bin_vec[2i+α+n_qubits] = 1
		phase *= -1
	end

	pw = pauli_word(bin_vec, phase)
	return pw
end

function single_majorana_to_pauli(p, m, n_qubits, transformation = F2Q_map)
	# p: spin-orbital index
	# m = 0,1 for different Majorana type
	if transformation != "jw" && transformation != "jordan-wigner"
		error("γpm to Pauli word transformation not defined for $transformation!")
	end

	bin_vec = zeros(Bool, 2*n_qubits)

	if m == 0
		for i in 1:p-1
			bin_vec[i+n_qubits] = 1
		end
		bin_vec[p] = 1
	else
		for i in 1:p-1
			bin_vec[i+n_qubits] = 1
		end
		bin_vec[p] = 1
		bin_vec[p+n_qubits] = 1
	end

	return pauli_word(bin_vec, 1)
end

function Q_OP(M :: M_OP, transformation = F2Q_map, tol = PAULI_TOL)
	if M.spin_orb
		N = M.N
	else
		N = 2*M.N #number of qubits
	end

	if M.body_sym == false
		error("Trying to convert Majorana without body-symmetry (i.e. only has γp0*γq1 products) to Qubit operator, not implemented!")
	end
	tol2 = tol^2

	n_paulis = Int(N^2 + N^4 + (N^2)*((N-1)^2)) #upper-bound for total number of Pauli words
	pws = [pw_zero(N) for i in 1:n_paulis]

	#identity term
	id_coeff = M.mbts[1][1] * M.t_coeffs[1]

	curr_pauli = 1
	
	#one-body terms
	if M.Nmajs ≥ 2
		if M.filled[2] #single Majorana terms
			error("Filled 1-majorana words not compatible with body_sym = true!")
		end

		if M.filled[3]
			if M.spin_orb == false
				for i in 1:M.N
					for j in 1:M.N
						ck = M.t_coeffs[3] * M.mbts[3][i,j]
						if abs2(ck) > tol2
							for σ in 0:1
								pws[curr_pauli] = ck * majorana_pair_to_pauli(i, j, σ, N, transformation)
								curr_pauli += 1
							end
						end
					end
				end
			else
				for i in 1:M.N
					for j in 1:M.N
						ck = M.t_coeffs[3] * M.mbts[3][i,j]
						if abs2(ck) > tol2
							pwi = single_majorana_to_pauli(i, 0, N)
							pwj = single_majorana_to_pauli(j, 1, N)
							pws[curr_pauli] = ck * pwi * pwj
							curr_pauli += 1
						end
					end
				end
			end
		end
	end

	#two-body terms
	if M.Nmajs ≥ 4
		if M.filled[4] #single Majorana terms
			error("Filled 3-majorana words not compatible with body_sym = true!")
		end

		if M.filled[5]
			if M.spin_orb == false
				for i in 1:M.N
					for j in 1:M.N
						for k in 1:M.N
							for l in 1:M.N
								ck = M.t_coeffs[5] * M.mbts[5][i,j,k,l]
								if abs2(ck) > tol2
									for α in 0:1
										β = mod(α-1, 2)
										pw1 = majorana_pair_to_pauli(i, j, α, N, transformation)
										pw2 = majorana_pair_to_pauli(k, l, β, N, transformation)
										pws[curr_pauli] = ck * pw1 * pw2
										curr_pauli += 1
									end
								end
							end
						end
					end
				end
				for i in 1:M.N
					for l in 1:M.N
						for k in 1:i-1
							for j in 1:l-1
								tbt_iso_ijkl = 2*(M.mbts[5][i,j,k,l] - M.mbts[5][i,l,k,j])
								ck = M.t_coeffs[5] * tbt_iso_ijkl
								if abs2(ck) > tol2
									for α in 0:1
										pw1 = majorana_pair_to_pauli(i, j, α, N, transformation)
										pw2 = majorana_pair_to_pauli(k, l, α, N, transformation)
										pws[curr_pauli] = ck * pw1 * pw2
										curr_pauli += 1
									end
								end
							end
						end
					end
				end
			else
				for i in 1:M.N
					for j in 1:M.N
						for k in 1:M.N
							for l in 1:M.N
								ck = M.t_coeffs[5] * M.mbts[5][i,j,k,l]
								if abs2(ck) > tol2
									pwi = single_majorana_to_pauli(i, 0, N)
									pwj = single_majorana_to_pauli(j, 1, N)
									pwk = single_majorana_to_pauli(k, 0, N)
									pwl = single_majorana_to_pauli(l, 1, N)
									pws[curr_pauli] = ck * pwi * pwj * pwk * pwl
									curr_pauli += 1
								end
							end
						end
					end
				end
			end
		end
	end

	n_paulis = curr_pauli-1

	if M.spin_orb == false
		return Q_OP(N, n_paulis, id_coeff, pws[1:n_paulis])
	else
		return simplify(Q_OP(N, n_paulis, id_coeff, pws[1:n_paulis]))
	end
end

function AC_group(Q :: Q_OP; ret_ops = false, verbose = false)
	group_arrs = Array{Int64, 1}[]
	vals_arrs = Array{Complex,1}[]

	vals_ord = [Q.paulis[i].coeff for i in 1:Q.n_paulis]
	ind_perm = sortperm(abs.(vals_ord))[end:-1:1]
	vals_ord = vals_ord[ind_perm]

	for i in 1:Q.n_paulis
		is_grouped = false
		for (grp_num,grp) in enumerate(group_arrs)
			antic_w_group = true
			for j in grp
				if pws_is_anticommuting(Q.paulis[ind_perm[i]],Q.paulis[ind_perm[j]]) == 0
					antic_w_group = false
					break
				end
			end
			if antic_w_group == true
				push!(grp, i)
				push!(vals_arrs[grp_num], vals_ord[i])
				is_grouped = true
				break
			end
		end
		if is_grouped == false
			push!(group_arrs, [i])
			push!(vals_arrs, Complex[vals_ord[i]])
		end
	end

	num_groups = length(group_arrs)
    group_L1 = zeros(num_groups)
    for i in 1:num_groups
        for val in vals_arrs[i]
            group_L1[i] += abs2(val)
        end
    end

    L1_norm = sum(sqrt.(group_L1))

    if ret_ops == false
    	return L1_norm, num_groups
    else
    	OPS = Q_OP[]
    	for i in 1:num_groups
    		pws_i = [Q.paulis[ind_perm[j]] for j in group_arrs[i]]
    		q_op = Q_OP(Q.N, length(pws_i), 0, pws_i)
    		push!(OPS, q_op)
    	end

    	return sqrt.(group_L1), OPS
    end
end

function AC_group(F :: F_OP; ret_ops = false, verbose=false)
	return AC_group(Q_OP(F), ret_ops=ret_ops, verbose=verbose)
end

function vectorized_AC_group(Q :: Q_OP; ret_ops = false, verbose = false)
	group_arrs = Array{Int64, 1}[]
	vals_arrs = Array{Complex,1}[]
	bin_mats = Array{Bool,2}[] #holds bra vectors of groups for fast anticommutativity check

	vals_ord = [Q.paulis[i].coeff for i in 1:Q.n_paulis]
	ind_perm = sortperm(abs.(vals_ord))[end:-1:1]
	vals_ord = vals_ord[ind_perm]

	for i in 1:Q.n_paulis
		my_bin = pw_to_bin_vec(Q.paulis[ind_perm[i]])
		is_grouped = false
		for (grp_num,grp) in enumerate(group_arrs)
			grp_prods = Bool.((bin_mats[grp_num] * my_bin) .% 2)
			if prod(grp_prods) == true
				push!(grp, i)
				push!(vals_arrs[grp_num], vals_ord[i])
				bin_mats[grp_num] = vcat(bin_mats[grp_num], pw_to_bin_bra(Q.paulis[ind_perm[i]]))
				is_grouped = true
				break
			end
		end
		if is_grouped == false
			push!(group_arrs, [i])
			push!(vals_arrs, Complex[vals_ord[i]])
			push!(bin_mats, pw_to_bin_bra(Q.paulis[ind_perm[i]]))
		end
	end

	num_groups = length(group_arrs)
    group_L1 = zeros(num_groups)
    for i in 1:num_groups
        for val in vals_arrs[i]
            group_L1[i] += abs2(val)
        end
    end

    L1_norm = sum(sqrt.(group_L1))

    if ret_ops == false
    	return L1_norm, num_groups
    else
    	OPS = Q_OP[]
    	for i in 1:num_groups
    		pws_i = [Q.paulis[ind_perm[j]] for j in group_arrs[i]]
    		q_op = Q_OP(Q.N, length(pws_i), 0, pws_i)
    		push!(OPS, q_op)
    	end

    	return sqrt.(group_L1), OPS
    end
end

function vectorized_AC_group(F :: F_OP; ret_ops = false, verbose=false)
	return AC_group(Q_OP(F), ret_ops=ret_ops, verbose=verbose)
end

function FC_group(Q :: Q_OP; ret_ops = false, verbose=false)
	is_grouped = zeros(Bool, Q.n_paulis)
	group_arrs = Array{Int64,1}[]
	vals_arrs = Array{Complex,1}[]

	vals_ord = [Q.paulis[i].coeff for i in 1:Q.n_paulis]
	ind_perm = sortperm(abs.(vals_ord))[end:-1:1]
	vals_ord = vals_ord[ind_perm]

	if verbose
		println("Running sorting-insertion algorithm")
		@show sum(vals_ord)
	end

	for i in 1:Q.n_paulis
    	if is_grouped[i] == false
    		curr_group = [i]
    		curr_vals = [vals_ord[i]]
    		is_grouped[i] = true
    		for j in i+1:Q.n_paulis
    			if is_grouped[j] == false
	    			if pws_is_anticommuting(Q.paulis[ind_perm[i]],Q.paulis[ind_perm[j]]) == 0
	    				c_w_group = true
	    				for k in curr_group[2:end]
	    					if pws_is_anticommuting(Q.paulis[ind_perm[k]],Q.paulis[ind_perm[j]]) == 1
		    					c_w_group = false
		    					break
		    				end
	    				end

	    				if c_w_group == true
		    				push!(curr_group,j)
		    				push!(curr_vals,vals_ord[j])
		    				is_grouped[j] = true
		    			end
	    			end
	    		end
	    	end
    		push!(group_arrs,curr_group)
    		push!(vals_arrs, curr_vals)
    	end
    end

    if prod(is_grouped) == 0
    	println("Error, not all terms are grouped after FC-SI algorithm!")
    	@show is_grouped
    end

    num_groups = length(group_arrs)
    group_L1 = zeros(num_groups)
    for i in 1:num_groups
        for val in vals_arrs[i]
            group_L1[i] += abs2(val)
        end
    end

    L1_norm = sum(sqrt.(group_L1))

    if ret_ops == false
    	return L1_norm, num_groups
    else
    	OPS = Q_OP[]
    	for i in 1:num_groups
    		pws_i = [Q.paulis[ind_perm[j]] for j in group_arrs[i]]
    		q_op = Q_OP(Q.N, length(pws_i), 0, pws_i)
    		push!(OPS, q_op)
    	end

    	return L1_norm, OPS
    end
end

function FC_group(F :: F_OP; ret_ops = false, verbose=false)
	return FC_group(Q_OP(F), ret_ops=ret_ops, verbose=verbose)
end

function simplify(Q :: Q_OP)
	ID = pauli_word(zeros(Bool, 2*Q.N))
	id_coeff = Q.id_coeff
	pws = pauli_word[]
	for i in 1:Q.n_paulis
		pw_curr = Q.paulis[i]
		already_included = false
		if pw_curr.bits == ID.bits
			already_included = true
			id_coeff += pw_curr.coeff
		end
		if already_included == false
			for counted_pw in pws
				if pw_curr.bits == counted_pw.bits
					already_included = true
					counted_pw.coeff += pw_curr.coeff
					break
				end
			end
		end
		if !already_included
			push!(pws, copy(pw_curr))
		end
	end

	for i in length(pws):-1:1
		if pws[i].coeff == 0
			deleteat!(pws, i)
		end
	end

	return Q_OP(Q.N, length(pws), id_coeff, pws)
end

import Base.+

function +(Q1 :: Q_OP, Q2 :: Q_OP)
	pws_1 = copy(Q1.paulis)
	pws_2 = copy(Q2.paulis)
	Nmax = maximum([Q1.N, Q2.N])

	if Q1.N != Q2.N
		if Q1.N == Nmax
			pws_1 = augment_qubits.(pws_1, Nmax)
		else
			pws_2 = augment_qubits.(pws_2, Nmax)
		end
	end

	pws = [pws_1..., pws_2...]

	Qsum = simplify(Q_OP(Nmax, length(pws), Q1.id_coeff + Q2.id_coeff, pws))

	return Qsum
end

import Base.*

function *(Q :: Q_OP, c :: Number)
	return Q_OP(Q.N, Q.n_paulis, Q.id_coeff*c, c .* Q.paulis)
end

function *(c :: Number,  Q :: Q_OP)
	return Q_OP(Q.N, Q.n_paulis, Q.id_coeff*c, c .* Q.paulis)
end

function *(Q1 :: Q_OP, Q2 :: Q_OP)
	pws_prod = pauli_word[]
	pws_1 = copy(Q1.paulis)
	pws_2 = copy(Q2.paulis)
	
	Nmax = maximum([Q1.N, Q2.N])
	if Q1.N != Q2.N
		if Q1.N > Q2.N
			pws_2 = augment_qubits.(pws_2, Nmax)
		else
			pws_1 = augment_qubits.(pws_1, Nmax)
		end
	end


	id_tot = Q1.id_coeff  * Q2.id_coeff

	for pw1 in pws_1
		push!(pws_prod, Q1.id_coeff*pw1)
	end

	for pw2 in pws_2
		push!(pws_prod, Q2.id_coeff*pw2)
	end

	for pw1 in pws_1
		for pw2 in pws_2
			push!(pws_prod, pw1*pw2)
		end
	end

	Qprod = simplify(Q_OP(Nmax, length(pws_prod), id_tot, pws_prod))

	return Qprod
end

import Base.-

-(Q1 :: Q_OP, Q2 :: Q_OP) = Q1 + (-1*Q2)

function commutator(Q1 :: Q_OP, Q2 :: Q_OP)
	return Q1*Q2 - Q2*Q1
end

#= DEPRECATED IMPLEMENTATION, REQUIRES COSTLY TYPE CONVERSIONS. MORE EFFICIENT IMPLEMENTATION USES OPENFERMION
function to_matrix(Q :: Q_OP; do_sparse = true)
	mat = Q.id_coeff * sparse(Diagonal(ones(Complex,2^(Q.N)))) |> SparseMatrixCSC{Complex, Int64}
	
	for i in 1:Q.n_paulis
		mat = mat + to_matrix(Q.paulis[i]) |> SparseMatrixCSC{Complex, Int64}
	end

	if do_sparse
		return mat
	else
		return collect(mat)
	end
end
# =#
function to_matrix(Q :: Q_OP)
	return py_sparse_import(of.get_sparse_operator(to_OF(Q), n_qubits = Q.N))
end

function to_matrix(F :: F_OP)
	return to_matrix(Q_OP(F))
end

function to_matrix(frag :: F_FRAG)
	return to_matrix(Q_OP(frag))
end

function sparse_matrix_commutator(mat1 :: SparseMatrixCSC, mat2 :: SparseMatrixCSC)
	m12 = mat1*mat2 |> SparseMatrixCSC{Complex, Int64}
	m21 = mat2*mat1 |> SparseMatrixCSC{Complex, Int64}

	return m12 - m21 |> SparseMatrixCSC{Complex, Int64}
end

function sparse_range(sp_mat :: SparseMatrixCSC; ncv = minimum([50, sp_mat.m]), tol=1e-3, imag_tol = 1e-8)
	if sp_mat.m >= 4
		E_max,_ = eigs(sp_mat, nev=1, which=:LR, maxiter = 500, tol=tol, ncv=ncv)
		E_min,_ = eigs(sp_mat, nev=1, which=:SR, maxiter = 500, tol=tol, ncv=ncv)
	else
		E,_ = eigen(collect(sp_mat))
		E = real.(E)
		E_max = maximum(E)
		E_min = minimum(E)
	end

	E_range = real.([E_min[1], E_max[1]])

	if sum(imag.([E_min[1], E_max[1]])) > imag_tol
		@warn "Imaginary values found inside range calculation, eigenvalues will be calculated with full diagonalization and absolute values range will be returned..."
		E,_ = eigen(collect(sp_mat))
		E = abs.(E)
		return [minimum(E), maximum(E)]
	end


	return E_range
end

function sparse_matrix_commutator_range(mat1 :: SparseMatrixCSC, mat2 :: SparseMatrixCSC; tol=1e-3)
	return sparse_range(sparse_matrix_commutator(mat1, mat2), tol=tol)
end

function Q_OP_range(Q :: Q_OP; ncv=minimum([50,2^(Q.N)]), tol=1e-3, debug=false)
	sp_mat = to_matrix(Q, do_sparse=true)

	E_range = sparse_range(sp_mat, ncv=ncv, tol=tol)

	if debug
		E, _ = eigen(collect(sp_mat))
		E = abs.(E)
	
		of_eigen = of.eigenspectrum(to_OF(Q))
		@show minimum(E), minimum(of_eigen)
		@show maximum(E), maximum(of_eigen)
	end	

	return E_range
end


