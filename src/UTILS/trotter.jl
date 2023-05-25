#utilities for Trotter

function trotter_α(FRAGS :: Array{F_FRAG};  tol=1e-5)
	#calculates commutator norm ∑_{nm} ||[Hn,Hm]||
	#must include 1-body fragment as well
	M = length(FRAGS)
	α = 0.0
	
	for i in 1:M
		op_i = qubit_transform(to_OF(FRAGS[i]))
		for j in 1:i-1
			op_j = qubit_transform(to_OF(FRAGS[j]))
			comm_range = OF_qubit_op_range(-1im*of.commutator(op_i,op_j), tol=tol)
			comm_sparse = sparse_matrix_commutator_range(-1im*to_matrix(FRAGS[i]), to_matrix(FRAGS[j]), tol=tol)
			@show comm_range
			@show comm_sparse
			α += 2*maximum(abs.(comm_range))
		end
	end

	return α
end

function trotter_β(FRAGS :: Array{F_FRAG})
	M = length(FRAGS)
	β = 0.0

	Δs = zeros(M)
	for i in 1:M
		curr_frag = FRAGS[i]
		Δs[i] = 2*SQRT_L1(curr_frag, count=false)
	end

	for i in 1:M
		for j in 1:i-1
			β += Δs[i] * Δs[j]
		end
	end

	return β
end

function trotter_exp(MATS, t)
	exp_t = exp(-1im*t*MATS[1])
	for mat in MATS[2:end]
		exp_t *= exp(-1im*t*mat)
	end

	return exp_t
end

function trotter_comparer(H :: F_OP, η, t = 1)
	DF_FRAGS = DF_decomposition(H, verbose=false)
	corr_frags = to_OP.(DF_FRAGS) - ob_correction.(DF_FRAGS, return_op = true)
	mats = to_matrix.(corr_frags)
	obmat = to_matrix(to_OBF(H.mbts[2] + ob_correction(H)))
	push!(mats, obmat)
	Uη = Ne_block_diagonalizer(H.N, η)
	mats_sym = []
	for i in 1:length(mats)
		push!(mats_sym, matrix_symmetry_block(mats[i], Uη))
	end
	push!(mats_sym, Diagonal(H.mbts[1][1] * ones(size(mats_sym[1])[1])))
	Utrotter = trotter_exp(mats_sym, t)
	Hmat = matrix_symmetry_block(to_matrix(H), Uη)
	Uexact = exp(-1im*t*Hmat)

	abs2_err = sum(abs2.(Utrotter - Uexact))
	E,_ = eigen(Utrotter - Uexact)

	spec_err = maximum(abs.(E))

	return abs2_err, spec_err
end

function trotter_comparer(Hmat, mats, t = 1e-5)
	Utrotter = trotter_exp(mats, t)
	Uexact = exp(-1im*t*Hmat)

	abs2_err = sum(abs2.(Utrotter - Uexact))
	E,_ = eigen(Utrotter - Uexact)

	spec_err = maximum(abs.(E))

	return abs2_err, spec_err
end