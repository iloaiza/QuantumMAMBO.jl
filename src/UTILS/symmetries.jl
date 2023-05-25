function symmetry_builder(H :: F_OP)
	#returns Ne and Neˆ2, both symmetries can be represented in orbitals (so no need for spin-orbitals)
	Ne_obt = zeros(H.N, H.N)
	for i in 1:H.N
		Ne_obt[i,i] = 1
	end

	Ne2_tbt = zeros(H.N, H.N, H.N, H.N)
	for i in 1:H.N
		for j in 1:H.N
			Ne2_tbt[i,i,j,j] = 1
		end
	end

	return F_OP(([0],Ne_obt), H.spin_orb), F_OP(([0],[0],Ne2_tbt), H.spin_orb)
end

function Ne_builder(N :: Int64, spin_orb=true)
	#returns Ne and Neˆ2, both symmetries can be represented in orbitals (so no need for spin-orbitals)
	Ne_obt = zeros(N, N)
	for i in 1:N
		Ne_obt[i,i] = 1
	end

	return F_OP(([0],Ne_obt), spin_orb)
end

function Ne2_lambda_builder(N :: Int64)
	#returns Ne^2-associated λ vector, such that cartan_2b(λ) = Ne^2
	λ = zeros(Int(N*(N+1)/2))

	idx = 1
	for i in 1:N
		for j in 1:i
			if i == j
				λ[idx] = 1
			else
				λ[idx] = 2
			end
			idx += 1
		end
	end

	return λ
end


function naive_tb_symmetry_shift(H :: F_OP)
	_,Ne2 = symmetry_builder(H)

	function cost(x)
		return L2_partial_cost(H, x[1] * Ne2)
	end
	println("L2 cost before two-body symmetry shift is $(cost(0))")

	x0 = [0.0]
	@time sol = optimize(cost, x0, BFGS())
	println("Final L2 cost after two-body symmetry shift is $(sol.minimum)")

	x2 = sol.minimizer[1]

	return x2, H - x2*Ne2
end

function naive_ob_symmetry_shift(H :: F_OP; verbose=true)
	Ne,_ = symmetry_builder(H)

	D,_ = eigen(H.mbts[2])
	function cost(x)
		return sum(abs.(D - x[1]*ones(H.N)))
	end

	if verbose
		println("L1 cost before one-body symmetry shift is $(sum(abs.(D)))")
	end
	x0 = [0.0]
	@time sol = optimize(cost, x0, BFGS())
	if verbose
		println("Final L1 cost after one-body symmetry shift is $(sol.minimum)")
	end

	x1 = sol.minimizer[1]

	return x1, H - x1*Ne
end

function symmetry_treatment(H :: F_OP; verbose=true, SAVENAME=DATAFOLDER*"SYM.h5", SAVELOAD=SAVING)
	if verbose
		println("Starting total L1 norm for Hamiltonian is $(PAULI_L1(H))")
	end
	Ne, Ne2 = symmetry_builder(H)

	if SAVELOAD
		fid = h5open(SAVENAME, "cw")
		if haskey(fid, "SYMMETRY_SHIFT")
			println("Found symmetry-shift saved data in $SAVENAME")
			SS_group = fid["SYMMETRY_SHIFT"]
			x2, x1 = read(SS_group, "shifts")
			close(fid)
			H2 = H - x2*Ne2
			H_sym = H2 - x1*Ne
			return H_sym, [x2,x1]
		end
		close(fid)
	end

	#x2, H2 = naive_tb_symmetry_shift(H)
	# =
	τ_mat = τ_mat_builder([Ne2.mbts[3]])
	x2 = L1_linprog_optimizer_tbt(H.mbts[3], τ_mat)[1]
	H2 = H - x2*Ne2
	# =#

	#run one-body optimization routine over corrected one-body tensor
	obt_corr = ob_correction(H2, return_op=true)
	H2 += obt_corr
	D,_ = eigen(H2.mbts[2])
	#x1, H_sym = naive_ob_symmetry_shift(H2, verbose=verbose)
	# =#
	x1 = L1_linprog_one_body_Ne(H2.mbts[2])
	H_sym = H2 - x1*Ne

	# =#

	if verbose
		println("Final total L1 norm for symmetry-shifted Hamiltonian is $(PAULI_L1(H_sym))")
	end

	if SAVELOAD
		fid = h5open(SAVENAME, "cw")
		create_group(fid, "SYMMETRY_SHIFT")
		SS_group = fid["SYMMETRY_SHIFT"]
		SS_group["shifts"] = [x2, x1]
		close(fid)
	end

	return H_sym - obt_corr, [x2, x1]
end

