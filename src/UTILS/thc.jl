#=
function of_thc(eri, nthc)
	ERI_THC, UMATS, ζ, INFO = of_thc.factorize(eri, nthc)
	
	@show sum(abs2.(eri - ERI_THC))
end
# =#

function THC_search(F :: F_OP, hop_num = 20; tol=ϵ, rank_max = 1000, verbose=true, RAND_START = false)
	if verbose
		println("Starting THC binary search routine...")
	end
	OB_ERI, TB_ERI = F_OP_to_eri(F)

	curr_cost = sum(abs2.(TB_ERI))
	curr_rank = 1

	ERI_THC = UMATS = ζ = 0

	while curr_rank < rank_max && curr_cost > tol
		curr_rank += hop_num
		if verbose
			println("Starting search for rank=$curr_rank")
			@time ERI_THC, UMATS, ζ, INFO = of_thc.factorize(TB_ERI, curr_rank, random_start_thc=RAND_START)
		else
			ERI_THC, UMATS, ζ, INFO = of_thc.factorize(TB_ERI, curr_rank, random_start_thc=RAND_START)
		end
		curr_cost = sum(abs2.(TB_ERI - ERI_THC))
		if verbose
			println("Current cost is $curr_cost")
		end
	end

	if verbose
		println("Finished hopped search for rank $curr_rank, final cost is $curr_cost")
	end

	if curr_cost > tol
		error("THC decomposition did not converge for rank=$curr_rank, try increasing maximum rank")
	end

	rank_min = maximum([1, curr_rank - hop_num+1])

	ERI_OLD = copy(ERI_THC)
	ζ_OLD = copy(ζ)
	UMATS_OLD = copy(UMATS)
	while curr_cost < tol && curr_rank > rank_min
		curr_rank -= 1
		if verbose
			println("Starting search for rank=$curr_rank")
			@time ERI_THC, UMATS, ζ, INFO = of_thc.factorize(TB_ERI, curr_rank, random_start_thc=RAND_START)
		else
			ERI_THC, UMATS, ζ, INFO = of_thc.factorize(TB_ERI, curr_rank, random_start_thc=RAND_START)
		end
		curr_cost = sum(abs2.(TB_ERI - ERI_THC))
		if verbose
			println("Current cost is $curr_cost")
		end
		if curr_cost < tol
			ERI_OLD = ERI_THC
			ζ_OLD = ζ
			UMATS_OLD = UMATS
		else
			curr_rank += 1
		end
	end

	fin_cost = sum(abs2.(TB_ERI - ERI_OLD))
	println("Finished THC routine, final rank is $curr_rank with cost $fin_cost")

	return ERI_OLD, ζ_OLD, UMATS_OLD
end

function THC_to_eri(ζ, Uvecs)
	R,N = size(Uvecs)
	eri_ret = zeros(N,N,N,N)

	@einsum eri_ret[i,j,k,l] = ζ[r1,r2] * Uvecs[r1,i] * Uvecs[r1,j] * Uvecs[r2,k] * Uvecs[r2,l]

	return eri_ret
end

function THC_normalizer(ζ, Uvecs)
	R,N = size(Uvecs)

	ζ_norm = copy(ζ)
	U_norm = copy(Uvecs)
	for r in 1:R
		norm_const = sum(abs2.(Uvecs[r,:]))
		Uvecs[r,:] /= sqrt(norm_const)
		ζ_norm[r,:] *= sqrt(norm_const)
	end

	return ζ_norm, U_norm
end

function THC_full(F :: F_OP)
	OB_ERI, TB_ERI = F_OP_to_eri(F)

	eri, ζ, Us = THC_search(F)
	ζ, Us = THC_normalizer(ζ, Us)

	println("Final THC cost is:")
	@show sum(abs2.(THC_to_eri(ζ, Us) - TB_ERI))

	ζ /= 2
	@show sum(abs2.(THC_to_eri(ζ, Us) - F.mbts[3]))

	num_ops = length(ζ)
	λ2 = [sum(abs.(ζ)), num_ops]

	return λ2
end

function DF_to_THC(F :: F_OP; debug=true)
	DF_FRAGS = DF_decomposition(F)
	M = length(DF_FRAGS)
	N = DF_FRAGS[1].N

	ζ = zeros(M*N, M*N)
	US = zeros(M*N, N)

	MN_idx = Dict{Tuple{Int64, Int64}, Int64}()
	idx = 0
	for α in 1:M
		for i in 1:N
			idx += 1
			get!(MN_idx, (α,i), idx)
		end
	end

	for α in 1:M
		for i in 1:N
			Uα = one_body_unitary(DF_FRAGS[α].U[1])
			αi_idx = MN_idx[(α,i)]
			US[αi_idx, :] = Uα[:,i]
			for j in 1:N
				αj_idx = MN_idx[(α,j)]
				ζ[αi_idx, αj_idx] = DF_FRAGS[α].C.λ[i] * DF_FRAGS[α].C.λ[j] * DF_FRAGS[α].coeff
			end
		end
	end

	if debug
		op_df = sum(to_OP.(DF_FRAGS))
		tbt_df = op_df.mbts[3]
		tbt_thc = THC_to_eri(ζ, US)

		@show sum(abs2.(tbt_df - tbt_thc))
	end

	λ1 = one_body_L1(H, count=true)
	ζ = sparse(ζ)
	num_ops = length(ζ.nzval)
	λ2 = [sum(abs.(ζ)), num_ops]

	@show λ1, λ2
	@show λ1 + λ2

	return λ1 + λ2
end

