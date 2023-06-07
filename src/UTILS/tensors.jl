function TLY_decomp(tsr, rank)
	factors = tly.decomposition.parafac(tsr, rank=rank)

	fact_tsr = tly.cp_to_tensor(factors)

	@show typeof(fact_tsr)
	@show sum(abs2.(tsr - fact_tsr))

	return factors.factors
end
# =#

function CP_decomp(tsr, rank_max; tol=ϵ, verbose=true, ini_rank = 1, rank_hop = 1)
	curr_cost = sum(abs2.(tsr))
	curr_rank = ini_rank

	factors = 0
	output = 0
	while curr_cost > tol && curr_rank < rank_max
		curr_rank += rank_hop
		if verbose
			println("Starting rank $curr_rank")
			@time factors, output = tfox.cpd(tsr, curr_rank)
			#@time factors = tly.decomposition.parafac(tsr, rank=curr_rank)
		else
			factors, output = tfox.cpd(tsr, curr_rank)
			#@time factors = tly.decomposition.parafac(tsr, rank=curr_rank)
		end
		#fact_tsr = pyconvert(Array{Float64}, tly.cp_to_tensor(factors))
		fact_tsr = pyconvert(Array{Float64}, tfox.cpd2tens(factors))
		curr_cost = sum(abs2.(tsr - fact_tsr))
		if verbose
			println("Remainder cost for rank $curr_rank is $curr_cost")
		end
	end

	println("CP4 decomposition converged to cost of $curr_cost using rank=$curr_rank")

	return pyconvert.(Array{Float64}, factors), curr_rank
end

function CP_factors_to_tsr(FACTORS)
	facts_dims = size.(FACTORS)

	rank = facts_dims[1][2]
	
	tsr_rank = length(facts_dims)
	tsr_dims = [facts_dims[i][1] for i in 1:tsr_rank]

	tsr = zeros(tsr_dims...)

	if tsr_rank == 4
		for i in 1:tsr_dims[1]
			for j in 1:tsr_dims[2]
				for k in 1:tsr_dims[3]
					for l in 1:tsr_dims[4]
						for r in 1:rank
							tsr[i,j,k,l] += FACTORS[1][i,r] * FACTORS[2][j,r] * FACTORS[3][k,r] * FACTORS[4][l,r]
						end
					end
				end
			end
		end
	else
		error("Not implemented for rank ≠ 4!")
		@warn "Non-efficient implementation of CP factors to tensor for tensor rank ≠ 4, continue at your own risk!"
		for_string = "for r in 1:rank; "
		end_string = "end; "
		i_string = ""
		FACTORS_string = ""
		for i in 1:tsr_rank
			for_string = for_string * "for i$i in 1:tsr_dims[$i]; "
			end_string = end_string * "end; "
			if i < tsr_rank
				i_string = i_string * "i$i,"
				FACTORS_string = FACTORS_string * "FACTORS[$i][i$i,r] * "
			else
				i_string = i_string * "i$i"
				FACTORS_string = FACTORS_string * "FACTORS[$i][i$i,r]; "
			end
		end
		fin_string = for_string * "tsr[" * i_string * "] += " * FACTORS_string * end_string

		@show fin_string

		eval(Meta.parse(fin_string))
	end

	return tsr
end

function CP4_factors_normalizer(FACTORS)
	@show size(FACTORS[1])
	rank = size(FACTORS[1])[2]

	F1 = copy(FACTORS[1])
	F2 = copy(FACTORS[2])
	F3 = copy(FACTORS[3])
	F4 = copy(FACTORS[4])
	λr = zeros(rank)
	for r in 1:rank
		λ1 = sum(abs2.(F1[:,r]))
		F1[:,r] /= sqrt(λ1)
		λ2 = sum(abs2.(F2[:,r]))
		F2[:,r] /= sqrt(λ2)
		λ3 = sum(abs2.(F3[:,r]))
		F3[:,r] /= sqrt(λ3)
		λ4 = sum(abs2.(F4[:,r]))
		F4[:,r] /= sqrt(λ4)
		λr[r] = sqrt(λ1*λ2*λ3*λ4)
	end

	return [F1, F2, F3, F4], λr
end

function CP4_weighted_factors_to_tsr(FACTORS, λr)
	facts_dims = size.(FACTORS)

	@show facts_dims

	rank = facts_dims[1][2]
	
	tsr_rank = length(facts_dims)
	tsr_dims = [facts_dims[i][1] for i in 1:tsr_rank]

	tsr = zeros(tsr_dims...)

	if tsr_rank == 4
		for i in 1:tsr_dims[1]
			for j in 1:tsr_dims[2]
				for k in 1:tsr_dims[3]
					for l in 1:tsr_dims[4]
						for r in 1:rank
							tsr[i,j,k,l] += FACTORS[1][i,r] * FACTORS[2][j,r] * FACTORS[3][k,r] * FACTORS[4][l,r] * λr[r]
						end
					end
				end
			end
		end
	else
		error("Not implemented for rank ≠ 4!")
	end

	return tsr
end

function CP4_decomposition(tsr, rank_max; tol=ϵ, verbose=false, ini_rank = 1, rank_hop = 1, SAVELOAD=false, SAVENAME=DATAFOLDER*"CP4.h5")
	if verbose
		println("Starting PARAFAC routine")
	end

	N = size(tsr)[1]
	FRAGS = F_FRAG[]

	if SAVELOAD
		fid = h5open(SAVENAME, "cw")
		if "PARAFAC" in keys(fid)
			PARAFAC_group = fid["PARAFAC"]
			x = read(PARAFAC_group, "x")
			x_len, α_curr = size(x)
			println("Found saved x under filename $SAVENAME for PARAFAC decomposition, loaded $α_curr fragments...")
			if x_len != 4N+1
				error("Trying to load from $SAVENAME, saved x has wrong dimensions for PARAFAC parameters of H!")
			end
			α_ini = α_curr + 1
			for i in 1:α_curr
				frag = MTD_PARAFAC_x_to_F_FRAG(x[:,i], N, false)
				push!(FRAGS, frag)
			end
		end
	end

	if length(FRAGS) == 0
		factors, rank = CP_decomp(tsr, rank_max, tol=tol, verbose=verbose, ini_rank = ini_rank, rank_hop = rank_hop)
		factors, Ω = CP4_factors_normalizer(factors)
		
		if SAVELOAD
			x_save = zeros(4N+1, rank)
		end
		for i in 1:rank
			U_ARR = zeros(N, 4)
			for α in 1:4
				for j in 1:N
					U_ARR[j,α] = factors[α][j,i]
				end
			end

			if SAVELOAD
				x_save[1:end-1,i] = vcat(U_ARR...)
				x_save[end,i] = Ω[i]
			end

			Us = tuple([single_orbital_rotation(N, U_ARR[:,α]) for α in 1:4]...)
			frag = F_FRAG(4, Us, MTD_PARAFAC(), cartan_m1(), N, false, Ω[i], true)

			push!(FRAGS, frag)
		end

		if SAVELOAD
			create_group(fid, "PARAFAC")
			PARAFAC_group = fid["PARAFAC"]
			PARAFAC_group["x"] = x_save
			close(fid)
		end
	end

	if verbose
		OP = sum(to_OP.(FRAGS))
		tbt = OP.mbts[3]

		@show sum(abs2.(tbt - tsr))
	end

	return FRAGS
end

function CP4_decomposition(F :: F_OP, rank_max; tol=ϵ, verbose=false, rank_hop = 1, SAVELOAD=false, SAVENAME=DATAFOLDER*"CP4.h5")
	return CP4_decomposition(F.mbts[3], rank_max, tol=tol, verbose=verbose, rank_hop = rank_hop, SAVELOAD=SAVELOAD, SAVENAME=SAVENAME)
end

	