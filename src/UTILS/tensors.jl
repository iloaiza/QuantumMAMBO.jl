tfox = pyimport("TensorFox")

# = TensorLy implementation, TensorFox is more efficient
tly = pyimport("tensorly")
function TLY_decomp(tsr, rank)
	factors = tly.decomposition.parafac(tsr, rank=rank)

	fact_tsr = tly.cp_to_tensor(factors)

	@show typeof(fact_tsr)
	@show sum(abs2.(tsr - fact_tsr))

	return factors.factors
end
# =#

function CP_decomp(tsr, rank_max; tol=ϵ, verbose=false, ini_rank = 2, rank_hop = 1)
	curr_cost = sum(abs2.(tsr))
	curr_rank = ini_rank

	factors = 0
	output = 0
	while curr_cost > tol && curr_rank < rank_max
		curr_rank += rank_hop
		if verbose
			println("Starting rank $curr_rank")
			#@time factors, output = tfox.cpd(tsr, curr_rank)
			@time factors = tly.decomposition.parafac(tsr, rank=curr_rank)
		else
			factors, output = tfox.cpd(tsr, curr_rank)
		end
		fact_tsr = tly.cp_to_tensor(factors)#tfox.cpd2tens(factors)
		curr_cost = sum(abs2.(tsr - fact_tsr))
		if verbose
			println("Remainder cost for rank $curr_rank is $curr_cost")
		end
	end

	println("CP4 decomposition converged to cost of $curr_cost using rank=$curr_rank")

	return factors.factors, curr_rank
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

function partial_CP4()
end

