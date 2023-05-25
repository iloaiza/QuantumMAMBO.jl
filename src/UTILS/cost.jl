function L2_TB_cost(F1 :: F_OP, F2 :: F_OP)
	#return L2-norm of two-body tensor difference
	if F1.Nbods < 2 || F2.Nbods < 2
		error("Trying to calculate two-body difference for operators that don't have two-body component!")
	end

	if F1.filled[3] && F2.filled[3]
		return sum(abs2.(F1.mbts[3] - F2.mbts[3]))
	elseif F1.filled[3]
		return sum(abs2.(F1.mbts[3]))
	else
		return sum(abs2.(F2.mbts[3]))
	end
end

function L2_partial_cost(F1 :: F_OP, F2 :: F_OP)
	#only calculates cost with respect to tensors that are filled in both, neglects identity term
	tot_cost = 0.0

	if F1.Nbods == F2.Nbods
		for i in 2:F1.Nbods+1
			if F1.filled[i] && F2.filled[i]
				tot_cost += sum(abs2.(F1.mbts[i] - F2.mbts[i]))
			end
		end
	else
		Nmax = maximum([F1.Nbods, F2.Nbods])

		filled_1 = zeros(Bool,Nmax+1)
		filled_2 = zeros(Bool,Nmax+1)
		filled_1[1:F1.Nbods+1] = F1.filled
		filled_2[1:F2.Nbods+1] = F2.filled

		for i in 2:Nmax+1
			if filled_1[i] && filled_2[2]
				tot_cost += sum(abs2.(F1.mbts[i] - F2.mbts[i]))
			end
		end
	end

	return tot_cost
end

function L2_partial_cost(F :: F_OP)
	#return L1 cost of F summed over all filled tensors except identity term
	tot_cost = 0.0

	for i in 2:F.Nbods+1
		if F.filled[i]
			tot_cost += sum(abs2.(F.mbts[i]))
		end
	end

	return tot_cost
end

function L2_total_cost(F1 :: F_OP, F2 :: F_OP)
	#calculates cost with respect to all tensors
	tot_cost = 0.0

	if F1.Nbods == F2.Nbods
		for i in 1:F1.Nbods+1
			if F1.filled[i]
				if F2.filled[i]
					tot_cost += sum(abs2.(F1.mbts[i] - F2.mbts[i]))
				else
					tot_cost += sum(abs2.(F1.mbts[i]))
				end
			elseif F2.filled[i]
				tot_cost += sum(abs2.(F2.mbts[i]))
			end
		end
	else
		Nmax = maximum([F1.Nbods, F2.Nbods])

		filled_1 = zeros(Bool,Nmax+1)
		filled_2 = zeros(Bool,Nmax+1)
		filled_1[1:F1.Nbods+1] = F1.filled
		filled_2[1:F2.Nbods+1] = F2.filled

		for i in 1:Nmax+1
			if filled_1[i] && filled_2[2]
				tot_cost += sum(abs2.(F1.mbts[i] - F2.mbts[i]))
			end
		end
	end

	return tot_cost
end

function L2_total_cost(F :: F_OP)
	tot_cost = 0.0

	for i in 1:F.Nbods+1
		tot_cost += sum(abs.(F.mbts[i]))
	end

	return tot_cost
end