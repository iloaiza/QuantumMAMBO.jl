#majorana utils
function M_OP(F :: F_OP)
	#transform Fermionic operator into Majorana operator
	Nmajs = 2*F.Nbods
	if Nmajs > 4
		error("Trying to build Majorana operator for more than 2-body fermionic operator, not implemented!")
	end
	
	spin_orb = F.spin_orb
	filled = zeros(Bool, Nmajs+1)
	MBTS = Array{Float64}[]
	body_sym = true
	N = F.N
	t_coeffs = ones(Complex, Nmajs+1)
	
	#identity term
	filled[1] = true
	id_const = F.mbts[1][1]
	if F.Nbods == 0
		return M_OP(Nmajs, tuple([id_const]), [1], [1], body_sym, spin_orb, N)
	end
	if spin_orb
		ID_FACT = 0.5
	else
		ID_FACT = 1
	end
	if F.filled[2]
		for i in 1:F.N
			id_const += ID_FACT * F.mbts[2][i,i]
		end
	end

	if F.Nbods == 1
		push!(MBTS, [id_const])
		if F.filled[2]
			obt = copy(F.mbts[2])/2
		else
			obt = [0]
		end
		filled[3] = true
		t_coeffs[3] = 1im
		push!(MBTS, [0])
		push!(MBTS, obt)

		return M_OP(Nmajs, tuple(MBTS...), t_coeffs, filled, body_sym, spin_orb, N)
	end

	if spin_orb
		if F.filled[3]
			for i in 1:F.N
				for j in 1:F.N
					id_const += 0.25 * F.mbts[3][i,i,j,j]
				end
			end
		end
	else
		if F.filled[3]
			for i in 1:F.N
				for j in 1:F.N
					id_const += F.mbts[3][i,i,j,j] + 0.5 * F.mbts[3][i,j,j,i]
				end
			end
		end
	end
	push!(MBTS, [id_const])
	
	#one-body (i.e. 2-Majorana term)
	if F.Nbods > 0
		filled[3] = true
		t_coeffs[3] = 1im
		if F.filled[2]
			obt = copy(F.mbts[2])/2
		elseif F.filled[3]
			obt = zeros(N, N)
		else
			obt = [0]
		end

		if F.Nbods > 1 && F.filled[3]
			for i in 1:N
				for j in 1:N
					obt[i,j] += ID_FACT * sum([F.mbts[3][i,j,k,k] for k in 1:N])
				end
			end
		end

		push!(MBTS, [0])
		push!(MBTS, obt)
	end

	#two-body
	if F.Nbods > 1
		push!(MBTS, [0])
	
		if F.filled[3] == false
			tbt = [0]
		else
			filled[5] = true
			tbt = -0.25*F.mbts[3]
		end
		
		push!(MBTS, tbt)
	end

	return M_OP(Nmajs, tuple(MBTS...), t_coeffs, filled, body_sym, spin_orb, N)
end