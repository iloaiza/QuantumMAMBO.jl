function PB0_calculator(H :: Q_OP; verbose=true)
	α = 0.0

	for i in 1:H.n_paulis
		for j in i+1:H.n_paulis
			if pws_is_anticommuting(H.paulis[i], H.paulis[j])
				α += abs(H.paulis[i].coeff * H.paulis[j].coeff)
			end
		end
	end

	α *= 2
	
	if verbose
		@show α
	end	
	
	return α
end

function PB0_orbital_optimizer(H :: F_OP; verbose=true)
	u_num = real_orbital_rotation_num_params(H.N)
	x0 = zeros(u_num)

	U = givens_real_orbital_rotation(H.N, x0)
	
	function cost(x)
		U.θs .= x
		return PB0_calculator(Q_OP(F_OP_rotation(U, H)), verbose=verbose)
	end
	
	ini_cost = cost(x0)

	if verbose
		@show ini_cost
		@time sol = optimize(cost, x0, BFGS())
		@show sol.minimum
	else
		sol = optimize(cost, x0, BFGS())
	end

	U.θs .= sol.minimizer
	
	return sol.minimum
end