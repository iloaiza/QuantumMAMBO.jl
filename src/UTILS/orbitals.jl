#orbital optimization routines

function orbital_l1_optimizer(F :: F_OP; verbose=false, ret_op = true, do_Givens=OO_GIVENS)
	#finds orbital rotation that minimizes Pauli l1-norm of fermionic operator
	u_num = real_orbital_rotation_num_params(F.N)
	x0 = zeros(u_num)

	if do_Givens
		U = givens_real_orbital_rotation(F.N, x0)
	else
		U = real_orbital_rotation(F.N, x0)
	end

	function cost(x)
		U.θs .= x
		return PAULI_L1(F_OP_rotation(U, F))
	end
	
	ini_cost = cost(x0)
	#u_params = 2π*rand(u_num) #randomizes initial angles, not necessary for convergence

	if verbose
		@show ini_cost
		@time sol = optimize(cost, x0, BFGS())
		@show sol.minimum
	else
		sol = optimize(cost, x0, BFGS())
	end

	U.θs .= sol.minimizer
	
	if ret_op
		return F_OP_rotation(U, F), U, sol.minimum, sol.minimizer
	else
		return U, sol.minimum, sol.minimizer
	end
end

#= Parallel routine, deprecated since convergence can be achieved in a single iteration
function parallel_orbital_l1_optimizer(F :: F_OP, reps = 10; verbose = false, do_Givens=OO_GIVENS)
	u_num = real_orbital_rotation_num_params(F.N)

	MINIMIZERS = SharedArray(zeros(u_num, reps))
	MINS = SharedArray(zeros(reps))

	@sync @distributed for i in 1:reps
		u, MINS[i] = orbital_l1_optimizer(F, verbose=false, ret_op=false)
		MINIMIZERS[:,i] = u.θs
	end

	ind_perm = sortperm(MINS)
	if verbose
		@show PAULI_L1(F)
		@show MINS[ind_perm[1]]
		@show MINS[ind_perm[end]]
	end

	x_min = MINIMIZERS[:,ind_perm[1]]
	if do_Givens
		U = givens_real_orbital_rotation(F.N, x_min)
	else
		U = real_orbital_rotation(F.N, x_min)
	end

	return F_OP_rotation(U, F), U, MINS[ind_perm[1]]
end
# =#
