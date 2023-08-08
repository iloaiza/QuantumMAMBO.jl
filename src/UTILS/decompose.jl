function CSA_greedy_decomposition(H :: F_OP, α_max; decomp_tol = ϵ, verbose=false, SAVELOAD=SAVING, SAVENAME=DATAFOLDER*"CSA.h5", kwargs...)
	F_rem = copy(H) #fermionic operator, tracks remainder after removing found greedy fragments
	F_rem.filled[1:2] .= false #only optimize 2-body tensor

	cartan_L = cartan_2b_num_params(H.N)
	unitary_L = real_orbital_rotation_num_params(H.N)

	tot_L = cartan_L + unitary_L
	if SAVELOAD
		X = zeros(tot_L, α_max)
	end

	Farr = F_FRAG[]
	α_curr = 0
	α_ini = 1

	if SAVELOAD
		fid = h5open(SAVENAME, "cw")
		if "CSA" in keys(fid)
			CSA_group = fid["CSA"]
			x = read(CSA_group, "x")
			x_len, α_curr = size(x)
			println("Found saved x under filename $SAVENAME for CSA decomposition, loaded $α_curr fragments...")
			if x_len != tot_L
				error("Trying to load from $SAVENAME, saved x has wrong dimensions for CSA parameters of H!")
			end
			α_ini = α_curr + 1
			for i in 1:α_curr
				frag = CSA_x_to_F_FRAG(x[:,i], H.N, H.spin_orb, cartan_L)
				push!(Farr, frag)
				F_rem = F_rem - to_OP(frag)
			end
			X[:,1:α_curr] = x
		else
			create_group(fid, "CSA")
			CSA_group = fid["CSA"]
		end
	end

	curr_cost = L2_partial_cost(F_rem)
	if verbose
		println("Initial L2 cost is $curr_cost")
	end

	while curr_cost > decomp_tol && α_curr < α_max
		α_curr += 1
		#x = [λ..., θ...] for cartan(λ) and U(θ)
		if verbose
			@time sol = CSA_greedy_step(F_rem; kwargs...)
			println("Current L2 cost after $α_curr fragments is $(sol.minimum)")
		else
			sol = CSA_greedy_step(F_rem; kwargs...)
		end
		frag = CSA_x_to_F_FRAG(sol.minimizer, H.N, H.spin_orb, cartan_L)
		push!(Farr, frag)
		F_rem = F_rem - to_OP(frag)
		curr_cost = sol.minimum
		if SAVELOAD
			X[:,α_curr] = sol.minimizer
			if haskey(CSA_group, "x")
				delete_object(CSA_group, "x") 
			end
			CSA_group["x"] = X[:, 1:α_curr]
		end
	end

	if verbose
		println("Finished CSA decomposition, total number of fragments is $α_curr, remainder L2-norm is $curr_cost")
	end

	if curr_cost > decomp_tol
		@warn "CSA decomposition did not converge, remaining L2-norm is $curr_cost"
	end

	if SAVELOAD
		close(fid)
	end

	return Farr
end

function CSA_greedy_step(F :: F_OP, do_svd = SVD_for_CSA, print = DECOMPOSITION_PRINT, do_grad=GRAD_for_CSA)
	cartan_L = cartan_2b_num_params(F.N)
	unitary_L = real_orbital_rotation_num_params(F.N)

	x0 = zeros(cartan_L + unitary_L)
	if do_svd
		frag = tbt_svd_1st(F.mbts[3]) #build initial guess from largest SVD fragment
		#frag = tbt_svd_avg(F.mbts[3]) #combine SVD fragments with largest unitary for initial guess
		x0[1:cartan_L] = frag.C.λ
		x0[cartan_L+1:end] = frag.U[1].θs
	else
		x0[cartan_L+1:end] = 2π*rand(unitary_L)
	end

	function cost(x)
		Fx = to_OP(CSA_x_to_F_FRAG(x, F.N, F.spin_orb, cartan_L))
		return L2_partial_cost(F, Fx)
	end
	
	function grad!(storage::Vector{Float64},x::Vector{Float64})
		Fx = to_OP(CSA_x_to_F_FRAG(x, F.N, F.spin_orb, cartan_L))
		diff=Fx.mbts[3]-F.mbts[3]
		storage.=gradient(size(diff,1),x,diff)
	end

	if print == false
		if do_grad
			return optimize(cost, grad!, x0, BFGS())
		else 
			return optimize(cost, x0,BFGS())
		end
	else
		if do_grad
			return optimize(cost, grad!, x0, BFGS(), Optim.Options(show_every=print, show_trace=true, extended_trace=true))
		else
			return optimize(cost, x0, BFGS(), Optim.Options(show_every=print, show_trace=true, extended_trace=true))
		end
	end
end

function CSA_SD_greedy_decomposition(H :: F_OP, α_max; decomp_tol = ϵ, verbose=false, SAVELOAD=SAVING, SAVENAME=DATAFOLDER*"CSA_SD.h5", kwargs...)
	#same as CSA decomposition, but includes optimization of one-body term
	F_rem = copy(H) #fermionic operator, tracks remainder after removing found greedy fragments
	F_rem.filled[1] = false #only optimize 1-body and 2-body tensors

	if H.spin_orb == true
		error("CSA_SD decomposition should be done with Hamiltonian represented in orbitals, not spin-orbitals!")
	end

	cartan_L = cartan_2b_num_params(H.N)
	unitary_L = real_orbital_rotation_num_params(H.N)

	tot_L = cartan_L + unitary_L + H.N
	if SAVELOAD
		X = zeros(tot_L, α_max)
	end

	Farr = F_FRAG[]
	α_curr = 0
	α_ini = 1

	if SAVELOAD
		fid = h5open(SAVENAME, "cw")
		if "CSA_SD" in keys(fid)
			CSA_SD_group = fid["CSA_SD"]
			x = read(CSA_SD_group, "x")
			x_len, α_curr = size(x)
			println("Found saved x under filename $SAVENAME for CSA_SD decomposition, loaded $α_curr fragments...")
			if x_len != tot_L
				error("Trying to load from $SAVENAME, saved x has wrong dimensions for CSA parameters of H!")
			end
			α_ini = α_curr + 1
			for i in 1:α_curr
				frag = CSA_SD_x_to_F_FRAG(x[:,i], H.N, H.spin_orb, cartan_L)
				push!(Farr, frag)
				F_rem = F_rem - to_OP(frag)
			end
			X[:,1:α_curr] = x
		else
			create_group(fid, "CSA_SD")
			CSA_SD_group = fid["CSA_SD"]
		end
	end

	curr_cost = L2_partial_cost(F_rem)
	if verbose
		println("Initial L2 cost is $curr_cost")
	end

	while curr_cost > decomp_tol && α_curr < α_max
		α_curr += 1
		#x = [λ..., θ...] for cartan(λ) and U(θ)
		if verbose
			@time sol = CSA_SD_greedy_step(F_rem; kwargs...)
			println("Current L2 cost after $α_curr fragments is $(sol.minimum)")
		else
			sol = CSA_SD_greedy_step(F_rem; kwargs...)
		end
		frag = CSA_SD_x_to_F_FRAG(sol.minimizer, H.N, H.spin_orb, cartan_L)
		push!(Farr, frag)
		F_rem = F_rem - to_OP(frag)
		curr_cost = sol.minimum
		if SAVELOAD
			X[:,α_curr] = sol.minimizer
			if haskey(CSA_SD_group, "x")
				delete_object(CSA_SD_group, "x") 
			end
			CSA_SD_group["x"] = X[:, 1:α_curr]
		end
	end

	if verbose
		println("Finished CSA_SD decomposition, total number of fragments is $α_curr, remainder L2-norm is $curr_cost")
		if curr_cost > decomp_tol
			@warn "CSA_SD decomposition did not converge, remaining L2-norm is $curr_cost"
		end
	end

	if SAVELOAD
		close(fid)
	end

	return Farr
end

function CSA_SD_greedy_step(F :: F_OP, do_svd = SVD_for_CSA_SD, print = DECOMPOSITION_PRINT, do_grad = GRAD_for_CSA_SD)
	cartan_L = cartan_2b_num_params(F.N)
	unitary_L = real_orbital_rotation_num_params(F.N)

	x0 = zeros(cartan_L + unitary_L + F.N)
	if do_svd
		frag = tbt_svd_1st(F.mbts[3]) #build initial guess from largest SVD fragment
		#frag = tbt_svd_avg(F.mbts[3]) #combine SVD fragments with largest unitary for initial guess
		x0[F.N+1:F.N+cartan_L] = frag.C.λ
		x0[F.N+cartan_L+1:end] = frag.U[1].θs
	else
		x0[F.N+cartan_L+1:end] = 2π*rand(unitary_L)
	end

	function cost(x)
		Fx = CSA_SD_x_to_F_FRAG(x, F.N, F.spin_orb, cartan_L)
		return L2_partial_cost(F, to_OP(Fx))
	end
	
	function SD_grad!(storage::Vector{Float64},x::Vector{Float64})
		Fx = to_OP(CSA_SD_x_to_F_FRAG(x, F.N, F.spin_orb, cartan_L))
		diff_ob=Fx.mbts[2]-F.mbts[2]
		diff_tb=Fx.mbts[3]-F.mbts[3]
		storage.=gradient_csa_sd(size(diff_ob,1),x,diff_ob,diff_tb)
	end

	if print == false
		if do_grad
			return optimize(cost, SD_grad!, x0, BFGS())
		else
			return optimize(cost, x0, BFGS())
		end
	else
		if do_grad
			return optimize(cost, SD_grad!, x0, BFGS(), Optim.Options(show_every=print, show_trace=true, extended_trace=true))
		else		
			return optimize(cost, x0, BFGS(), Optim.Options(show_every=print, show_trace=true, extended_trace=true))
		end
	end
end

function THC_fixed_decomposition(Ftarg :: F_OP, α, θ0 = 2π*rand(Ftarg.N-1, α), ζ0 = zeros(Int(α*(α+1)/2)))
	#do THC decomp, will find angles corresponding to α rotations
	if size(θ0)[2] < α #starting θ guess does not cover all
		diff = α - size(θ0)[2]
		θ0 = hcat(θ0, 2π*rand(Ftarg.N-1, diff))
	end

	x0 = cat(ζ0, reshape(θ0, :), dims=1)

	function cost(x)
		FRAGSx = THC_x_to_F_FRAGS(x, α, Ftarg.N)
		F = to_OP(FRAGSx[1])
		for i in 2:length(FRAGSx)
			F += to_OP(FRAGSx[i])
		end

		return L2_partial_cost(Ftarg, F)
	end

	return optimize(cost, x0, BFGS())
end

function THC_iterative_decomposition(H :: F_OP, α_max, decomp_tol = ϵ)
	F = copy(H)
	F.filled[1:2] .= false

	println("Initial L2 cost is $(L2_partial_cost(F))")
	@time sol = THC_fixed_decomposition(F, 1)

	i = 2
	curr_cost = sol.minimum
	println("L2 cost with THC dimension 1 is $curr_cost")

	while i <= α_max && curr_cost > decomp_tol
		old_ζL = Int(i*(i-1)/2)
		ζ0 = zeros(Int(i*(i+1)/2))
		ζ0[1:Int(i*(i-1)/2)] = sol.minimizer[1:old_ζL]
		θ0 = reshape(sol.minimizer[old_ζL+1:end],H.N-1,i-1)
		@time sol = THC_fixed_decomposition(F, i, θ0, ζ0)
		i += 1
		curr_cost = sol.minimum

		println("L2 cost with THC dimension $(i-1) is $curr_cost")
	end

	return THC_x_to_F_FRAGS(sol.minimizer, i-1, H.N)
end


function DF_decomposition(H :: F_OP; tol=SVD_tol, tiny=SVD_tiny, verbose=false, debug=false, do_Givens=DF_GIVENS)
	#do double-factorization
	#do_Givens will try to transform each orbital rotation into Givens rotations, false returns one-body rotation matrices directly
	if verbose
		println("Starting Double-Factorization routine")
	end
	if H.spin_orb
		@warn "Doing Double-Factorization for spin-orb=true, be wary of results..."
	end
	
	n = H.N
	N = n^2


	tbt_full = reshape(H.mbts[3], (N,N))
	tbt_res = Symmetric(tbt_full)
	if sum(abs.(tbt_full - tbt_res)) > tiny
		println("Non-symmetric two-body tensor as input for DF routine, calculations might have errors...")
		tbt_res = tbt_full
	end

	if verbose
		println("Diagonalizing two-body tensor")
		@time Λ,U = eigen(tbt_res)
	else
		Λ,U = eigen(tbt_res)
	end
	## U*Diagonal(Λ)*U' == tbt_res
	ind=sortperm(abs.(Λ))[end:-1:1]
    Λ = Λ[ind]
    U=U[:,ind]
	
	num_ops = N
    for i in 1:N
    	if abs(Λ[i]) < tol
    		if verbose
    			println("Truncating DF for SVD coefficients with magnitude smaller or equal to $(abs(Λ[i])), using $(i-1) fragments")
    		end
    		num_ops = i-1
    		break
    	end
	end
	Λ = Λ[1:num_ops]
	U = U[:,1:num_ops]

	FRAGS = [initialize_FRAG(n, DF()) for i in 1:num_ops]
	TBT = zeros(n,n,n,n)

	for i in 1:num_ops
        full_l = reshape(U[:, i], (n,n))
        cur_l = Symmetric(full_l)
        sym_dif = sum(abs2.(cur_l - full_l))
        if sym_dif > tiny
        	if sum(abs.(full_l + full_l')) > tiny
				error("DF fragment $i is neither Hermitian or anti-Hermitian!")
			end
        	cur_l = Hermitian(1im * full_l)
        	Λ[i] *= -1
        end
    
		ωl, Ul = eigen(cur_l)
		if do_Givens
			if sum(abs.(imag.(log(Ul)))) > 1e-8
				Rl = f_matrix_rotation(n, Ul)
				C = cartan_1b(H.spin_orb, ωl, n)
				FRAGS[i] = F_FRAG(1, tuple(Rl), DF(), C, n, H.spin_orb, Λ[i], true)
			else
				Rl = SOn_to_MAMBO_full(Ul, verbose = false)
				if sum(abs2.(Ul - one_body_unitary(Rl))) > ϵ_Givens
					Rl = f_matrix_rotation(n, Ul)
				end
				C = cartan_1b(H.spin_orb, ωl, n)
				FRAGS[i] = F_FRAG(1, tuple(Rl), DF(), C, n, H.spin_orb, Λ[i], true)
			end
		else
			Rl = f_matrix_rotation(n, Ul)
			C = cartan_1b(H.spin_orb, ωl, n)
			FRAGS[i] = F_FRAG(1, tuple(Rl), DF(), C, n, H.spin_orb, Λ[i], true)
		end
	end

	if debug
		tbt_tot = zeros(n,n,n,n)
		for i in 1:num_ops
			tbt_tot += to_OP(FRAGS[i]).mbts[3]
		end

		L2_rem = sum(abs2.(tbt_tot - H.mbts[3]))
		if verbose
			@show L2_rem
		end
		if L2_rem > ϵ
			@warn "Remainder of DF decomposition is larger than tolerance $ϵ..."
		end
	end
    
    return FRAGS
end

function DF_based_greedy(F :: F_OP)
	#obtains greedy CSA fragment by using orbital frame from largest DF fragment and collecting Cartan coefficients from operator in this frame
	DF_FRAGS = DF_decomposition(F, do_Givens=false)
	F1 = DF_FRAGS[1]
	Lmat = zeros(F.N, F.N)

	for i in 1:F.N
		Lmat[i,i] = F1.C.λ[i]^2
		for j in i+1:F.N
			Lmat[i,j] = Lmat[j,i] = F1.C.λ[i] * F1.C.λ[j]
		end
	end
	Lmat *= F1.coeff

	U1 = one_body_unitary(F1.U[1])
	U1dag = U1'

	for m in 2:length(DF_FRAGS)
		Fm = DF_FRAGS[m]
		Um = one_body_unitary(Fm.U[1])
		Vm = Um * U1dag
		for i in 1:F.N
			for j in 1:F.N
				for p in 1:F.N
					for q in 1:F.N
						Lmat[i,j] += Fm.C.λ[p] * Fm.C.λ[q] * abs2(Vm[i,p]) * abs2(Vm[j,q]) * Fm.coeff
					end
				end
			end
		end
	end

	C = cartan_mat_to_2b(Lmat, F.spin_orb)
	F = F_FRAG(1, F1.U, CSA(), C, F.N, F.spin_orb)

	return F
end


function MTD_CP4_greedy_step(F :: F_OP; x0 = false, print = DECOMPOSITION_PRINT)

	function cost(x)
		Fx = MTD_CP4_x_to_F_FRAG(x, F.N, F.spin_orb)

		return L2_partial_cost(F, to_OP(Fx))
	end

	if x0 == false
		x0 = zeros(4*F.N - 3)
		x0[1:4*F.N - 4] .= 2π*rand(4*F.N - 4)
	end

	if print == false
		return optimize(cost, x0, BFGS())
	else
		return optimize(cost, x0, BFGS(), Optim.Options(show_every=print, show_trace=true, extended_trace=false))
	end
end

function MTD_CP4_greedy_decomposition(H :: F_OP, α_max; decomp_tol = ϵ, verbose=true, SAVELOAD=SAVING, SAVENAME=DATAFOLDER*"MTD_CP4.h5")
	F_rem = copy(H) #fermionic operator, tracks remainder after removing found greedy fragments
	F_rem.filled[1:2] .= false #only optimize 2-body tensor

	tot_L = 4*H.N - 3
	if SAVELOAD
		X = zeros(tot_L, α_max)
	end

	Farr = F_FRAG[]
	α_curr = 0
	α_ini = 1

	if SAVELOAD
		fid = h5open(SAVENAME, "cw")
		if "MTD_CP4" in keys(fid)
			MTD_CP4_group = fid["MTD_CP4"]
			x = read(MTD_CP4_group, "x")
			x_len, α_curr = size(x)
			println("Found saved x under filename $SAVENAME for MTD_CP4 decomposition, loaded $α_curr fragments...")
			if x_len != tot_L
				error("Trying to load from $SAVENAME, saved x has wrong dimensions for MTD_CP4 parameters of H!")
			end
			α_ini = α_curr + 1
			for i in 1:α_curr
				frag = MTD_CP4_x_to_F_FRAG(x[:,i], H.N, H.spin_orb)
				push!(Farr, frag)
				F_rem = F_rem - to_OP(frag)
			end
			X[:,1:α_curr] = x
		else
			create_group(fid, "MTD_CP4")
			MTD_CP4_group = fid["MTD_CP4"]
		end
	end

	curr_cost = L2_partial_cost(F_rem)
	if verbose
		println("Initial L2 cost is $curr_cost")
	end

	while curr_cost > decomp_tol && α_curr < α_max
		α_curr += 1
		#x = [λ..., θ...] for cartan(λ) and U(θ)
		if verbose
			@time sol = MTD_CP4_greedy_step(F_rem)
			println("Current L2 cost after $α_curr fragments is $(sol.minimum)")
		else
			sol = MTD_CP4_greedy_step(F_rem)
		end
		frag = MTD_CP4_x_to_F_FRAG(sol.minimizer, H.N, H.spin_orb)
		push!(Farr, frag)
		F_rem = F_rem - to_OP(frag)
		curr_cost = sol.minimum
		if SAVELOAD
			X[:,α_curr] = sol.minimizer
			if haskey(MTD_CP4_group, "x")
				delete_object(MTD_CP4_group, "x") 
			end
			MTD_CP4_group["x"] = X[:, 1:α_curr]
		end
	end

	if verbose
		println("Finished MTD_CP4 decomposition, total number of fragments is $α_curr, remainder L2-norm is $curr_cost")
	end

	if curr_cost > decomp_tol
		@warn "MTD_CP4 decomposition did not converge, remaining L2-norm is $curr_cost"
	end

	if SAVELOAD
		close(fid)
	end

	return Farr
end
