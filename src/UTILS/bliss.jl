#routines for finding block-invariant symmetry shifts (BLISS), see Ref.[2] in README for more information
function bliss_sym_params_to_F_OP(ovec, t1, t2, η, N = Int((sqrt(8*length(ovec)+1) - 1)/2), spin_orb=false)
	#builds S symmetry shift corresponding to S = s0+s1+s2
	obt = zeros(N, N)
	idx = 0
	for i in 1:N
		for j in 1:i
			idx += 1
			obt[i,j] += ovec[idx]
			obt[j,i] += ovec[idx]
		end
	end

	s0 = [-t1*(η^2) - t2*η]
	s1 = t2*collect(Diagonal(ones(N))) - η*obt

	s2 = zeros(N,N,N,N)
	for i in 1:N
		for j in 1:N
			s2[i,i,j,j] += t1
			for k in 1:N
				s2[i,j,k,k] += 0.5*obt[i,j]
				s2[k,k,i,j] += 0.5*obt[i,j]
			end
		end
	end

	return F_OP((s0, s1, s2), spin_orb)
end


function quadratic_bliss_params_to_F_OP(u1, u2, ovec, η, N)
	
	omat = zeros(N,N)
	idx=1
	for i=1:N
		for j=i:N
			omat[i,j]=ovec[idx]
			omat[j,i]=ovec[idx]
			idx+=1
		end
	end
	
	
	
	Sconst = [-η - η^2]

	Sobt = zeros(N, N)
	Tobt = zeros(N, N)
	for i in 1:N
		Sobt[i,i] = u1
		for j in 1:N
			Tobt[i,j] = -2*η*omat[i,j]
		end
	end

	Stbt = zeros(N, N, N, N)
	Ttbt = zeros(N, N, N, N)

	for i in 1:N
		for j in 1:N
			Stbt[i,i,j,j] = u2
			for k in 1:N
				
				Ttbt[i,j,k,k] += omat[i,j]
				Ttbt[k,k,i,j] += omat[i,j]
			end
		end
	end

	S = F_OP((Sconst, Sobt, Stbt))
	T = F_OP(([0], Tobt, Ttbt))
	ST = S+T
	return ST
end
	

function quadratic_ss(F :: F_OP, η)
	hij = F.mbts[2]
	gijkl = F.mbts[3]

	H = sum([hij[i,i] for i in 1:F.N])
	G = sum([gijkl[i,i,j,j] for i in 1:F.N, j in 1:F.N])

	G1 = 0.0
	for i in 1:F.N
		for j in 1:i-1
			G1 += gijkl[i,j,j,i] - gijkl[i,i,j,j]
		end
	end

	u2 = 4*(G + G1/2)/(5*(F.N^2) - F.N)
	u1 = (H + 2G - 2*(F.N^2)*u2)/F.N

	Sconst = [-η - η^2]

	Sobt = zeros(F.N, F.N)
	for i in 1:F.N
		Sobt[i,i] = u1
	end

	Stbt = zeros(F.N, F.N, F.N, F.N)
	
	for i in 1:F.N
		for j in 1:F.N
			Stbt[i,i,j,j] = u2
		end
	end

	S = F_OP((Sconst, Sobt, Stbt))

	return F - S
end


function quadratic_bliss(F, η; verbose=false)
	hij = F.mbts[2]
	gijkl = F.mbts[3]

	H = sum([hij[i,i] for i in 1:F.N])
	G = sum([gijkl[i,i,j,j] for i in 1:F.N, j in 1:F.N])

	G1 = 0.0
	

	for i in 1:F.N
		for j in 1:i-1
			G1 += gijkl[i,j,j,i] - gijkl[i,i,j,j]
		end
	end

	Gnm = zeros(F.N, F.N)
	
	
	Gtilde_nm = zeros(F.N, F.N)
	Htilde_nm = zeros(F.N, F.N)

	for n in 1:F.N
		for m in 1:F.N
			Gnm[n,m] = sum([gijkl[n,m,k,k] for k in 1:F.N])
			
			for k in 1:F.N
				if k != n && k != m
					Gtilde_nm[n,m] += gijkl[n,k,k,m] - gijkl[n,m,k,k]
				end
			end
			Htilde_nm[n,m] = 4*(η-F.N)*(hij[n,m] + 2*Gnm[n,m]) - 2*Gnm[n,m] + 2*Gtilde_nm[n,m]
		end
	end

	omat = zeros(F.N, F.N)
	for i in 1:F.N
		for j in i+1:F.N
			omat[i,j] = omat[j,i] = -Htilde_nm[i,j]/(8(η-F.N)^2+2F.N+2(F.N-2))
		end
	end

	α = 8*(η-F.N)^2+4(F.N-1)
	β = 24F.N-16η+4
	γ = 8F.N-4η
	δ = 16F.N^2-8η*F.N+4F.N-2

	Htilde_n = zeros(F.N)
	for n in 1:F.N
		Htilde_n[n] = 4*(η-F.N)*(hij[n,n] + 2*Gnm[n,n]) + 2*Gtilde_nm[n,n] - 2*Gnm[n,n]
	end

	μ1 = -2/F.N
	μ2 = (2G1-G)/(F.N*(1-2F.N))
	λ1 = (H+2*G)/F.N
	λ2 = -2*F.N
	λ3 = 2*(η-2*F.N)/F.N

	ee = β+γ*λ1*μ1+γ*λ3+δ*μ1
	Htt_n = zeros(F.N)
	for n in 1:F.N
		Htt_n[n] = Htilde_n[n]-4H-8G + γ*(λ1+λ2*μ2) + δ*μ2
	end

	Htt = sum(Htt_n)
	Osum = -Htt/(α+ee*F.N)

	for n in 1:F.N
		omat[n,n] = -(Htt_n[n] + Osum*ee)/α
	end

	u2 = μ1*Osum + μ2
	u1 = λ1 + λ2*u2 + λ3*Osum
	

	Sconst = [-η - η^2]

	Sobt = zeros(F.N, F.N)
	Tobt = zeros(F.N, F.N)
	for i in 1:F.N
		Sobt[i,i] = u1
		for j in 1:F.N
			Tobt[i,j] = -2*η*omat[i,j]
		end
	end

	Stbt = zeros(F.N, F.N, F.N, F.N)
	Ttbt = zeros(F.N, F.N, F.N, F.N)

	for i in 1:F.N
		for j in 1:F.N
			Stbt[i,i,j,j] = u2
			for k in 1:F.N
				Ttbt[i,j,k,k] += omat[i,j]
				Ttbt[k,k,i,j] += omat[i,j]
			end
		end
	end
	
	S = F_OP((Sconst, Sobt, Stbt))
	T = F_OP(([0], Tobt, Ttbt))

	FT = F - S - T
	if verbose
		println("Finished quadratic BLISS routine, showing initial and final 2-norms...")
		@show PAULI_L2(F)
		@show PAULI_L2(FT)
	end

	return FT
end


function bliss_optimizer(F :: F_OP, η; verbose=true, SAVELOAD = SAVING, SAVENAME=DATAFOLDER*"BLISS.h5")
	if SAVELOAD
		fid = h5open(SAVENAME, "cw")
		if haskey(fid, "BLISS")
			BLISS_group = fid["BLISS"]
			if haskey(BLISS_group, "ovec")
				println("Loading results for BLISS optimization from $SAVENAME")
				ovec = read(BLISS_group,"ovec")
				t1 = read(BLISS_group,"t1")
				t2 = read(BLISS_group,"t2")
				close(fid)
				S = bliss_sym_params_to_F_OP(ovec, t1, t2, η, F.N, F.spin_orb)
				close(fid)
				return F-S
			end
		end
		close(fid)
	end

	L = Int(F.N*(F.N+1)/2)
	x0 = zeros(L + 2)

	function cost(x)
		return PAULI_L1(F - bliss_sym_params_to_F_OP(x[3:end], x[1], x[2], η, F.N, F.spin_orb))
	end

	if verbose
		println("Starting 1-norm cost:")
		@show cost(x0)
		@time sol = optimize(cost, x0, BFGS(), Optim.Options(f_tol = 1e-3))
		println("Final 1-norm cost:")
		@show sol.minimum
	else
		sol = optimize(cost, x0, BFGS())
	end

	xsol = sol.minimizer
	if SAVELOAD
		fid = h5open(SAVENAME, "cw")
		create_group(fid, "BLISS")
		BLISS_group = fid["BLISS"]
		println("Saving results of BLISS optimization to $SAVENAME")
		BLISS_group["ovec"] = xsol[3:end]
		BLISS_group["t1"] = xsol[1]
		BLISS_group["t2"] = xsol[2]
		close(fid)
	end
	S = bliss_sym_params_to_F_OP(xsol[3:end], xsol[1], xsol[2], η, F.N, F.spin_orb)

	return F - S
end


function Sz_builder(n_qubit)
	Sz = zeros(n_qubit, n_qubit)
	for i in 1:n_qubit
		if i%2 == 1
			Sz[i,i] = 1
		else
			Sz[i,i] = -1
		end
	end

	return Sz
end

function S2_builder(n_qubit)
	S2_tbt = zeros(n_qubit,n_qubit,n_qubit,n_qubit)
	n_orbs = Int(n_qubit/2)
	for i in 1:n_orbs
		ka = 2i-1
		kb = 2i
		S2_tbt[ka,ka,ka,ka] += 1
		S2_tbt[kb,kb,kb,kb] += 1
		S2_tbt[ka,ka,kb,kb] += -1
		S2_tbt[kb,kb,ka,ka] += -1
		S2_tbt[ka,kb,kb,ka] += 2
		S2_tbt[kb,ka,ka,kb] += 2
	end

	for i in 1:n_orbs
		for j in 1:n_orbs
			if i != j
				ka = 2i-1
				kb = 2i
				la = 2j-1
				lb = 2j
				S2_tbt[ka,kb,lb,la] += 1
				S2_tbt[lb,la,ka,kb] += 1
				S2_tbt[kb,ka,la,lb] += 1
				S2_tbt[la,lb,kb,ka] += 1
				S2_tbt[ka,ka,la,la] += 0.5
				S2_tbt[la,la,ka,ka] += 0.5
				S2_tbt[kb,kb,lb,lb] += 0.5
				S2_tbt[lb,lb,kb,kb] += 0.5
				S2_tbt[ka,ka,lb,lb] += -0.5
				S2_tbt[lb,lb,ka,ka] += -0.5
				S2_tbt[kb,kb,la,la] += -0.5
				S2_tbt[la,la,kb,kb] += -0.5
			end
		end
	end

	S2_tbt /= 4

	return S2_tbt
end

function hubbard_bliss_sym_params_to_F_OP(ovec, pvec, tvec, η, sz, s2, N = Int((sqrt(8*length(ovec)+1) - 1)/2), Sz = Sz_builder(N), S2=S2_builder(N))
	#builds S symmetry shift corresponding to S = s0+s1+s2
	#includes Sz and Sˆ2 contributions, considers spin-orb=true
	O_obt = zeros(N, N)
	idx = 0
	for i in 1:N
		for j in 1:i
			idx += 1
			O_obt[i,j] += ovec[idx]
			O_obt[j,i] += ovec[idx]
		end
	end

	P_obt = zeros(N, N)
	idx = 0
	for i in 1:N
		for j in 1:i
			idx += 1
			P_obt[i,j] += pvec[idx]
			P_obt[j,i] += pvec[idx]
		end
	end

	s0 = [-tvec[1]*(η^2) - tvec[2]*η - tvec[3]*sz - tvec[4]*(sz^2) - tvec[5]*s2]
	
	s1 = tvec[2]*collect(Diagonal(ones(N))) - η*O_obt - sz*P_obt - tvec[3]*Sz

	s2 = zeros(N,N,N,N)
	for i in 1:N
		for j in 1:N
			s2[i,i,j,j] += tvec[1]
			for k in 1:N
				s2[i,j,k,k] += 0.5*O_obt[i,j]
				s2[k,k,i,j] += 0.5*O_obt[i,j]
				for l in 1:N
					s2[i,j,k,l] += tvec[4] * Sz[i,j] * Sz[k,l]
					s2[i,j,k,l] += tvec[5] * S2[i,j,k,l]
					s2[i,j,k,l] += P_obt[i,j] * Sz[k,l]
				end
			end
		end
	end

	return F_OP((s0, s1, s2), true)
end

function hubbard_bliss_optimizer(F :: F_OP, η, sz, s2; verbose=true, SAVELOAD = SAVING, SAVENAME=DATAFOLDER*"BLISS.h5")
	if SAVELOAD
		fid = h5open(SAVENAME, "cw")
		if haskey(fid, "BLISS")
			BLISS_group = fid["BLISS"]
			if haskey(BLISS_group, "ovec")
				println("Loading results for BLISS optimization from $SAVENAME")
				ovec = read(BLISS_group,"ovec")
				tvec = read(BLISS_group,"tvec")
				pvec = read(BLISS_group,"pvec")
				close(fid)
				S = hubbard_bliss_sym_params_to_F_OP(ovec, pvec, tvec, η, sz, s2, F.N)
				close(fid)
				return F-S
			end
		end
		close(fid)
	end

	L = Int(F.N*(F.N+1)/2)
	x0 = zeros(2L + 5)

	if F.spin_orb == true
		n_qubits = F.N
	else
		n_qubits = 2*F.N
	end
	Sz = Sz_builder(n_qubits)
	S2 = S2_builder(n_qubits)


	function cost(x)
		return PAULI_L1(F - hubbard_bliss_sym_params_to_F_OP(x[1:L], x[L+1:2L], x[2L+1:end], η, sz, s2, F.N))
	end

	if verbose
		println("Starting 1-norm cost:")
		@show cost(x0)
		@time sol = optimize(cost, x0, BFGS(), Optim.Options(f_tol = 1e-3))
		println("Final 1-norm cost:")
		@show sol.minimum
	else
		sol = optimize(cost, x0, BFGS())
	end

	xsol = sol.minimizer
	if SAVELOAD
		fid = h5open(SAVENAME, "cw")
		create_group(fid, "BLISS")
		BLISS_group = fid["BLISS"]
		println("Saving results of BLISS optimization to $SAVENAME")
		BLISS_group["ovec"] = xsol[1:L]
		BLISS_group["pvec"] = xsol[L+1:2L]
		BLISS_group["tvec"] = xsol[2L+1:end]
		close(fid)
	end
	S = hubbard_bliss_sym_params_to_F_OP(xsol[1:L], xsol[L+1:2L], xsol[2L+1:end], η, sz, s2, F.N)

	return F - S
end

function bliss_linprog(F :: F_OP, η; model="highs", verbose=true)
	@warn "Linear programming function under construction, not working!"
	if F.spin_orb
		error("BLISS not defined for spin-orb=true!")
	end

    if model == "highs"
        L1_OPT = Model(HiGHS.Optimizer)
    elseif model == "ipopt"
        L1_OPT = Model(Ipopt.Optimizer)
    else
        error("Not defined for model = $model")
    end
    
    if verbose == false
        set_silent(L1_OPT)
    end

    ovec_len = Int(F.N*(F.N+1)/2)

    ν1_len = F.N^2
    ν2_len = F.N^4
    ν3_len = Int((F.N*(F.N-1)/2)^2)
    
    @variables(L1_OPT, begin
        t[1:2]
        obt[1:ν1_len]
        tbt1[1:ν2_len]
        tbt2[1:ν3_len]
        omat[1:ν1_len]
    end)

    @objective(L1_OPT, Min, sum(obt)+sum(tbt1)+sum(tbt2))

    obt_corr = ob_correction(F)
    #1-body 1-norm
    λ1 = zeros(ν1_len)
    idx = 0
    for i in 1:F.N
    	for j in 1:F.N
    		idx += 1
    		λ1[idx] = F.mbts[2][i,j] + obt_corr[i,j]
    	end
    end

    τ_11 = zeros(ν1_len)
    idx = 0
    for i in 1:F.N
    	for j in 1:F.N
    		idx += 1
    		if i == j
    			τ_11[idx] = 2*F.N
    		end
    	end
    end
    τ_12 = zeros(ν1_len)
    idx = 0
    for i in 1:F.N
    	for j in 1:F.N
    		idx += 1
    		if i == j
    			τ_11[idx] = 1
    		end
    	end
    end
    T1 = zeros(ν1_len,ν1_len)
    T1 += Diagonal((η - F.N/2)*ones(ν1_len))
    idx1 = 0
    for i in 1:F.N
    	for j in 1:F.N
    		idx1 += 1
    		idx2 = 0
    		for k in 1:F.N
    			for l in 1:F.N
    				idx2 += 1
    				if i == j && k == l
 	   					T1[idx1,idx2] -= 1
 	   				end
 	   			end
 	   		end
 	   	end
 	end
 	@constraint(L1_OPT, low_1, λ1 - τ_11*t[1] - τ_12*t[2] + T1*omat - obt .<= 0)
 	@constraint(L1_OPT, high_1, λ1 - τ_11*t[1] - τ_12*t[2] + T1*omat + obt .<= 0)

 	#2-body αβ/βα 1-norm
 	λ2 = zeros(ν2_len)
    idx = 0
    for i in 1:F.N
    	for j in 1:F.N
    		for k in 1:F.N
    			for l in 1:F.N
    				idx += 1
    				λ2[idx] = 0.5 * F.mbts[3][i,j,k,l]
    			end
    		end
    	end
    end

    τ_21 = zeros(ν2_len)
    idx = 0
    for i in 1:F.N
    	for j in 1:F.N
    		for k in 1:F.N
    			for l in 1:F.N
    				idx += 1
    				if i == j && k == l
    					τ_21[idx] = 0.5
    				end
    			end
    		end
    	end
    end

    T2 = zeros(ν2_len,ν1_len)
    idx = 0
    idx_ij = 0
    for i in 1:F.N
    	for j in 1:F.N
    		idx_ij += 1
    		idx_kl = 0
    		for k in 1:F.N
    			for l in 1:F.N
    				idx += 1
    				idx_kl += 1
    				if i == j
    					T2[idx,idx_kl] += 1
    				end
    				if k == l
    					T2[idx,idx_ij] += 1
    				end
    			end
    		end
    	end
    end
    @constraint(L1_OPT, low_2, λ2 - τ_21*t[1] - 0.5*T2*omat - tbt1 .<= 0)
    @constraint(L1_OPT, high_2, λ2 - τ_21*t[1] - 0.5*T2*omat + tbt1 .<= 0)

    #2-body αα/ββ 1-norm
    λ3 = zeros(ν3_len)
    idx = 0
    for i in 1:F.N
    	for j in 1:F.N
    		for k in 1:i-1
    			for l in 1:j-1
    				idx += 1
    				λ3[idx] = F.mbts[3][i,j,k,l] - F.mbts[3][i,l,k,j]
    			end
    		end
    	end
    end
    
    τ_31 = zeros(ν3_len)
    idx = 0
    for i in 1:F.N
    	for j in 1:F.N
    		for k in 1:i-1
    			for l in 1:j-1
    				idx += 1
    				if i == j && k == l
    					τ_31[idx] += 1
    				end
    				if i == l && k == j
    					τ_31[idx] -= 1
    				end
    			end
    		end
    	end
    end
    
    T_dict = zeros(Int64,F.N,F.N)
    idx = 0
    for i in 1:F.N
    	for j in 1:F.N
    		idx += 1
    		T_dict[i,j] = idx
    	end
    end

    T3 = zeros(ν3_len,ν1_len)
    idx = 0
    for i in 1:F.N
    	for j in 1:F.N
    		for k in 1:i-1
    			for l in 1:j-1
    				idx += 1
    				
    				idx_ij = T_dict[i,j]
    				if k == l
    					T3[idx,idx_ij] += 1
    				end
    				
    				idx_kl = T_dict[k,l]
    				if i == j
    					T3[idx,idx_kl] += 1
    				end

    				idx_il = T_dict[i,l]
    				if k == j
    					T3[idx,idx_il] -= 1
    				end

    				idx_kj = T_dict[k,j]
    				if i == l
    					T3[idx,idx_kj] -= 1
    				end
    			end
    		end
    	end
    end

    @constraint(L1_OPT, low_3, λ3 - τ_31*t[1] - 0.5*T3*omat - tbt2 .<= 0)
    @constraint(L1_OPT, high_3, λ3 - τ_31*t[1] - 0.5*T3*omat + tbt2 .<= 0)

    optimize!(L1_OPT)

    t_opt = value.(t)
    o_opt = value.(omat)
    O = zeros(F.N,F.N)
    for i in 1:F.N
    	for j in 1:F.N
    		O[i,j] = o_opt[T_dict[i,j]]
    	end
    end

    Ne,Ne2 = symmetry_builder(F)
    
    s2_tbt = t_opt[1] * Ne2.mbts[3]
    for i in 1:F.N
    	for j in 1:F.N
    		for k in 1:F.N
    			s2_tbt[i,j,k,k] += 0.5*O[i,j]
    			s2_tbt[k,k,i,j] += 0.5*O[i,j]
    		end
    	end
    end
    s2 = F_OP(([0],[0],s2_tbt))

    s1_obt = t_vec[2]*Ne.mbts[2] - η*O
    s1 = F_OP(([0],s1_obt))

    return F - s2 - s1, s1+s2
end

