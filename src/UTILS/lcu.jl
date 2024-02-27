#module for creating LCU from fragments and calculating the resulting 1-norm
function L1(F :: F_FRAG; debug = false, count = false)
	#calculates 1-norm of fragment
	#debug: run debugging routines
	if F.TECH == CSA()
		return CSA_L1(F, debug = debug, count = count)
	elseif F.TECH == DF()
		return DF_L1(F, debug = debug, count = count)
	elseif F.TECH == THC()
		return THC_L1(F, debug = debug, count = count)
	elseif F.TECH == OBF()
		return OBF_L1(F, debug = debug, count = count)
	else
		error("Trying to calculate LCU 1-norm decomposition for fermionic fragment with FRAGMENTATION_TECH=$(F.TECH), not implemented!")
	end
end

function Fock_bound(H :: F_OP)
	#gives a lower bound for the spectral range (Emax-Emin)/2 from Fock matrix
	obt = H.mbts[2]
	if H.Nbods == 2
		obt += ob_correction(H)
	end

	Hob = F_OP(obt, H.spin_orb)
	frag = to_OBF(Hob)

	return OBF_L1(frag, count=false)
end

function L1(F_arr :: Array{F_FRAG}; debug=false, count=true)
	return sum(L1.(F_arr, debug=debug, count=count))
end

function SQRT_L1(F :: F_OP; count = false, tol=1e-3, verbose=false)
	#return minimal 1-norm for fermionic operator, does not scale well!
	if verbose == false
		mat = to_matrix(F)
		range = mat_range(mat, tol=tol)
	else
		println("Building sparse matrix...")
		@time mat = to_matrix(F)
		println("Obtaining matrix spectral range...")
		@time range = mat_range(mat, tol=tol)
	end
	spec_range = (range[2] - range[1])/2

	if count
		return [spec_range, 2]
	else
		return spec_range
	end
end

function SQRT_L1(F :: F_FRAG; count = false)
	#lower bound of the 1-norm of a fragment, removes one-body correction
	if typeof(F.TECH) == CSA
		return SQRT_CSA_L1(F, count=count)
	elseif typeof(F.TECH) == DF
		return DF_L1(F, count=count)
	elseif typeof(F.TECH) == OBF
		return OBF_L1(F, count=count)
	else
		Q_OP = qubit_transform(to_OF(F))
		obt_corr = ob_correction(F)
		Q_OP -= qubit_transform(to_OF(F_OP(obt_corr)))
		
		range = OF_qubit_op_range(Q_OP)
		spec_range = (range[2] - range[1])/2

		if count
			return [spec_range, 2]
		else
			return spec_range
		end
	end
end

function SQRT_CSA_L1(F :: F_FRAG; count = false)
	#lower bound for CSA fragment
	if F.spin_orb
		n_so = F.N
		λ_corr = zeros(F.N, F.N)
		idx = 0
		for i in 1:F.N
			for j in 1:i
				idx += 1
				if i != j
					λ_corr[i,j] = λ_corr[j,i] = F.C.λ[idx]
				end
			end
		end
	else
		n_so = 2*F.N
		λ_mat = zeros(F.N, F.N)
		idx = 0
		for i in 1:F.N
			for j in 1:i
				idx += 1
				λ_mat[i,j] = λ_mat[j,i] = F.C.λ[idx]
			end
		end

		λ_corr = zeros(2*F.N, 2*F.N)
		for i in 1:F.N
			for j in 1:F.N
				for a in -1:0
					for b in -1:0
						λ_corr[2i+a,2j+b] = λ_corr[2j+b, 2i+a] = λ_mat[i,j]
					end
				end
			end
		end
		for i in 1:F.N
			for a in -1:0
				λ_corr[2i+a,2i+a] -= 2*sum(λ_mat[i,:])
			end
		end
	end

	E_VALS = zeros(2^(n_so))
	for i in 0:2^(n_so)-1
		n_arr = digits(i, base=2, pad=n_so)
		for j1 in 1:n_so
			for j2 in 1:n_so
				E_VALS[i+1] += λ_corr[j1,j2] * n_arr[j1] * n_arr[j2]
			end
		end
	end

	spec_range = (maximum(E_VALS) - minimum(E_VALS))/2
	if count
		return [spec_range, 2]
	else
		return spec_range
	end
end

function CSA_L1(F :: F_FRAG; debug = false, count = false)
	if F.spin_orb
		idx = 0
		l1 = 0.0
		for i in 1:F.C.N
			for j in 1:i
				idx += 1
				if i != j
					l1 += 0.5*abs(F.C.λ[idx])
				end
			end
		end

		if debug
			tbt = cartan_2b_to_tbt(F.C)
			for i in 1:F.N
				tbt[i,i,i,i] = 0
			end
			@show l1
			@show sum(abs.(tbt))/4
		end
	else
		idx = 0
		l1 = 0.0
		for i in 1:F.C.N
			for j in 1:i #sum i ≥ j
				idx += 1
				#λ_ij = λ_ji = F.C.λ[idx]
				if i != j
					#|λ_ij| + |λ_ji|
					l1 += 2*abs(F.C.λ[idx])
				else
					#|λ_ii|/2
					l1 += 0.5*abs(F.C.λ[idx])
				end
			end
		end

		if debug
			tbt = tbt_orb_to_so(cartan_2b_to_tbt(F.C))
			for i in 1:2F.N
				tbt[i,i,i,i] = 0
			end
			@show l1
			@show sum(abs.(tbt))/4
		end
	end

	if count
		N_Us = 0
		idx = 0
		for i in 1:F.C.N
			for j in 1:i
				idx += 1
				if i != j
					if abs(F.C.λ[idx]) > LCU_tol
						N_Us += 4
					end
				else
					if abs(F.C.λ[idx]) > LCU_tol
						N_Us += 2
					end
				end
			end
		end
		return [l1, N_Us]
	else
		return l1
	end
end

function DF_L1(F :: F_FRAG; debug = false, count = false)
	if F.spin_orb == false
		l1 = 0.5 * abs(F.coeff) * ((sum(abs.(F.C.λ)))^2)
		if count
			return [l1, 1]
		else
			return l1
		end
	else
		error("1-norm DF calculation not implemented for spin-orb=true")
	end
end

function THC_L1(F :: F_FRAG; debug = false, count = false)
	if F.spin_orb == false
		if count
			if abs(F.coeff) > LCU_tol
				return [abs(F.coeff), 4]
			else
				return [abs(F.coeff), 0]
			end
		else
			return abs(F.coeff)
		end
	else
		error("1-norm THC calculation not implemented for spin-orb=true")
	end
end

function OBF_L1(F :: F_FRAG; debug = false, count = false)
	if F.spin_orb == false
		l1 = sum(abs.(F.C.λ))
	else
		l1 = 0.5 * sum(abs.(F.C.λ))
	end

	if count
		N_Us = 0
		for λ_coeff in F.C.λ
			if abs(λ_coeff) > LCU_tol
				N_Us += 1
			end
		end
		if !F.spin_orb
			N_Us *= 2
		end
		return [l1, N_Us]
	else
		return l1
	end
end

function one_body_L1(H :: F_OP; count=false)
	#get one-body 1-norm after correction from 2-body term
	obf = to_OBF(H.mbts[2] + ob_correction(H), H.spin_orb)

	return OBF_L1(obf, count=count)
end

function PAULI_L1(Q :: Q_OP; count=false)
	l1 = 0.0

	if count
		N_Us = 0
	end

	for pw in Q.paulis
		l1 += abs(pw.coeff)
		if count
			if abs(pw.coeff) > LCU_tol
				N_Us += 1
			end
		end
	end

	if count
		return [l1, N_Us]
	else
		return l1
	end
end

function PAULI_L1(F :: F_OP; count=false)
	if count
		return PAULI_L1(Q_OP(F), count=true)
	end

	if F.spin_orb
		return PAULI_L1(Q_OP(F), count=false)
	end

	l1 = 0.0
	if F.Nbods > 2
		error("Trying to calculate Pauli cost for fermionic operator with more than 2-body terms, not implemented!")
	end
	
	if size(F.mbts[2],1)==size(F.mbts[3],1)
	
		if F.filled[2] && F.filled[3]
			obt_mod = F.mbts[2] + ob_correction(F)
			λ1 = sum(abs.(obt_mod))
		elseif F.filled[2]
			obt_mod = F.mbts[2]
			λ1 = sum(abs.(obt_mod))
		elseif F.filled[3]
			obt_mod = ob_correction(F)
			λ1 = sum(abs.(obt_mod))
		else
			λ1 = 0
		end
		
		if F.filled[3]
			λ2 = 0.5 * sum(abs.(F.mbts[3]))
			for r in 1:F.N
				for p in r+1:F.N
					for q in 1:F.N
						for s in q+1:F.N
							λ2 += abs(F.mbts[3][p,q,r,s] - F.mbts[3][p,s,r,q])
						end
					end
				end
			end
		else
			λ2 = 0
		end
	else
		if F.filled[2] && F.filled[3]
			obt_mod = F.mbts[2] + ob_correction(F)
			
			λ1 = sum(abs.(obt_mod))
		elseif F.filled[2]
			obt_mod = F.mbts[2]
			λ1 = sum(abs.(obt_mod))
		elseif F.filled[3]
			obt_mod = ob_correction(F)
			λ1 = sum(abs.(obt_mod))
		else
			λ1 = 0
		end
		λ1=λ1/2
		@show λ1
		if F.filled[3]
			λ2 = 0.25 * sum(abs.(F.mbts[3][2:3,:,:,:,:]))
			@show λ2
			temp=λ2
			arr_align=[1,4]
			for sigma in arr_align
				for r in 1:F.N
					for p in r+1:F.N
						for q in 1:F.N
							for s in q+1:F.N
								
								λ2 += 0.5*abs.(F.mbts[3][sigma,p,q,r,s] - F.mbts[3][sigma,p,s,r,q])
							end
						end
					end
				end
			end
		else
			λ2 = 0
		end
	end
		
		
	@show λ2-temp

	return λ1+λ2 
end

function PAULI_L2(F::F_OP; count=false)
	if count
		return PAULI_L1(Q_OP(F), count=true)
	end

	if F.spin_orb
		return PAULI_L1(Q_OP(F), count=false)
	end

	l1 = 0.0
	if F.Nbods > 2
		error("Trying to calculate Pauli cost for fermionic operator with more than 2-body terms, not implemented!")
	end
	
	if F.filled[2] && F.filled[3]
		obt_mod = F.mbts[2] + ob_correction(F)
		λ1 = sum(abs2.(obt_mod))
		
	elseif F.filled[2]
		obt_mod = F.mbts[2]
		λ1 = sum(abs2.(obt_mod))
	elseif F.filled[3]
		obt_mod = ob_correction(F)
		λ1 = sum(abs2.(obt_mod))
	else
		λ1 = 0
	end
	
	if F.filled[3]
		λ2 = 0.5 * sum(abs2.(F.mbts[3]))
		for r in 1:F.N
			for p in r+1:F.N
				for q in 1:F.N
					for s in q+1:F.N
						λ2 += abs2(F.mbts[3][p,q,r,s] - F.mbts[3][p,s,r,q])
					end
				end
			end
		end
	else
		λ2 = 0
	end
	return λ1+λ2

end
