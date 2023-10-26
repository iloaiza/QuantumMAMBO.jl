function mbt_schmidt(mbt :: Array{Float64}; debug=false)
	Smbt = size(mbt)
	if length(Smbt) == 1
		error("Trying to do Schmidt decomposition over 1-dimenional array!")
	end

	s1 = Smbt[1]
	s2 = prod(Smbt[2:end])

	mbt_reshape = reshape(mbt, (s1,s2))

	U,S,V = svd(mbt_reshape)

	if debug
		mbt_reb = 0 * mbt_reshape
		for k in 1:s1
			uk = U[:,k]
			sk = S[k]
			for p in 1:s1
				for qrs in 1:s2
					mbt_reb[p,qrs] += sk * uk[p] * V[qrs,k]
				end
			end
		end

		@show sum(abs.(mbt_reshape - mbt_reb))
		@show sum(abs.(reshape(mbt_reb,Smbt) - mbt))
	end

	return U,S,V
end

function r_acos(x, tiny = 1e-7)
	#returns acos(x), if abs(x) is close to ±1 by tiny rounds x to ±1
	if abs(x) > 1 + tiny
		error("acos not defined for x=$x")
	elseif abs(x) > 1
		if x > 0
			return acos(1)
		else
			return acos(-1)
		end
	else
		return acos(x)
	end
end

function r_asin(x, tiny = 1e-7)
	#returns asin(x), if abs(x) is close to ±1 by tiny rounds x to ±1
	if abs(x) > 1 + tiny
		error("asin not defined for x=$x")
	elseif abs(x) > 1
		if x > 0
			return asin(1)
		else
			return asin(-1)
		end
	else
		return asin(x)
	end
end

function majorana_coefs_to_unitary(cns, N = length(cns); tiny = 1e-10, debug=false)
	#obtains angles for transforming γ_c⃗ = ∑_n c_n γ_n into MTD_CP4 angles, where ∑_n |cns|ˆ2 = 1
	if abs(1-sum(abs2.(cns))) > tiny
		error("Trying to convert Majorana coeffs into angles, not normalized!")
	end

	θs = 0.5 * π * ones(N-1)
	θs[1] = r_acos(cns[1])/2

	for i in 2:N-1
		cum_coef = sum(abs2.(cns[1:i-1]))
		# =
		if abs(1-cum_coef) < tiny
			break
		end
		# =#
		if abs(cns[i]) > tiny
			ccur = cns[i]/(prod(sin.(2*θs[1:i-1])))
		else
			ccur = 0
		end
		θs[i] = r_acos(ccur)/2
	end

	if abs(cns[end]) > tiny
		θs[end] *= sign(cns[end])
	end
	
	U = single_majorana_rotation(N, θs)

	if debug == true
		u_dbg = one_body_rotation_coeffs(U)
		dif = sum(abs2.(u_dbg - cns))
		if dif > tiny
			@warn "Majorana coefficients transformed to single_majorana_rotation unitary not converged!"
			@show cns
			@show u_dbg
		end
	end

	return U
end


function iterative_schmidt(tbt :: Array{Float64, 4}; tol=1e-6, debug=false, return_ops=false, count=true)
	#return iterative schmidt decomposition of tbt
	U1, S1, V1 = mbt_schmidt(tbt)
	N = size(tbt)[1]

	S2s = zeros(N, N)
	U2s = zeros(N, N, N)
	V2s = zeros(N^2, N, N)
	for i in 1:N
		U2s[:,:,i], S2s[:,i], V2s[:,:,i] = mbt_schmidt(reshape(V1[:,i], (N,N,N)))
	end

	S3s = zeros(N, N, N)
	U3s = zeros(N, N, N, N)
	V3s = zeros(N, N, N, N)
	for i in 1:N
		for j in 1:N
			U3s[:,:,j,i], S3s[:,j,i], V3s[:,:,j,i] = mbt_schmidt(reshape(V2s[:,j,i], (N,N)))
		end
	end

	if debug
		tbt_reb = zeros(N,N,N,N)
		for k1 in 1:N
			s1 = S1[k1]
			for p in 1:N
				u1 = U1[p,k1]
				for k2 in 1:N
					s2 = S2s[k2,k1]
					for q in 1:N
						u2 = U2s[q,k2,k1]
						for k3 in 1:N
							s3 = S3s[k3,k2,k1]
							for r in 1:N
								u3 = U3s[r,k3,k2,k1]
								for s in 1:N
									u4 = V3s[s,k3,k2,k1]
									tbt_reb[p,q,r,s] += s1*u1*s2*u2*s3*u3*u4
								end
							end
						end
					end
				end
			end
		end

		@show sum(abs.(tbt_reb - tbt))
	end

	if return_ops
		FRAGS = F_FRAG[]
		for k1 in 1:N
			s1 = S1[k1]
			for k2 in 1:N
				s2 = S2s[k2,k1]
				for k3 in 1:N
					s3 = S3s[k3,k2,k1]
					u1 = majorana_coefs_to_unitary(U1[:,k1])
					u2 = majorana_coefs_to_unitary(U2s[:,k2,k1])
					u3 = majorana_coefs_to_unitary(U3s[:,k3,k2,k1])
					u4 = majorana_coefs_to_unitary(V3s[:,k3,k2,k1])

					Us = (u1,u2,u3,u4)

					frag = F_FRAG(4, Us, MTD_CP4(), cartan_m1(), N, false, s1*s2*s3, true)
					push!(FRAGS, frag)
				end
			end
		end

		OP_TOT = sum(to_OP.(FRAGS))

		if debug
			@show sum(abs2.(OP_TOT.mbts[3] - tbt))
		end

		return FRAGS
	end

	Stot = zeros(N, N, N)
	discarded = 0
	accepted = 0
	for k1 in 1:N
		s1 = S1[k1]
		for k2 in 1:N
			s2 = S2s[k2,k1]
			for k3 in 1:N
				s3 = S3s[k3,k2,k1]
				scurr = s1*s2*s3
				if abs(scurr) < tol
					discarded += 4
				else
					accepted += 4
					Stot[k3,k2,k1] = scurr
				end
			end
		end
	end

	if count
		return [sum(abs.(Stot)), accepted]
	else
		return sum(abs.(Stot))
	end
end

function split_schmidt(tbt :: Array{Float64, 4}; tol=1e-6, debug=false, count=true)
	#return iterative schmidt decomposition of tbt for αααα/ββββ and ααββ/ββαα sectors separated
	l_hetero, accepted_hetero = iterative_schmidt(tbt, tol=tol, debug=debug, count=true)

	tbt_homo = copy(tbt)
	N = size(tbt)[1]
	for i in 1:N
		for j in 1:N
			for k in 1:N
				tbt_homo[i,j,i,k] = 0
				tbt_homo[j,i,k,i] = 0
			end
		end
	end

	l_homo, accepted_homo = iterative_schmidt(tbt_homo, tol=tol, debug=debug, count=true)

	if count
		return [(l_hetero + l_homo)/2, (accepted_hetero + accepted_homo)/2]
	else
		return (l_hetero + l_homo)/2
	end
end