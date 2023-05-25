#different ways of finding initial guess for greedy optimizations

function SVD_to_CSA(λ, ω, U; debug=false)
	#transform λ value, ω vec and U rotation coming from SVD into CSA fragment parameters
	#returns parameters for Givens rotations and Cartan coeffs giving largest SVD fragment
	N = size(U)[1]

	cartan_L = Int(N*(N+1)/2)
	coeffs = zeros(cartan_L)


	idx = 1
	for i in 1:N
		for j in i:N
			coeffs[idx] = ω[i]*ω[j]
			idx += 1
		end
	end
	coeffs .*= λ

	u_rot = SOn_to_MAMBO_full(U, verbose=false)
	C = cartan_2b(false, coeffs, N)
	frag = F_FRAG(1, tuple(u_rot), CSA(), C, N, false, 1, false)

	if debug == true
		tbt_svd_CSA = zeros(typeof(ω[1]),N,N,N,N)
		for i1 in 1:N
		   	tbt_svd_CSA[i1,i1,i1,i1] = ω[i1]^2
	    end

	    for i1 in 1:N
	    	for i2 in i1+1:N
	    		tbt_svd_CSA[i1,i1,i2,i2] = ω[i1]*ω[i2]
	    		tbt_svd_CSA[i2,i2,i1,i1] = ω[i1]*ω[i2]
	    	end
	    end
		    
	    tbt_svd_CSA .*= λ
	    tbt_svd = cartan_tbt_rotation(U, tbt_svd_CSA)

	    tbt_frag = to_OP(frag).mbts[3]
	    @show sum(abs.(tbt_svd - tbt_frag))
	end

	return frag
end

function tbt_svd_1st(tbt :: Array; debug=false, tiny=SVD_tiny)
	#returns the largest SVD component from tbt in F_FRAG format
	#println("Starting tbt_svd_1st routine!")
	n = size(tbt)[1]
	N = n^2

	tbt_res = Symmetric(reshape(tbt, (N,N)))

	#println("Diagonalizing")
	Λ,U = eigen(tbt_res)
	ind=sortperm(abs.(Λ))[end:-1:1]
    Λ = Λ[ind]
    U=U[:,ind]
	#@show Λ

	## U*Diagonal(Λ)*U' == tbt_res

	full_l = reshape(U[:, 1], (n,n))
    cur_l = Symmetric(full_l)

    sym_dif = sum(abs.(cur_l - full_l))
    if sym_dif > tiny
      	if sum(abs.(full_l + full_l')) > tiny
			error("SVD operator is neither Hermitian or anti-Hermitian, cannot do double factorization into Hermitian fragment!")
		end
      	
       	cur_l = Hermitian(1im * full_l)
       	Λ[1] *= -1
    end

    ωl, Ul = eigen(cur_l)
    
    return SVD_to_CSA(Λ[1], ωl, Ul)
end

function tbt_svd_avg(tbt :: Array, svd_coeff = 1e-5; debug=false, tiny=SVD_tiny)
	#returns the largest SVD component from tbt in F_FRAG format
	#println("Starting tbt_svd_1st routine!")
	n = size(tbt)[1]
	N = n^2

	tbt_res = Symmetric(reshape(tbt, (N,N)))

	#println("Diagonalizing")
	Λ,U = eigen(tbt_res)
	ind=sortperm(abs.(Λ))[end:-1:1]
    Λ = Λ[ind]
    U=U[:,ind]
	#@show Λ

	## U*Diagonal(Λ)*U' == tbt_res
    full_1 = reshape(U[:, 1], (n,n))
    cur_1 = Symmetric(full_1)
    sym_dif = sum(abs.(cur_1 - full_1))
    if sym_dif > tiny
      	if sum(abs.(full_1 + full_1')) > tiny
			error("SVD operator is neither Hermitian or anti-Hermitian, cannot do double factorization into Hermitian fragment!")
		end      	
       	cur_1 = Hermitian(1im * full_1)
       	Λ[1] *= -1
	end
	ω1, U1 = eigen(cur_1)
	frag = SVD_to_CSA(Λ[1], ω1, U1)
	real_rot_1 = frag.U[1]
	tbt_curr = to_OP(frag).mbts[3]

    λs = zeros(Int(n*(n+1)/2))
    idx = 0
    for i in 1:n
    	for j in 1:i
    		idx += 1
    		λs[idx] += tbt_curr[i,i,j,j,]
    	end
    end

    ind_max = 1
    while abs(Λ[ind_max+1]) > svd_coeff
    	ind_max += 1

		full_l = reshape(U[:, ind_max], (n,n))
    	cur_l = Symmetric(full_l)

	    sym_dif = sum(abs.(cur_l - full_l))
	    if sym_dif > tiny
	      	if sum(abs.(full_l + full_l')) > tiny
				error("SVD operator is neither Hermitian or anti-Hermitian, cannot do double factorization into Hermitian fragment!")
			end
	      	
	       	cur_l = Hermitian(1im * full_l)
	       	Λ[ind_max] *= -1
	    end

    	ωl, Ul = eigen(cur_l)
    	frag = SVD_to_CSA(Λ[ind_max], ωl, Ul)
    	new_U = real_orb_rot_composer(collect(U1'), frag.U[1])
    	F_new = F_FRAG(1, tuple(new_U), CSA(), frag.C, n, false, 1, false)
    	tbt_curr = to_OP(F_new).mbts[3]
    	idx = 0
    	for i in 1:n
    		for j in 1:i
    			idx += 1
    			λs[idx] += tbt_curr[i,i,j,j]
    		end
    	end
    end

    C = cartan_2b(false, λs, n)

    return F_FRAG(1,  tuple(real_rot_1), CSA(), C, n, false, 1, false)
end