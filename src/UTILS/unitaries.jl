# all functions relating to implementation and representation of unitary operators
function one_body_unitary(U :: f_matrix_rotation)
	return U.mat
end

function one_body_unitary(U :: givens_real_orbital_rotation)
	#returns one-body matrix corresponding to unitary real rotation (SO(N))
	Urot = collect(Diagonal(ones(U.N)))
	n_tot = length(U.θs)

	idx = 1
	for i in 1:U.N
		for j in i+1:U.N
			Ug = collect(Diagonal(ones(U.N)))
			Ug[i,i] = cos(U.θs[idx])
			Ug[j,j] = Ug[i,i]
			Ug[i,j] = sin(U.θs[idx])
			Ug[j,i] = -Ug[i,j]
			Urot = Ug * Urot
			idx += 1
		end
	end 

	return Urot
end

function one_body_unitary(U :: real_orbital_rotation)
	#returns one-body matrix corresponding to unitary real rotation (SO(N))
	Ulog = zeros(U.N, U.N)
	n_tot = length(U.θs)

	idx = 1
	for i in 1:U.N
		for j in i+1:U.N
			Ulog[i,j] = U.θs[idx]
			Ulog[j,i] = -U.θs[idx]
			idx += 1
		end
	end 

	return exp(Ulog)
end

function real_orbital_rotation_num_params(N)
	#returns number of rotation parameters for SO(N) unitary rotaion
	return Int(N*(N-1)/2)
end

function one_body_unitary(U :: givens_orbital_rotation)
	#returns one-body matrix corresponding to unitary rotation (SU(N))
	Urot = collect(Diagonal(ones(Complex,U.N)))
	n_tot = length(U.θs)

	idx = 1
	for i in 1:U.N
		for j in i+1:U.N
			Ug = collect(Diagonal(ones(U.N)))
			Ug[i,i] = cos(U.θs[idx])
			Ug[j,j] = Ug[i,i]
			Ug[i,j] = sin(U.θs[idx])
			Ug[j,i] = -Ug[i,j]
			Urot = Ug * Urot
			idx += 1
		end
	end 
	for i in 1:U.N
		for j in i+1:U.N
			Ug = collect(Diagonal(ones(Complex,U.N)))
			Ug[i,i] = cos(U.θs[idx])
			Ug[j,j] = Ug[i,i]
			Ug[i,j] = Ug[j,i] = -1im*sin(U.θs[idx])
			Urot = Ug * Urot
			idx += 1
		end
	end 
	for i in 1:U.N
		Ug = collect(Diagonal(ones(Complex,U.N)))
		Ug[i,i] = exp(1im*U.θs[idx])
		Urot = Ug*Urot
		idx += 1
	end
	
	return Urot
end

function orbital_rotation_num_params(N)
	#returns number of rotation parameters for SO(N) unitary rotaion
	return Int(N^2)
end

function SOn_to_MAMBO_greedy(Uobj, tol = ϵ_Givens)
	#uses greedy non-linear optimization to decompose SO(N) unitary into Givens rotations
	N = size(Uobj)[1]

	num_G = Int(N*(N-1)/2)
	G = zeros(num_G)

	u_curr = givens_real_orbital_rotation(N, G)
	num_iterations = 0
	for i in 1:num_G
		num_iterations += 1
		x0 = G[1:num_iterations]
		function cost(x)
			G[1:num_iterations] = x
			Ucurr = one_body_unitary(u_curr)
			return sum(abs2.(Ucurr - Uobj))
		end

		sol = optimize(cost, x0, BFGS())
		G[1:num_iterations] = sol.minimizer
		
		if sol.minimum < tol
			break
		end
	end


	fin_cost = sum(abs2.(one_body_unitary(u_curr) - Uobj))
	if fin_cost > tol
		println("Warning, Givens decomposition of orbital rotation did not achieve target accuracy")
		@show fin_cost
	end
	return u_curr
end

function SOn_unitary_to_params(U)
	#obtains parameters from SO(N) unitary
	#corresponds to E^p_q matrix G coefficients such that e^G = U
	N = size(U)[1]

	u_num  = Int(N*(N-1)/2)
	U_log = real.(log(U))
	u_params = zeros(u_num)
	idx = 0
	for i in 1:N
		for j in i+1:N
			idx += 1
			u_params[idx] = U_log[i,j]
		end
	end

	return u_params
end

function SOn_to_MAMBO_full(Uobj, tol = ϵ_Givens; verbose=true)
	#uses greedy non-linear optimization to find Givens decomposition of Uobj
	N = size(Uobj)[1]

	num_G = Int(N*(N-1)/2)
	G = zeros(num_G)

	u_curr = givens_real_orbital_rotation(N, G)
	function cost(x)
		G .= x
		Umat = one_body_unitary(u_curr)
		return sum(abs2.(Umat - Uobj))
	end

	x0 = copy(G)
	sol = optimize(cost, x0, BFGS())

	fin_cost = cost(G)
	if fin_cost > tol
		if verbose
			println("Warning, Givens decomposition of orbital rotation did not achieve target accuracy")
			@show fin_cost
		end
	end
	return u_curr
end

function SUn_to_MAMBO_full(Uobj, tol = ϵ_Givens; verbose=true)
	#uses greedy non-linear optimization to find Givens decomposition of Uobj
	N = size(Uobj)[1]

	num_G = Int(N^2)
	G = zeros(num_G)

	u_curr = givens_orbital_rotation(N, G)
	function cost(x)
		G .= x
		Umat = one_body_unitary(u_curr)
		return sum(abs2.(Umat - Uobj))
	end

	x0 = copy(G)
	sol = optimize(cost, x0, BFGS())

	fin_cost = cost(G)
	if fin_cost > tol
		if verbose
			println("Warning, Givens decomposition of orbital rotation did not achieve target accuracy")
			@show fin_cost
		end
	end
	return u_curr
end

function SOn_params_to_MAMBO(params)
	#rotates parameters coming from SO(n) unitary into Givens rotations parameters
	num_params = length(params)
	N = Int((1+sqrt(1+8*num_params))/2)

	U = one_body_unitary(real_orbital_rotation(N, params))
	return restricted_orbital_rotation(N, SOn_to_MAMBO_full(U))
end

function one_body_unitary(U :: restricted_orbital_rotation)
	Urot = collect(Diagonal(ones(U.N)))
	n_tot = length(U.θs)

	for i in 1:U.N-1
		Ug = collect(Diagonal(ones(U.N)))
		Ug[1,1] = cos(U.θs[i])
		Ug[i,i] = Ug[i,i]
		Ug[1,i] = sin(U.θs[i])
		Ug[i,1] = -Ug[1,i]
		Urot = Ug * Urot
	end 

	return Urot
end

function one_body_rotation_coeffs(U :: single_majorana_rotation)
	#γ_u⃗ = ∑_n u_n γ_n, with ∑_n|u_n|ˆ2 = 1
	# u1 = cos(2θ_1), u2 = sin(2θ_2)cos(2θ_1), ..., ui = cos(2θ_i)∏_{j<i}sin(2θ_j), ...
	u_coeffs = zeros(U.N)

	u_coeffs[1] = cos(2*U.θs[1])
	for i in 2:U.N-1
		u_coeffs[i] = cos(2*U.θs[i]) * prod(sin.(2*U.θs[1:i-1]))
	end
	u_coeffs[end] = prod(sin.(2*U.θs))

	return u_coeffs
end

function cartan_tbt_complex_rotation(Umat :: Array, tbt, n = size(tbt)[1])
	#rotates cartan tbt  with singles rotation Umat (i.e. one-body tensor)
	rotated_tbt = zeros(Complex,n,n,n,n)
	@einsum rotated_tbt[a,b,c,d] = Umat[a,l] * conj(Umat[b,l]) * Umat[c,m] * conj(Umat[d,m]) * tbt[l,l,m,m]

	return rotated_tbt
end

function cartan_tbt_complex_rotation(U :: F_UNITARY, tbt)
	return cartan_tbt_complex_rotation(one_body_unitary(U), tbt, U.N)
end

function cartan_tbt_rotation(Umat :: Array, tbt, n = size(tbt)[1])
	#rotates cartan tbt  with singles rotation Umat (i.e. one-body tensor)
	rotated_tbt = zeros(Float64,n,n,n,n)
	@einsum rotated_tbt[a,b,c,d] = Umat[a,l] * Umat[b,l] * Umat[c,m] * Umat[d,m] * tbt[l,l,m,m]

	return rotated_tbt
end

function cartan_tbt_rotation(U :: F_UNITARY, tbt)
	return cartan_tbt_rotation(one_body_unitary(U), tbt, U.N)
end

function tbt_rotation(Umat :: Array, tbt, n = size(tbt)[1])
	#rotates arbitrary tbt with singles rotation Umat (i.e. one-body tensor)
	rotated_tbt = zeros(Float64,n,n,n,n)
	@einsum rotated_tbt[a,b,c,d] = Umat[a,l] * Umat[b,k] * Umat[c,m] * Umat[d,p] * tbt[l,k,m,p]
	
	return rotated_tbt
end

function tbt_rotation(U :: F_UNITARY, tbt)
	return tbt_rotation(one_body_unitary(U), tbt, U.N)
end

function tbt_complex_rotation(Umat :: Array, tbt, n = size(tbt)[1])
	#rotates arbitrary tbt with singles rotation Umat (i.e. one-body tensor)
	rotated_tbt = zeros(Complex,n,n,n,n)
	@einsum rotated_tbt[a,b,c,d] = Umat[a,l] * conj(Umat[b,k]) * Umat[c,m] * conj(Umat[d,p]) * tbt[l,k,m,p]
	
	return rotated_tbt
end

function tbt_complex_rotation(U :: F_UNITARY, tbt)
	return tbt_complex_rotation(one_body_unitary(U), tbt, U.N)
end

function cartan_obt_rotation(Umat :: Array, obt, n=size(obt)[1])
	#rotates diagonal obt with singles rotation Umat
	rotated_obt = zeros(Float64,n,n)
	@einsum rotated_obt[a,b] = Umat[a,i] * Umat[b,i] * obt[i,i]

	return rotated_obt
end

function cartan_obt_rotation(U :: F_UNITARY, obt)
	return cartan_obt_rotation(one_body_unitary(U), obt, U.N)
end

function cartan_obt_complex_rotation(Umat :: Array, obt, n=size(obt)[1])
	#rotates diagonal obt with singles rotation Umat
	rotated_obt = zeros(Complex,n,n)
	@einsum rotated_obt[a,b] = Umat[a,i] * conj(Umat[b,i]) * obt[i,i]

	return rotated_obt
end

function cartan_obt_complex_rotation(U :: F_UNITARY, obt)
	return cartan_obt_complex_rotation(one_body_unitary(U), obt, U.N)
end

function obt_rotation(Umat :: Array, obt, n=size(obt)[1])
	#rotates arbitrary obt with singles rotation Umat
	rotated_obt = zeros(Float64,n,n)
	@einsum rotated_obt[a,b] = Umat[a,i] * Umat[b,j] * obt[i,j]

	return rotated_obt
end

function obt_rotation(U :: F_UNITARY, obt)
	return obt_rotation(one_body_unitary(U), obt, U.N)
end

function obt_complex_rotation(Umat :: Array, obt, n=size(obt)[1])
	#rotates arbitrary obt with singles rotation Umat
	rotated_obt = zeros(Complex,n,n)
	@einsum rotated_obt[a,b] = Umat[a,i] * conj(Umat[b,j]) * obt[i,j]

	return rotated_obt
end

function obt_complex_rotation(U :: F_UNITARY, obt)
	return obt_complex_rotation(one_body_unitary(U), obt, U.N)
end

function real_orb_rot_composer(R1 :: real_orbital_rotation, R2 :: real_orbital_rotation)
	#returns real_orbital_rotation corresponding to R1*R2
	U1 = one_body_unitary(R1)
	U2 = one_body_unitary(R2)

	Uprod = U1 * U2

	return SOn_to_MAMBO_full(Uprod)
end

function real_orb_rot_composer(U1 :: Array{Float64,2}, R2 :: real_orbital_rotation)
	#returns real_orbital_rotation corresponding to R1*R2
	U2 = one_body_unitary(R2)

	Uprod = U1 * U2

	return SOn_to_MAMBO_full(Uprod)
end

function real_orb_rot_composer(R1 :: real_orbital_rotation, U2 :: Array{Float64,2})
	#returns real_orbital_rotation corresponding to R1*R2
	U1 = one_body_unitary(R1)
	
	Uprod = U1 * U2

	return SOn_to_MAMBO_full(Uprod)
end

function F_OP_rotation(Umat :: Array, F :: F_OP)
	if F.Nbods ≥ 3
		error("Trying to do orbital rotation of fermionic operator with 3 or more bodies, not implemented!")
	end

	if F.filled[2]
		obt_rot = obt_rotation(Umat, F.mbts[2])
	else
		obt_rot = [0]
	end

	if F.filled[3]
		tbt_rot = tbt_rotation(Umat, F.mbts[3])
	else
		tbt_rot = [0]
	end

	return F_OP((F.mbts[1], obt_rot, tbt_rot), F.spin_orb)
end

function F_OP_rotation(U :: F_UNITARY, F :: F_OP)
	return F_OP_rotation(one_body_unitary(U), F)
end

function get_anti_symmetric(n, N = Int(n*(n-1)/2))  
	# Construct list of anti-symmetric matrices kappa_pq based on n*(n-1)/2 
	R = zeros(N,n,n)
	idx = 1
	for p in 1:n
		for q in p+1:n
			R[idx,p,q] = 1
			R[idx,q,p] = -1
			idx += 1
		end
	end

	return R
end

function construct_anti_symmetric(n, params, N=length(params))
	#Construct the nxn anti-symmetric matrix based on the sum of basis with params as coefficients
	real_anti = get_anti_symmetric(n, N)
	anti_symm = zeros(n,n)
	for idx in 1:N
		anti_symm += params[idx] * real_anti[idx,:,:]
	end

	return anti_symm
end

function get_generator(n,idx)  #Generators for real orbital rotations
	i=Int64(ceil(((2n-1)-sqrt((1-2n)^2-8*idx))/2))
	i_prev=i-1
	j=i+Int64(idx-i_prev*n+i_prev*(i_prev+1)/2)
	generator=zeros(Int64,n,n)
	generator[i,j]=1
	generator[j,i]=-1
	return generator
end
