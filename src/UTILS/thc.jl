using LeastSquaresOptim


#=
function of_thc(eri, nthc)
	ERI_THC, UMATS, ζ, INFO = of_thc.factorize(eri, nthc)
	
	@show sum(abs2.(eri - ERI_THC))
end
# =#

function THC_search(F :: F_OP, hop_num = 20; tol=ϵ, rank_max = 1000, verbose=true, RAND_START = false)
	if verbose
		println("Starting THC binary search routine...")
	end
	OB_ERI, TB_ERI = F_OP_to_eri(F)

	curr_cost = sum(abs2.(TB_ERI))
	curr_rank = 1

	ERI_THC = UMATS = ζ = 0

	while curr_rank < rank_max && curr_cost > tol
		curr_rank += hop_num
		if verbose
			println("Starting search for rank=$curr_rank")
			@time ERI_THC, UMATS, ζ, INFO = of_thc.factorize(TB_ERI, curr_rank, random_start_thc=RAND_START)
		else
			ERI_THC, UMATS, ζ, INFO = of_thc.factorize(TB_ERI, curr_rank, random_start_thc=RAND_START)
		end
		curr_cost = sum(abs2.(TB_ERI - ERI_THC))
		if verbose
			println("Current cost is $curr_cost")
		end
	end

	if verbose
		println("Finished hopped search for rank $curr_rank, final cost is $curr_cost")
	end

	if curr_cost > tol
		error("THC decomposition did not converge for rank=$curr_rank, try increasing maximum rank")
	end

	rank_min = maximum([1, curr_rank - hop_num+1])

	ERI_OLD = copy(ERI_THC)
	ζ_OLD = copy(ζ)
	UMATS_OLD = copy(UMATS)
	while curr_cost < tol && curr_rank > rank_min
		curr_rank -= 1
		if verbose
			println("Starting search for rank=$curr_rank")
			@time ERI_THC, UMATS, ζ, INFO = of_thc.factorize(TB_ERI, curr_rank, random_start_thc=RAND_START)
		else
			ERI_THC, UMATS, ζ, INFO = of_thc.factorize(TB_ERI, curr_rank, random_start_thc=RAND_START)
		end
		curr_cost = sum(abs2.(TB_ERI - ERI_THC))
		if verbose
			println("Current cost is $curr_cost")
		end
		if curr_cost < tol
			ERI_OLD = ERI_THC
			ζ_OLD = ζ
			UMATS_OLD = UMATS
		else
			curr_rank += 1
		end
	end

	fin_cost = sum(abs2.(TB_ERI - ERI_OLD))
	println("Finished THC routine, final rank is $curr_rank with cost $fin_cost")

	return ERI_OLD, ζ_OLD, UMATS_OLD
end

function THC_to_eri(ζ, Uvecs)
	R,N = size(Uvecs)
	eri_ret = zeros(N,N,N,N)

	@einsum eri_ret[i,j,k,l] = ζ[r1,r2] * Uvecs[r1,i] * Uvecs[r1,j] * Uvecs[r2,k] * Uvecs[r2,l]

	return eri_ret
end

function THC_normalizer(ζ, Uvecs)
	R,N = size(Uvecs)

	ζ_norm = copy(ζ)
	U_norm = copy(Uvecs)
	for r in 1:R
		norm_const = sum(abs2.(Uvecs[r,:]))
		Uvecs[r,:] /= sqrt(norm_const)
		ζ_norm[r,:] *= sqrt(norm_const)
	end

	return ζ_norm, U_norm
end

function THC_full(F :: F_OP)
	OB_ERI, TB_ERI = F_OP_to_eri(F)

	eri, ζ, Us = THC_search(F)
	ζ, Us = THC_normalizer(ζ, Us)

	println("Final THC cost is:")
	@show sum(abs2.(THC_to_eri(ζ, Us) - TB_ERI))

	ζ /= 2
	@show sum(abs2.(THC_to_eri(ζ, Us) - F.mbts[3]))

	num_ops = length(ζ)
	λ2 = [sum(abs.(ζ)), num_ops]

	return λ2
end

function DF_to_THC(F :: F_OP; debug=true)
	DF_FRAGS = DF_decomposition(F)
	M = length(DF_FRAGS)
	N = DF_FRAGS[1].N

	ζ = zeros(M*N, M*N)
	US = zeros(M*N, N)

	MN_idx = Dict{Tuple{Int64, Int64}, Int64}()
	idx = 0
	for α in 1:M
		for i in 1:N
			idx += 1
			get!(MN_idx, (α,i), idx)
		end
	end

	for α in 1:M
		for i in 1:N
			Uα = one_body_unitary(DF_FRAGS[α].U[1])
			αi_idx = MN_idx[(α,i)]
			US[αi_idx, :] = Uα[:,i]
			for j in 1:N
				αj_idx = MN_idx[(α,j)]
				ζ[αi_idx, αj_idx] = DF_FRAGS[α].C.λ[i] * DF_FRAGS[α].C.λ[j] * DF_FRAGS[α].coeff
			end
		end
	end

	if debug
		op_df = sum(to_OP.(DF_FRAGS))
		tbt_df = op_df.mbts[3]
		tbt_thc = THC_to_eri(ζ, US)

		@show sum(abs2.(tbt_df - tbt_thc))
	end

	λ1 = one_body_L1(H, count=true)
	ζ = sparse(ζ)
	num_ops = length(ζ.nzval)
	λ2 = [sum(abs.(ζ)), num_ops]

	@show λ1, λ2
	@show λ1 + λ2

	return λ1 + λ2
end


function check_unitary(mat,N)
	trans=zeros(N,N)
	for i in 1:N
		for j in 1:N
			trans[i,j]=mat[j,i]
		end
	end
	prod=zeros(N,N)
	for i in 1:N
		for j in 1:N
			for k in 1:N
				prod[i,j]+=mat[i,k]*trans[k,j]
			end
		end
	end
	return prod
end


function Us_for_THC(N, a, x, lambda_L)
	Us=zeros(a,N)
	
	for i in 1:a
		Us[i,1]=1
		for j in 1:N-1
			Us[i,j+1]=x[lambda_L+(i-1)*(N-1)+j]
		end
		norm=0
		for j in 1:N
			norm+=Us[i,j]^2
		end
		norm=norm^(0.5)
		Us[i,:]=[j/norm for j in Us[i,:]]
	end
	
	return Us
end

function lambda_for_THC(N, a, x)
	idx=0
	lambda=zeros(a,a)
	for i in 1:a
		for j in i:a
			idx+=1
			lambda[i,j]=x[idx]
			lambda[j,i]=x[idx]
		end
	end
	return lambda
end

function theta_for_THC(N, a, x, lambda_L)
	theta=zeros(a, N-1)
	
	for i in 1:a
		for j in 1:N-1
			theta[i,j]=x[lambda_L+(i-1)*(N-1)+j]
		end
	end
	
	return theta
	
end


function THC_tb_x_to_F_OP(x, N, a,lambda_L,spin,test=false)
	g=zeros(N,N,N,N)
	h=zeros(N,N)
	
	#lambda=zeros(a,a)
	#=Us=zeros(a,N)
	
	for i in 1:a
		Us[i,1]=1
		for j in 1:N-1
			Us[i,j+1]=x[lambda_L+(i-1)*(N-1)+j]
		end
		norm=0
		for j in 1:N
			norm+=Us[i,j]^2
		end
		norm=norm^(0.5)
		Us[i,:]=[j/norm for j in Us[i,:]]
	end=#
	
	Us=Us_for_THC(N, a, x, lambda_L)
	lambda=lambda_for_THC(N, a, x)
	
	#=if test == true
		for i in 1:a
			norm=0
			for j in 1:N
				norm+=Us[i,j]^2
			end
			norm=norm^(0.5)
			@show norm
		end
		
		
	end
	
	idx=0
	for i in 1:a
		for j in i:a
			idx+=1
			lambda[i,j]=x[idx]
			lambda[j,i]=x[idx]
		end
	end
	
	id=zeros(N,N)
	for i in 1:N
		id[i,i]=1
	end=#
	
	
	if test==true
		Unitaries=zeros(a,N,N)
		for i in 1:a
			for p in 1:N
				for q in 1:N
				
					Unitaries[i,p,q]=2*Us[i,p]*Us[i,q]-id[p,q]
				end
			end
			@show check_unitary(Unitaries[i,:,:],N)
		end
	end
	
	if test==false
		for p in 1:N
			for q in 1:N
				for r in 1:N
					for s in 1:N
						for m in 1:a
							for n in 1:a
								g[p,q,r,s]+=lambda[m,n]*(Us[m,p]*Us[m,q])*(Us[n,r]*Us[n,s])
							end
						end
					end
				end
			end
		end
	else
		for p in 1:N
			for q in 1:N
				#for r in 1:N
					#for s in 1:N
						#g[p,q,r,s]=(2*Us[1,p]*Us[1,q]-id[p,q])*(2*Us[2,r]*Us[2,s]-id[r,s])+(2*Us[2,p]*Us[2,q]-id[p,q])*(2*Us[1,r]*Us[1,s]-id[r,s])+(2*Us[1,p]*Us[1,q]-id[p,q])*(2*Us[1,r]*Us[1,s]-id[r,s])
						#g[p,q,r,s]=(2*Us[1,p]*Us[1,q]-id[p,q])*(2*Us[2,r]*Us[2,s]-id[r,s])+(2*Us[2,p]*Us[2,q]-id[p,q])*(2*Us[1,r]*Us[1,s]-id[r,s])
						#g[p,q,r,s]=(2*Us[1,p]*Us[1,q]-id[p,q])*(2*Us[1,r]*Us[1,s]-id[r,s])
						h[p,q]=2*Us[1,p]*Us[1,q]-id[p,q]
					end
				end
			end
		#end
	#end
	
	return F_OP(([0.0], h, g),spin)
	#return F_OP(([0.0],zeros(N,N),g),false)
end
					


	

#Function to implement LSQFit based THC Decomposition

function THC_grad(F::F_OP, step_size=4,tol=1e-4, initial=false,iter=1)
	a=step_size
	lambda_L=Int64(a*(a+1)/2)
	unitary_L=a*(F.N-1)
	
	if initial==false
		x0=zeros(lambda_L+unitary_L)
		x0.=rand(0:1,length(x0))
	else
		x0=initial
	end
	
	function cost(x)
		Fx=THC_tb_x_to_F_OP(x, F.N, step_size,lambda_L)
		return L2_TB_cost(F,Fx)
	end
	sol=optimize(cost, x0, BFGS(), Optim.Options(show_trace=true, extended_trace=false))
	F_sol=THC_tb_x_to_F_OP(sol.minimizer, F.N, step_size,lambda_L)
	
	if sol.minimum>tol && iter<5
		iter+=1
		a=2a
		lambda_L_new=Int64(a*(a+1)/2)
		unitary_L_new=a*(F.N-1)
		
		x1=zeros(lambda_L_new+unitary_L_new)
		x1[1:lambda_L]=sol.minimizer[1:lambda_L]
		x1[lambda_L_new+1:lambda_L_new+unitary_L]=sol.minimizer[lambda_L+1:end]
		THC_LSQ(F-F_sol, a, tol, x1,iter)
	end
			
end
		
		
function THC_fixed_uni_step(F::F_OP, step_size=4)
	a=step_size
	lambda_L=Int64(a*(a+1)/2)
	unitary_L=a*(F.N-1)
	x0=zeros(lambda_L)
	x0.=rand(0:1,length(x0))
	x1=zeros(lambda_L+unitary_L)
	x1[1:lambda_L]=x0
	x1[lambda_L+1:end].=rand(0:1,unitary_L)
	
	function cost(x)
		x1[1:lambda_L]=x
		Fx=THC_tb_x_to_F_OP(x1, F.N, step_size,lambda_L)
		return L2_TB_cost(F,Fx)
	end
	sol=optimize(cost, x0, BFGS(), Optim.Options(show_trace=true, extended_trace=false))
	#F_sol=THC_x_to_F_OP(sol.minimizer, F.N, step_size,lambda_L)
end

function THC_fixed_uni_step_lsq(F::F_OP, step_size=4)
	a=step_size
	N=F.N
	lambda_L=Int64(a*(a+1)/2)
	unitary_L=a*(F.N-1)
	x0=zeros(lambda_L)
	x0.=rand(0:1,length(x0))
	x1=zeros(lambda_L+unitary_L)
	x1[1:lambda_L]=x0
	x1[lambda_L+1:end].=rand(0:1,unitary_L)
	
	function cost(x)
		x1[1:lambda_L]=x
		Fx=THC_tb_x_to_F_OP(x1, F.N, step_size,lambda_L)
		output=zeros(F.N,F.N,F.N,F.N)
		idx=0
		for p in 1:N
			for q in 1:N
				for r in 1:N
					for s in 1:N
						idx+=1
						output[idx]=F.mbts[3][p,q,r,s]-Fx.mbts[3][p,q,r,s]
					end
				end
			end
		end
		return output
					
		#=output=zeros(lambda_L)
		output[1]=L2_TB_cost(F,Fx)
		return output=#
		#return [L2_TB_cost(F,Fx)]
	end
	sol=LeastSquaresOptim.optimize(cost, x0, LevenbergMarquardt(LeastSquaresOptim.LSMR()))
	@show sol
	#F_sol=THC_x_to_F_OP(sol.minimizer, F.N, step_size,lambda_L)
end

function thc_cost_vec(F::F_OP, x, p, N, a, lambda_L)
	Fx=THC_tb_x_to_F_OP(x, N, a,lambda_L,F.spin_orb)  
	output=zeros(F.N^4+a^2)
	idx=0
	for p in 1:N
		for q in 1:N
			for r in 1:N
				for s in 1:N
					idx+=1
					output[idx]=F.mbts[3][p,q,r,s]-Fx.mbts[3][p,q,r,s]
				end
			end
		end
	end
	
	lambda=lambda_for_THC(N,a,x)
	for i in 1:a
		for j in 1:a
			idx+=1
			output[idx]=p*lambda[i,j]
		end
	end
			
	return output
end

function THC_tb_lsq(F::F_OP, step_size=4, tol=1e-6, iter=1, iter_max=20,p=0.2)
	a=step_size
	N=F.N
	lambda_L=Int64(a*(a+1)/2)
	unitary_L=a*(F.N-1)
	count=0
	
	
	
	x0=ones(lambda_L+unitary_L)
	#x0.=rand(0:1,length(x0))
	#x0[lambda_L+1:end].=rand(0:1,length(unitary_L))
	#x0[1:lambda_L].=[i/10^(4) for i in x0[1:lambda_L]]
	#x0[1:lambda_L]=ones(lambda_L)
	
	
	obt=zeros(N,N)
	F=F_OP(([0],obt,F.mbts[3]),false)
	
	if iter==1
		
		@show L1_TB_cost(F)
		
	elseif iter==2
		#p=0.1
	else
		#p=0.1
	end
	
	residue_norm=L2_TB_cost(F)
	if residue_norm<tol^2
		p=0.1
	end
	
	if iter==1
		s=1
	else
		s=1
	end
	
	
	function cost_f!(output, x)
		
		Fx=THC_tb_x_to_F_OP(x, F.N, step_size,lambda_L,F.spin_orb)
		#output=zeros(F.N^4+a^2)
		idx=0
		for p in 1:N
			for q in 1:N
				for r in 1:N
					for s in 1:N
						idx+=1
						output[idx]=F.mbts[3][p,q,r,s]-Fx.mbts[3][p,q,r,s]
					end
				end
			end
		end
		x_lambda=x[1:lambda_L]
		lambda=zeros(a,a)
		id=0
		for i in 1:a
			for j in i:a
				id+=1
				lambda[i,j]=x_lambda[id]
				lambda[j,i]=x_lambda[id]
			end
		end
		for i in 1:a
			for j in 1:a
				idx+=1
				output[idx]=p*lambda[i,j]
			end
		end
				
		#return output
	end
	
	function cost_g!(J, x)
		J.=gradient_thc(N, a, x, lambda_L,p)
		#J.=numeric_gradient_thc(F,N,a,x, lambda_L, p)
	end
	
	
	function rosenbrock_f!(out, x)
	 out[1] = 1 - x[1]
	 out[2] = 100 * (x[2]-x[1]^2)
	end
	
	function rosenbrock_g!(J, x)
	    J[1, 1] = -1
	    J[1, 2] = 0
	    J[2, 1] = -200 * x[1]
	    J[2, 2] = 100
	end
	
	
	#sol=LeastSquaresOptim.optimize!(LeastSquaresProblem(x = x0, f! = cost_f!, output_length=N^4+a^2, autodiff=:central), LevenbergMarquardt(LeastSquaresOptim.Cholesky()), show_trace=true, show_every=1)
	
	
	print("Analytic Gradients:\n\n\n")
	x0=ones(lambda_L+unitary_L)
	sol=LeastSquaresOptim.optimize!(LeastSquaresProblem(x = x0, f! = cost_f!, g! = cost_g!, output_length=N^4+a^2), LevenbergMarquardt(LeastSquaresOptim.Cholesky()), show_trace=false, show_every=1)
	
	
	
	#sol=LeastSquaresOptim.optimize!(LeastSquaresProblem(x = x0, f! = cost_f!, g!=cost_g!, output_length = N^4+a^2), LevenbergMarquardt(LeastSquaresOptim.Cholesky()))
	
	
	#sol=optimize(cost, x0, LevenbergMarquardt(LeastSquaresOptim.Cholesky()))
	
	#optimize!(LeastSquaresProblem(x=x0, f!=cost_f!, output_length=N^4+a^2, autodiff=:central),  LevenbergMarquardt(LeastSquaresOptim.Cholesky()))
	
	print(sol)
	x=sol.minimizer
	
	
	x_lambda=x[1:lambda_L]
	lambda=zeros(a,a)
	idx=0
	for i in 1:a
		for j in i:a
			idx+=1
			lambda[i,j]=x_lambda[idx]
			lambda[j,i]=x_lambda[idx]
		end
	end
	
	norm=0
	for i in 1:a
		for j in 1:a
			norm+=abs(lambda[i,j])
		end
	end
	
	if F.spin_orb==false
		norm=4*norm
	end
	
	@show norm
	#tbt=zeros(N,N,N,N)
	#F1=F_OP(([0],F.mbts[2],tbt))
	#@show PAULI_L1(F1)
	
	Fx=THC_tb_x_to_F_OP(x, F.N, step_size,lambda_L,F.spin_orb,false)
	#return Fx
	F_res=F-Fx
	@show L1_TB_cost(F_res)
	minimum=L2_TB_cost(F,Fx)
	if minimum>tol && iter<iter_max
		iter+=1		
		F_iter, norm_recursive, count=THC_tb_lsq(F_res, a, tol, iter, iter_max)
		return Fx+F_iter, norm+norm_recursive, count+1
	else
		return Fx, norm, 1
	end
	
	
end
