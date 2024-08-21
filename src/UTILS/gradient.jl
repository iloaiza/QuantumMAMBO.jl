#Gradients for Optimization

#PART-A: Gradients for two-body terms

function del_wr_lr(n,l,m,U)
	wr_lr=zeros(n,n,n,n)
	Ul = U[:,l]
	Um = U[:,m]
	@einsum wr_lr[p,q,r,s]=Ul[p]*Ul[q]*Um[r]*Um[s]
	return wr_lr
end

function grad_lr_ij(x,i,j,diff,do_givens=CSA_GIVENS)
	n=size(diff,1)
	u_params=x[cartan_2b_num_params(n)+1:end]
	if do_givens
		U = one_body_unitary(givens_real_orbital_rotation(n, u_params))
    	else
    		U = one_body_unitary(real_orbital_rotation(n, u_params))
    	end
	cw=2*diff
	wr_lr=del_wr_lr(n,i,j,U)
	cl=sum(cw.*wr_lr)
	if i!=j
		cl*=2
	end
	return cl
end

function grad_lr(n,x,diff)
	cartan_grad=zeros(cartan_2b_num_params(n))
	idx=1
	for i=1:n
		for j=1:i
			cartan_grad[idx]=grad_lr_ij(x,i,j,diff)
			idx+=1
		end
	end
	
	return cartan_grad
end	

function get_cartan_matrix(n,x)
	l=zeros(n,n)
	coeffs=x[1:cartan_2b_num_params(n)]
	idx=1
	for i=1:n
		for j=1:i
			l[i,j]=coeffs[idx]
			l[j,i]=l[i,j]
			idx+=1
		end
	end
	return l
end	

function del_w_u(n,x,do_givens=CSA_GIVENS)
	#Computing del W_pqrs / del U_ab
    wu= zeros(n, n, n, n, n, n)
    u_params=x[cartan_2b_num_params(n)+1:end]
    if do_givens
		U = one_body_unitary(givens_real_orbital_rotation(n, u_params))
    else
    		U = one_body_unitary(real_orbital_rotation(n, u_params))
    end
    lmbda_matrix = get_cartan_matrix(n,x)
    delta = ones(n)
    @einsum wu[p,q,r,s,p,b] += delta[p] * U[q,b] * U[r,m] * U[s,m] * lmbda_matrix[b,m]
    @einsum wu[p,q,r,s,q,b] += delta[q] * U[p,b] * U[r,m] * U[s,m] * lmbda_matrix[b,m]
    @einsum wu[p,q,r,s,r,b] += delta[r] * U[p,l] * U[q,l] * U[s,b] * lmbda_matrix[l,b]
    @einsum wu[p,q,r,s,s,b] += delta[s] * U[p,l] * U[q,l] * U[r,b] * lmbda_matrix[l,b]

    return wu
end #get w_o

function del_u_theta(n,u_params,idx,do_givens=CSA_GIVENS)  
	# returns the gradient w.r.t i'th angles
	
	if do_givens
		i,j=get_rot_indices(n,idx)
		d_ug=collect(Diagonal(zeros(n)))
		d_ug[i,i] = -sin(u_params[idx])
		d_ug[j,j] = d_ug[i,i]
		d_ug[i,j] = cos(u_params[idx])
		d_ug[j,i] = -d_ug[i,j]
		
		
		
		dUrot = collect(Diagonal(ones(n)))
		for k=1:idx-1
			i,j=get_rot_indices(n,k)
			Ug = collect(Diagonal(ones(n)))
			Ug[i,i] = cos(u_params[k])
			Ug[j,j] = Ug[i,i]
			Ug[i,j] = sin(u_params[k])
			Ug[j,i] = -Ug[i,j]
			dUrot = Ug * dUrot
		end
		
		dUrot=d_ug*dUrot
		
		for k=idx+1:real_orbital_rotation_num_params(n)
			i,j=get_rot_indices(n,k)
			Ug = collect(Diagonal(ones(n)))
			Ug[i,i] = cos(u_params[k])
			Ug[j,j] = Ug[i,i]
			Ug[i,j] = sin(u_params[k])
			Ug[j,i] = -Ug[i,j]
			dUrot = Ug * dUrot
		end
		return dUrot	
	else
		kappa = get_generator(n,idx)
	    	K = construct_anti_symmetric(n, u_params)
	    	D, O = eigen(K)
		I = O' * kappa * O
	    	for a in 1:n
			for b in 1:n
		    		if abs(D[a] - D[b]) > 1e-8
		        		I[a, b] *= (exp(D[a] - D[b]) - 1) / (D[a] - D[b])
		    		end
			end
	    	end

	    	expD = Diagonal(exp.(D))
	    	return real.(O * I * expD * O')
	end
		
end #get_o_angles

function grad_theta(n,x,diff)
    opnum = real_orbital_rotation_num_params(n)

    wu = del_w_u(n,x)
    u_params=x[cartan_2b_num_params(n)+1:end]

    u_grad = zeros(opnum)
    w_theta = zeros(n,n,n,n)
    for i in 1:opnum
        u_theta = del_u_theta(n,u_params,i)
        @einsum w_theta[p,q,r,s] = wu[p,q,r,s,a,b] * u_theta[a,b]
        u_grad[i] = 2 * sum(diff .* w_theta)
    end
    
    return u_grad
end

function gradient(n,x,diff)
	
	return vcat(grad_lr(n,x,diff),grad_theta(n,x,diff))
end

#PART-B: Gradient for CSA_SD 



function del_h_l(n,k,U)
	h_l=zeros(n,n)
	Uk=U[:,k]
	@einsum h_l[r,s]=Uk[r]*Uk[s]
	
	#@show h_l
	return h_l
end

function grad_l_k(x,k,diff_ob,do_givens=CSA_GIVENS)
	n=size(diff_ob,1)
	u_params=x[cartan_SD_num_params(n)+1:end]
    	if do_givens
		U = one_body_unitary(givens_real_orbital_rotation(n, u_params))
    	else
    		U = one_body_unitary(real_orbital_rotation(n, u_params))
    	end
	ch=2*diff_ob
	h_l=del_h_l(n,k,U)
	cl=sum(ch.*h_l)
	return cl
end

function grad_l_ob(n,x,diff_ob)
	cartan_ob_grad=zeros(n)
	for k=1:n
		cartan_ob_grad[k]=grad_l_k(x,k,diff_ob)
	end
	return cartan_ob_grad
end

function grad_l_tb_ij(x,i,j,diff_tb,do_givens=CSA_GIVENS)
	n=size(diff_tb,1)
	u_params=x[cartan_SD_num_params(n)+1:end]
    	if do_givens
		U = one_body_unitary(givens_real_orbital_rotation(n, u_params))
    	else
    		U = one_body_unitary(real_orbital_rotation(n, u_params))
    	end
	cw=2*diff_tb
	wr_lr=del_wr_lr(n,i,j,U)
	cl=sum(cw.*wr_lr)
	if i!=j
		cl*=2
	end
	return cl
end

function grad_l_tb(n,x,diff_tb)
	cartan_grad=zeros(cartan_2b_num_params(n))
	idx=1
	for i=1:n
		for j=1:i
			cartan_grad[idx]=grad_l_tb_ij(x,i,j,diff_tb)
			idx+=1
		end
	end
	
	return cartan_grad
end	

function del_h_u(n,x,do_givens=CSA_GIVENS)
	hu=zeros(n,n,n,n)
	cartan_ob_coeffs=x[1:n]
	u_params=x[cartan_SD_num_params(n)+1:end]
	if do_givens
		U = one_body_unitary(givens_real_orbital_rotation(n, u_params))
    	else
    		U = one_body_unitary(real_orbital_rotation(n, u_params))
    	end
	delta=ones(n)
	@einsum hu[r,s,r,j]+=delta[r]*U[s,j]*cartan_ob_coeffs[j]
	@einsum hu[r,s,s,j]+=delta[s]*U[r,j]*cartan_ob_coeffs[j]
	return hu
end

function get_cartan_matrix_SD(n,x)
	l=zeros(n,n)
	coeffs=x[n+1:cartan_SD_num_params(n)]
	idx=1
	for i=1:n
		for j=1:i
			l[i,j]=coeffs[idx]
			l[j,i]=l[i,j]
			idx+=1
		end
	end
	return l
end	

function del_w_u_SD(n,x,do_givens=CSA_GIVENS)
	wu= zeros(n, n, n, n, n, n)
	u_params=x[cartan_SD_num_params(n)+1:end]
	if do_givens
		U = one_body_unitary(givens_real_orbital_rotation(n, u_params))
    	else
    		U = one_body_unitary(real_orbital_rotation(n, u_params))
    	end
        lmbda_matrix = get_cartan_matrix_SD(n,x)
        delta = ones(n)
        @einsum wu[p,q,r,s,p,b] += delta[p] * U[q,b] * U[r,m] * U[s,m] * lmbda_matrix[b,m]
        @einsum wu[p,q,r,s,q,b] += delta[q] * U[p,b] * U[r,m] * U[s,m] * lmbda_matrix[b,m]
        @einsum wu[p,q,r,s,r,b] += delta[r] * U[p,l] * U[q,l] * U[s,b] * lmbda_matrix[l,b]
        @einsum wu[p,q,r,s,s,b] += delta[s] * U[p,l] * U[q,l] * U[r,b] * lmbda_matrix[l,b]

        return wu
end
	

function grad_SD_theta(n,x,diff_ob,diff_tb)
    opnum = real_orbital_rotation_num_params(n)
    hu=del_h_u(n,x)

    wu = del_w_u_SD(n,x)
    u_params=x[cartan_SD_num_params(n)+1:end]

    u_grad = zeros(opnum)
    h_theta=zeros(n,n)
    w_theta = zeros(n,n,n,n)
    for i in 1:opnum
        u_theta = del_u_theta(n,u_params,i)
        @einsum h_theta[p,q]=hu[p,q,a,b]*u_theta[a,b]
        @einsum w_theta[p,q,r,s] = wu[p,q,r,s,a,b] * u_theta[a,b]
        u_grad[i] = 2 * (sum(diff_tb .* w_theta)+sum(diff_ob.*h_theta))
    end
    
    return u_grad
end


function gradient_csa_sd(n,x,diff_ob,diff_tb)
	
	return vcat(vcat(grad_l_ob(n,x,diff_ob),grad_l_tb(n,x,diff_tb)),grad_SD_theta(n,x,diff_ob,diff_tb))
end

	
#Gradients for THC


function del_J_tbt_lambda(lambda, Us, p, q, r, s, m, n)
	
	derivative = Us[m,p]*Us[m,q]*Us[n,r]*Us[n,s]
	
	if m!=n
		derivative+=Us[n,p]*Us[n,q]*Us[m,r]*Us[m,s]
	end
	
	return derivative
end

function del_J_punish_lambda(lambda, Us, μ, ν, m, n, p)
	if m==n && μ==m && ν==n
		return p
	
	elseif (μ==m && ν==n) || (μ==n && ν==m)
		return p
	else
		return 0
	end
end

function del_term(lambda, Us, theta, p,q,r,s, m, k, N, a)
	#term=zeros(N,N,N,N, a, N-1)
	term=0
	id=zeros(N, N)
	for i in 1:N-1
		id[i,i]=1
	end
	
	#@einsum term[p,q,r,s, m, k] += lambda[m,n]*(id[p,k]*Us[m,1]-theta[m,p]*theta[m,k]*Us[m,1]^3)*Us[m,q]*Us[n,r]*Us[n,s]
	
	#for p in 1:N
		#for q in 1:N
			#for r in 1:N
				#for s in 1:N
					#for m in 1:a
						#for k in 1:N-1
							for n in 1:a
								if p!=1
									term += lambda[m,n]*(id[p-1,k]*Us[m,1]-theta[m,p-1]*theta[m,k]*Us[m,1]^3)*Us[m,q]*Us[n,r]*Us[n,s]
								else
									term+= lambda[m,n]*(-theta[m,k]*Us[m,1]^3)*Us[m,q]*Us[n,r]*Us[n,s]
								end
							end
						#end
					#end
				#end
			#end
		#end
	#end
	
	return term
end
	
function del_J_tbt_U(lambda, Us, theta, p, q, r, s, m, k, N, a)
	return del_term(lambda, Us, theta, p,q,r,s, m, k, N, a) + del_term(lambda, Us, theta, q,p, r, s, m, k, N, a) + del_term( lambda, Us, theta, r, s, p, q, m, k, N, a) + del_term(lambda, Us, theta, s, r, p, q, m, k, N, a)
end

function gradient_thc(N, a, x, lambda_L, p)
	lambda=lambda_for_THC(N, a, x)
	Us=Us_for_THC(N, a, x, lambda_L)
	theta=theta_for_THC(N,a,x, lambda_L)
	J=zeros(N^4+a^2, Int64(a*(a+1)/2+a*(N-1)))
	
	#Derivative wrt lambda
	
	idx_mn=0
	for m in 1:a
		for n in m:a
			idx_mn+=1
			
			idx_tbt=0
			for p in 1:N
				for q in 1:N
					for r in 1:N
						for s in 1:N
							idx_tbt+=1
							
							J[idx_tbt, idx_mn]=-del_J_tbt_lambda(lambda, Us, p, q, r, s, m, n)
						end
					end
				end
			end
			
			idx_punish=N^4
			for μ in 1:a
				for ν in 1:a
					idx_punish+=1
					
					J[idx_punish, idx_mn]=del_J_punish_lambda(lambda, Us, μ, ν, m, n, p)
				end
			end
			
		end
	end
	
	idx_U=Int64(a*(a+1)/2)
	
	for m in 1:a
		for k in 1:N-1
			idx_U+=1
			
			idx_tbt=0
			for p in 1:N
				for q in 1:N
					for r in 1:N
						for s in 1:N
							idx_tbt+=1
							
							 J[idx_tbt, idx_U]=-del_J_tbt_U(lambda, Us, theta, p, q, r, s, m, k, N, a)
						end
					end
				end
			end
			
		end
	end
	#@show J
	#exit()
	return J
end

		
#Numerical gradient for THC

function numeric_gradient_thc(F::F_OP,N,a,x, lambda_L, p)
	dx=0.001
	J=zeros(N^4+a^2, length(x))
	#cost_vec=thc_cost_vec(F, x, p, N, a, lambda_L)	
	x1=zeros(length(x))
	x2=zeros(length(x))
	for i in 1:length(x)
		x1.=x
		x1[i]=x[i]+dx
		x2.=x
		x2[i]=x[i]-dx
		J[:,i].=(thc_cost_vec(F, x1, p, N, a, lambda_L)-thc_cost_vec(F, x2, p, N, a, lambda_L))/(2dx)
	end
	@show J
	exit()
	return J
end
		
							
			
	
	
	

	
	
	

	
	
	
