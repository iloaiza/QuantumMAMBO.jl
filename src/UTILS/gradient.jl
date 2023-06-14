#Gradients for Optimization

#PART-A: Gradients for two-body terms

function del_wr_lr(n,l,m,U)
	wr_lr=zeros(n,n,n,n)
	Ul = U[:,l]
	Um = U[:,m]
	@einsum wr_lr[p,q,r,s]=Ul[p]*Ul[q]*Um[r]*Um[s]
	return wr_lr
end

function grad_lr_ij(x,i,j,diff)
	n=size(diff,1)
	u_params=x[cartan_2b_num_params(n)+1:end]
    	U = one_body_unitary(real_orbital_rotation(n, u_params))
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

function del_w_u(n,x)
	#Computing del W_pqrs / del U_ab
    wu= zeros(n, n, n, n, n, n)
    u_params=x[cartan_2b_num_params(n)+1:end]
    U = one_body_unitary(real_orbital_rotation(n, u_params))
    lmbda_matrix = get_cartan_matrix(n,x)
    delta = ones(n)
    @einsum wu[p,q,r,s,p,b] += delta[p] * U[q,b] * U[r,m] * U[s,m] * lmbda_matrix[b,m]
    @einsum wu[p,q,r,s,q,b] += delta[q] * U[p,b] * U[r,m] * U[s,m] * lmbda_matrix[b,m]
    @einsum wu[p,q,r,s,r,b] += delta[r] * U[p,l] * U[q,l] * U[s,b] * lmbda_matrix[l,b]
    @einsum wu[p,q,r,s,s,b] += delta[s] * U[p,l] * U[q,l] * U[r,b] * lmbda_matrix[l,b]

    return wu
end #get w_o

function del_u_theta(n,u_params,i)  #changes have to be finalized after a successful end to the debugging process
	# returns the gradient w.r.t i'th angles
	kappa = get_generator(n,i)
    	K = construct_anti_symmetric(n, u_params)
    	#U = one_body_unitary(real_orbital_rotation(n, u_params))
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
    	#O_herm_trans=O'
    	#@einsum u_theta[i,j]:=O[i,k]*I[k,l]*expD[l,l]*O_herm_trans[l,j]
    	#return u_theta=#
    	
    	#return kappa*U	
    
    	return real.(O * I * expD * O')
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

function grad_l_k(x,k,diff_ob)
	n=size(diff_ob,1)
	u_params=x[cartan_SD_num_params(n)+1:end]
    	U = one_body_unitary(real_orbital_rotation(n, u_params))
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

function grad_l_tb_ij(x,i,j,diff_tb)
	n=size(diff_tb,1)
	u_params=x[cartan_SD_num_params(n)+1:end]
    	U = one_body_unitary(real_orbital_rotation(n, u_params))
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

function del_h_u(n,x)
	hu=zeros(n,n,n,n)
	cartan_ob_coeffs=x[1:n]
	u_params=x[cartan_SD_num_params(n)+1:end]
	U=one_body_unitary(real_orbital_rotation(n,u_params))
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

function del_w_u_SD(n,x)
	wu= zeros(n, n, n, n, n, n)
	u_params=x[cartan_SD_num_params(n)+1:end]
	U = one_body_unitary(real_orbital_rotation(n, u_params))
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

	

	
	

	
	
	
