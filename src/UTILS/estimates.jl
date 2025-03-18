const coeff_accuracy = 1e-5
const rot_accuracy = 1e-10
const givens_beta = 16

function quantum_estimate(H :: F_OP, method :: String; kwargs...)
	#return T-count and number of logical qubits estimates for different LCUs
	if method == "sparse"
		T,Q,R = sparse_estimate(H; kwargs...)
	elseif method == "AC"
		T,Q,R = AC_estimate(H; kwargs...)
	elseif method == "DF"
		T,Q,R = DF_estimate(H; kwargs...)
	elseif method == "MPS"
		T,Q,R = MPS_estimate(H; kwargs...)
	elseif method == "CP4T"
		T,Q,R = CP4T_estimate(H; kwargs...)
	elseif method == "CP4Q"
		T,Q,R = CP4Q_estimate(H; kwargs...)
	end
	@show T,Q,R


	Tr, _ = R * rotation_estimate(rot_accuracy)

	T += Tr
	if rot_accuracy < 0.016
		Q += 1
	end

	return Int(ceil(T)), Int(Q)
end


function get_2_factors(N)
	#find largest k such that N = L*2^k
	#ceil(log2(N)) = k + ceil(log2(L))
	k = Int(floor(log2(N)))
	L = N/(2^k)

	return ceil(log2(L)),k
end

function sparsify!(X, trunc_eps)
	#erase all values beneath threshold
	for (i,x) in enumerate(X)
		if abs(x) < trunc_eps
			X[i] = 0
		end
	end
end

function rotation_estimate(eps)
	#cost of implementing controlled Rz rotation cRz
	#see [1] https://www.nature.com/articles/s41534-022-00651-y for optimal implementation of cRz
	#more efficient repeat-until-success implementation is given in 
	#[2] https://arxiv.org/pdf/1404.5320, where the cost of controlled Rz is of two Rz's (see Fig.1.b of Ref.[1])
	if eps < 0.016
		T_count = minimum([-3.067*log2(eps) + 9.678, 2*(-1.149*log2(eps) + 9.2)])
		Q_count = 1
	else
		T_count = -6.134*log2(eps) - 8.644
		Q_count = 0
	end

	return [T_count, Q_count]
end

function prep_estimate(K, control = false)
	μ = maximum([0, Int(ceil(-log2(coeff_accuracy * K)))])
	@show μ
	l,k = get_2_factors(K)

	T_count = 8l + 4*(K-1) + 8μ - 4 + 7*ceil(log2(K))
	Q_count = 2*ceil(log2(K)) + 2μ + 3 + maximum([2μ-1, K-1, k+2l])

	if control
		T_count += 8 + 2k + 2l
		Q_count += 2
	end

	return [T_count, Q_count, 2]
end

function sparse_estimate(H :: F_OP; trunc_thresh = 1e-5)
	#return resource estimates for sparse LCU
	obt = H.mbts[2]
	tbt = H.mbts[3]

	ζ1 = zeros(H.N,H.N)
	ζ2 = zeros(H.N,H.N,H.N,H.N)

	for i in 1:H.N
		for j in 1:H.N
			if i < j
				ζ1[i,j] = sqrt(2)
			elseif i == j
				ζ1[i,j] = 1
			end
		end
	end

	for i in 1:H.N
		for j in 1:H.N
			for k in 1:H.N
				for l in 1:H.N
					if i<k || (i==k && j<l)
						ζ2[i,j,k,l] = sqrt(2)
					elseif i==k && j==l
						ζ2[i,j,k,l] = 1
					end
				end
			end
		end
	end

	g_trunc = zeros(H.N,H.N,H.N,H.N)
	for i in 1:H.N
		for j in 1:H.N
			for k in 1:H.N
				for l in 1:H.N
					if abs(tbt[i,j,k,l]) > trunc_thresh
						g_trunc[i,j,k,l] = ζ1[i,j] * ζ1[k,l] * ζ2[i,j,k,l] * tbt[i,j,k,l]
					end
				end
			end
		end
	end

	S = length(g_trunc[abs.(g_trunc) .> 0]) + H.N^2
	@show S
	ks,s = get_2_factors(S)
	μ = maximum([0, Int(ceil(-log2(coeff_accuracy * S)))])
	@show μ

	T_count = 8*ceil(log2(s)) + 4*(S-1) + 8μ + 3 + 7*8*ceil(log2(H.N))

	Q_count = ceil(log2(S)) + 8*ceil(log2(H.N)) + 2μ + 8 + maximum([ks+2s, ceil(log2(S)) - 1, 2μ-1])
	R_count = 2

	prep2_cost = [2*T_count, Q_count, 2*R_count]

	T_sel = 16*(H.N + 1)
	Q_sel = 6 + 2*H.N + 4*ceil(log2(H.N)) + ceil(log2(H.N+1))

	sel_cost = [T_sel, Q_sel, 0]

	return prep2_cost + sel_cost
end


function AC_estimate(H :: F_OP; trunc_thresh = 1e-5)
	sparsify!(H.mbts[3], trunc_thresh)

	AC_coeffs, AC_ops = AC_group(H, ret_ops = true)
	G = length(AC_ops)

	Gns = zeros(Int64, G)
	for i in 1:G
		Gns[i] = length(AC_ops[i].paulis)
	end
	S = sum(Gns)

	@show S, G
	T_prep, Q_prep, R_prep = prep_estimate(G)
	prep2_cost = [2*T_prep, Q_prep, 2*R_prep]

	T_sel = 4*(G-1)
	Q_sel = ceil(log2(G)) - 1
	R_sel = 2*(S-G)

	sel_cost = [T_sel, Q_sel, R_sel]

	return prep2_cost + sel_cost
end

function CP4T_estimate(H :: F_OP; trunc_thresh = 1e-5, rank_max = 1000, rank_hop = 10, ini_rank = 2)
	sparsify!(H.mbts[3], trunc_thresh)

	CP4_groups = CP4_decomposition(H, rank_max, rank_hop=rank_hop, ini_rank = ini_rank)

	W = length(CP4_groups)

	T_prep, Q_prep, R_prep = prep_estimate(W)
	prep2_cost = [2*T_prep, Q_prep, 2*R_prep]

	T_sel = 4*7*H.N + 8*(W-1) + 16*7*(givens_beta - 2) + 4
	Q_sel = 3 + 2*ceil(log2(W)) + 2*H.N + 4*givens_beta
	R_sel = 0

	sel_cost = [T_sel, Q_sel, R_sel]

	return prep2_cost + sel_cost
end