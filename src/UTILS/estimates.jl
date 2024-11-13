const max_givens = 14

function quantum_estimate(H :: F_OP, method :: String; ini_rank=250, rank_hop=25, kwargs...)
	#return T-count and number of logical qubits estimates for different LCUs
	if method == "sparse"
		T,Q,R,λ = sparse_estimate(H; kwargs...)
	elseif method == "AC"
		T,Q,R,λ = AC_estimate(H; kwargs...)
	elseif method == "DF_openfermion"
		T,Q,R,λ = DF_openfermion_estimate(H; kwargs...)
	elseif method == "DF"
		T,Q,R,λ = DF_estimate(H; kwargs...)
	elseif method == "MPS"
		T,Q,R,λ = MPS_estimate(H; kwargs...)
	elseif method == "SYM4"
		T,Q,R,λ = SYM4_estimate(H; kwargs...)
	elseif method == "CP4"
		CT_t,CT_q,CT_r,CQ_t,CQ_q,CQ_r,λ = CP4_estimates(H; ini_rank=ini_rank, rank_hop=rank_hop, kwargs...)
		T_tot = total_estimate(CT_t,CT_q,CT_r)
		Q_tot = total_estimate(CQ_t,CQ_q,CQ_r)

		return T_tot, Q_tot, λ
	elseif method == "SVD"
		CT_t,CT_q,CT_r,CQ_t,CQ_q,CQ_r,λ = SVD_estimates(H; kwargs...)
		T_tot = total_estimate(CT_t,CT_q,CT_r)
		Q_tot = total_estimate(CQ_t,CQ_q,CQ_r)

		return T_tot, Q_tot, λ
	end

	return total_estimate(T,Q,R), λ
end

function total_estimate(T,Q,R)
	@show T,Q,R
	rot_resources = rotation_estimate(rot_accuracy)
	Ttot = T + R*rot_resources[1]
	Qeff = Q + rot_resources[2]

	return Int(ceil(Ttot)), Int(Qeff)
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
	#see https://www.nature.com/articles/s41534-022-00651-y for optimal implementation of cRz
	if eps < 0.016
		T_count = -3.067*log2(eps) + 9.678
		Q_count = 1
	else
		T_count = -6.134*log2(eps) - 8.644
		Q_count = 0
	end

	return [T_count, Q_count]
end

function prep_estimate(K, control = false)
	μ = maximum([0, Int(ceil(-log2(coeff_accuracy * K)))])
	#@show μ
	l,k = get_2_factors(K)

	T_count = 8l + 4K + 8μ - 4 + 7*ceil(log2(K)) - 8
	Q_count = ceil(log2(K)) + 2μ + 3
	Rq_count = maximum([2μ-1, ceil(log2(K))-1, l])

	if control
		T_count += 4 + 2k + 2l
		Q_count += 1
		Rq_count += 1
	end

	return [T_count, Q_count, Rq_count, 2]
end

function sparse_estimate(H :: F_OP; trunc_thresh = 1e-5)
	#return resource estimates for sparse LCU
	obt = H.mbts[2] + ob_correction(H)
	tbt = H.mbts[3]
	sparsify!(tbt, trunc_thresh)

	λ = sum(abs.(obt)) + sum(abs.(tbt))
	
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
	#@show S
	ks,s = get_2_factors(S)
	μ = maximum([0, Int(ceil(-log2(coeff_accuracy * S)))])
	#@show μ

	T_prep = 8*ceil(log2(s)) + 4S + 8μ + 56*ceil(log2(H.N)) - 1

	Q_prep = ceil(log2(S)) + 8*ceil(log2(H.N)) + 2μ + 8
	Rq_prep = maximum([s, ceil(log2(S)) - 1, 2μ-1])
	R_prep = 2

	T_sel = 32*(H.N) - 16
	Q_sel = 1 + 2*H.N #other qubits are included in prep
	Rq_sel = ceil(log2(2*H.N)) + 1

	T_tot = T_sel + 2*T_prep
	Q_tot = Q_prep + Q_sel + maximum([Rq_prep, Rq_sel])
	tot_cost = [T_tot, Q_tot, 2 * R_prep]

	return tot_cost..., λ
end


function AC_estimate(H :: F_OP; trunc_thresh = 1e-5)
	sparsify!(H.mbts[3], trunc_thresh)

	AC_coeffs, AC_ops = AC_group(H, ret_ops = true)
	λ = sum(abs.(AC_coeffs))
	G = length(AC_ops)

	Gns = zeros(Int64, G)
	for i in 1:G
		Gns[i] = length(AC_ops[i].paulis)
	end
	S = sum(Gns)

	#@show S, G
	T_prep, Q_prep, Rq_prep, R_prep = prep_estimate(G)
	

	T_sel = 4*(G-1)
	Q_sel = ceil(log2(G)) + 2*H.N + 1
	Rq_sel = ceil(log2(G))
	R_sel = 2*(S-G)

	T_tot = T_sel + 2*T_prep
	Q_tot = Q_prep + Q_sel + maximum([Rq_prep, Rq_sel])
	tot_cost = [T_tot, Q_tot, 2 * R_prep + R_sel]

	return tot_cost..., λ
end

function SYM4_estimate(H :: F_OP; trunc_thresh = 1e-5, max_frags = 10000, kwargs...)
	SYM_frags = SYM4_greedy_decomposition(H, max_frags; kwargs...)

	λ = 0.0

	for frag in SYM_frags
		λ += abs(frag.coeff)
	end

	@show λ
end

function CP4_estimates(H :: F_OP; trunc_thresh = 1e-5, rank_max = 10000, rank_hop = 10, ini_rank = 2, kwargs...)
	sparsify!(H.mbts[3], trunc_thresh)

	CP4_frags = CP4_decomposition(H, rank_max, rank_hop=rank_hop, ini_rank = ini_rank; kwargs...)

	W = length(CP4_frags)
	λ = sum(L1.(CP4_frags))
	Top = H.mbts[2] + ob_correction(H)
	λ += L1(to_OBF(Top))
	givens_beta = max_givens#ceil(5.652 + log2(H.N * λ / Givens_accuracy))

	T_prep, Q_prep, Rq_prep, R_prep = prep_estimate(W)

	T_sel_CP4T = H.N*(112*givens_beta - 196) + 8*W-4
	T_sel_CP4Q = H.N*(112*givens_beta - 196) + 20*W-16
	Q_sel_CP4T = 4 + ceil(log2(W)) + 2*H.N + 4*givens_beta
	Q_sel_CP4Q = 4 + ceil(log2(W)) + 2*H.N + givens_beta

	Rq_sel = ceil(log2(W)) 
	R_sel = 0

	T_tot_CP4T = 2*T_prep + T_sel_CP4T
	T_tot_CP4Q = 2*T_prep + T_sel_CP4Q

	Q_tot_CP4T = Q_prep + Q_sel_CP4T + maximum([Rq_prep, Rq_sel])
	Q_tot_CP4Q = Q_prep + Q_sel_CP4Q + maximum([Rq_prep, Rq_sel])

	tot_cost_T = [T_tot_CP4T, Q_tot_CP4T, 2*R_prep]
	tot_cost_Q = [T_tot_CP4Q, Q_tot_CP4Q, 2*R_prep]

	return tot_cost_T..., tot_cost_Q..., λ
end

function SVD_estimates(H :: F_OP; trunc_thresh = 1e-5, kwargs...)
	sparsify!(H.mbts[3], trunc_thresh)

	SVD_frags = iterative_schmidt(H.mbts[3], tol=trunc_thresh, return_ops=true; kwargs...)

	W = length(SVD_frags)
	λ = sum(L1.(SVD_frags))
	Top = H.mbts[2] + ob_correction(H)
	λ += L1(to_OBF(Top))
	givens_beta = max_givens#ceil(5.652 + log2(H.N * λ / Givens_accuracy))

	T_prep, Q_prep, Rq_prep, R_prep = prep_estimate(W)

	T_sel_CP4T = H.N*(112*givens_beta - 196) + 8*W-4
	T_sel_CP4Q = H.N*(112*givens_beta - 196) + 20*W-16
	Q_sel_CP4T = 4 + ceil(log2(W)) + 2*H.N + 4*givens_beta
	Q_sel_CP4Q = 4 + ceil(log2(W)) + 2*H.N + givens_beta

	Rq_sel = ceil(log2(W)) 
	R_sel = 0

	T_tot_CP4T = 2*T_prep + T_sel_CP4T
	T_tot_CP4Q = 2*T_prep + T_sel_CP4Q

	Q_tot_CP4T = Q_prep + Q_sel_CP4T + maximum([Rq_prep, Rq_sel])
	Q_tot_CP4Q = Q_prep + Q_sel_CP4Q + maximum([Rq_prep, Rq_sel])

	tot_cost_T = [T_tot_CP4T, Q_tot_CP4T, 2*R_prep]
	tot_cost_Q = [T_tot_CP4Q, Q_tot_CP4Q, 2*R_prep]

	return tot_cost_T..., tot_cost_Q..., λ
end

function MPS_estimate(H :: F_OP; kwargs...)
	S1, S2, S3, U1, U2, U3, V3 = tbt_to_mps(H.mbts[3]; kwargs...)
	u_idx, _ = inds(S1)
	v_idx, _ = inds(S2)
	w_idx, _ = inds(S3)

	α1 = u_idx.space
	α2 = v_idx.space
	α3 = w_idx.space

	prep1_cost = prep_estimate(H.N)
	prep2_cost = prep_estimate(α1)
	prep3_cost = prep_estimate(α2)
	prep4_cost = prep_estimate(α3)

	μ1 = maximum([0, Int(ceil(-log2(coeff_accuracy * H.N)))])
	μ2 = maximum([0, Int(ceil(-log2(coeff_accuracy * α1)))])
	μ3 = maximum([0, Int(ceil(-log2(coeff_accuracy * α2)))])
	μ4 = maximum([0, Int(ceil(-log2(coeff_accuracy * α3)))])

	T_prep = prep1_cost[1] + prep2_cost[1] + prep3_cost[1] + prep4_cost[1]
	Q_prep = 13 + ceil(log2(H.N)) + ceil(log2(α2)) + ceil(log2(α3)) + 2μ1 + 2μ2 + 2μ3 + 2μ4
	Rq_prep = 4 + ceil(log2(α2)) + 2μ3
	R_prep = 9

	λ = mps_1_norm(S1,S2,S3)
	Top = H.mbts[2] + ob_correction(H)
	λ += L1(to_OBF(Top))
	givens_beta = max_givens#ceil(5.652 + log2(H.N * λ / Givens_accuracy))
	T_sel = 4α2*(H.N + 2α1 + 2α3) + H.N*(112*givens_beta - 192) + 8α1 + 4α3 - 24
	Rq_sel = ceil(log2(α2*α3))
	Q_sel = 4 + 2*H.N + givens_beta + ceil(log2(H.N)) + ceil(log2(α2)) + ceil(log2(α3))

	tot_cost = [2*T_prep + T_sel, Q_prep + Q_sel + maximum([Rq_prep, Rq_sel]), 2*R_prep]

	return tot_cost..., λ
end

function DF_openfermion_estimate(H :: F_OP; br = num_br, trunc_thresh = 1e-5, kwargs...)
	if DF_tools == false
		error("Trying to do DF estimate with DF_tools set to false, no native QuantumMAMBO implementation... Check config.jl file for more info")
	end

	DF_frags = DF_decomposition(H)
	L = length(DF_frags)
	@show L
	Lxi = 0
	for frag in DF_frags
		e_vec = sqrt(abs(frag.coeff)) .* frag.C.λ
		truncation = sum(abs.(e_vec)) * abs.(e_vec)
		idx = [t > trunc_thresh for t in truncation]
		Lxi += sum(idx)
	end
	@show Lxi

	λ = sum(L1.(DF_frags))
	Top = ob_correction(H) + H.mbts[2]
	λ += L1(to_OBF(Top))
	givens_beta = max_givens#ceil(5.652 + log2(H.N * λ / Givens_accuracy))

	@show givens_beta
	μ = maximum([0, Int(ceil(-log2(coeff_accuracy * (L+1))))])

	@show μ

	T,Q = pyconvert.(Int64, df_tools.compute_cost(2*H.N, λ, L, Lxi, μ, givens_beta, br)) 
	return T,Q,0,λ
end

function get_estimate_variables(N)
	bN = ceil(log2(N))
	kN = floor(log2(N))
	lN = ceil(log2(N/(2^kN)))

	@show bN, kN, lN
	return bN, kN, lN
end

function DF_estimate(H :: F_OP; br = num_br, trunc_thresh = 1e-5, kwargs...)
	if DF_tools == false
		error("Trying to do DF estimate with DF_tools set to false, no native QuantumMAMBO implementation... Check config.jl file for more info")
	end

	DF_frags = DF_decomposition(H; kwargs...)
	L = length(DF_frags)

	T_prep, Q_prep, Rq_prep, R_prep = prep_estimate(L+1)
	
	λ = sum(L1.(DF_frags))
	Top = ob_correction(H) + H.mbts[2]
	λ += L1(to_OBF(Top))

	givens_beta = max_givens#ceil(5.652 + log2(H.N * λ / Givens_accuracy)); @show givens_beta, max_givens

	μN = maximum([0, Int(ceil(-log2(coeff_accuracy * (H.N))))])
	bN, kN, lN = get_estimate_variables(H.N)
	T_sel = L*(8+40lN+16*H.N+32μN+28bN+9kN) + H.N*(28givens_beta-48) + 4bN - 20

	bL,_,_ = get_estimate_variables(L)
	Q_sel = 6+bL+2*H.N+bN+2μN

	Rq_sel = 7+givens_beta+3bN+bL+maximum([2μN-1,bN-1,lN])
	R_sel = 8*L


	tot_cost = [2*T_prep + T_sel, Q_prep + Q_sel + maximum([Rq_prep, Rq_sel]), 2*R_prep + R_sel]

	return tot_cost..., λ
end