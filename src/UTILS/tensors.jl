function itensor_from_tbt(tbt)
	N = size(tbt)[1]

	i = Index(N, "i")
	j = Index(N, "j")
	k = Index(N, "k")
	l = Index(N, "l")

	A = ITensor(typeof(tbt[1]), tbt, i, j, k, l)

	return A
end

function tbt_to_mps(tbt; debug=false)
	i_tbt = itensor_from_tbt(tbt)
	i_idx, j_idx, k_idx, l_idx = inds(A)

	U1, S1, V1 = svd(i_tbt, i_idx, lefttags="u_left", righttags="u_right")
	u_left, u_right = inds(S1)

	U2, S2, V2 = svd(V1, u_right, j_idx, lefttags="v_left", righttags="v_right")
	v_left, v_right = inds(S2)

	U3, S3, V3 = svd(V2, v_right, k_idx, lefttags="w_left", righttags="w_right")
	
	
	if debug
		i_tbt_rebuilt = U1 * S1 * U2 * S2 * U3 * S3 * V3
		N = size(tbt)[1]
		tbt_dbg = Array(i_tbt_rebuilt, i_idx, j_idx, k_idx, l_idx)

		@show sum(abs.(tbt - tbt_dbg))
	end

	return S1, S2, S3, U1, U2, U3, V3
end

function mtd_svd_lcu(tbt)
	S1, S2, S3, U1, U2, U3, V3 = tbt_to_mps(tbt)

	u_idx, _ = inds(S1)
	v_idx, _ = inds(S2)
	w_idx, _ = inds(S3)

	u_len = u_idx.space
	v_len = v_idx.space
	w_len = w_idx.space
	
	Stot = zeros(u_len, v_len, w_len)
	for i1 in 1:u_len
		for i2 in 1:v_len
			for i3 in 1:w_len
				Stot[i1,i2,i3] = S1[i1,i1] * S2[i2,i2] * S3[i3,i3]
			end
		end
	end

	return sum(abs.(Stot))
end