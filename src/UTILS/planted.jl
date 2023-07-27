function MF_planted(H :: F_OP; method="DF-boost", OB = false)
	#=
	returns planted solution (i.e. MF-solvable H) for Hamiltonian H
	method: how the planted solution is obtained. Choose from:
		- DF-boost: CSA fragment that was obtained from frame of largest DF singular value
		- CSA: greedy CSA
		- DF: largest DF fragment
	options are:
		- OB: whether one-body part is included in optimization (true) or optimized in the end for a fixed frame. 
				CSA method runs CSA_SD, which includes one-body tensor in optimization
	=#
	if method == "CSA"
		FRAG = CSA_SD_greedy_decomposition(Hp, 1, SAVELOAD=false)[1]
		H1 = to_OP(FRAG)

		return FRAG
	end

	if OB
		Hso = F_OP_collect_obt(H)
		if method == "DF"
			FRAG = DF_decomposition(Hso)[1]
			return FRAG
		end

		if method == "DF-boost"
			FRAG = DF_based_greedy(Hso)
			return FRAG
		end
	else
		if method == "DF"
			FRAG = DF_decomposition(H)[1]
		end

		if method == "DF-boost"
			FRAG = DF_based_greedy(H)
		end

		FRAG = to_CSA_SD(FRAG)
		Urot = one_body_unitary(FRAG.U[1])
		obt_in_frame = Urot' * H.mbts[2] * Urot

		FRAG.C.λ1 .= diag(obt_in_frame)

		return FRAG
	end
end