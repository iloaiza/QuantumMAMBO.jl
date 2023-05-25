using DistributedArrays

function parallel_trotter_α(F_ARR; verbose=true, SAVELOAD=false, SAVENAME = DATAFOLDER*"PARALLEL_MATS.h5")
	#calculates commutator norm ∑_{nm} ||[Hn,Hm]||
	#must include 1-body fragment as well
	M = length(F_ARR)
	α_ijs = SharedArray(zeros(M, M-1))

	ij_dict = zeros(Int64, Int(M*(M-1)/2), 2)
	idx = 0
	for i in 1:M
		for j in 1:i-1
			idx += 1
			ij_dict[idx,:] = [i,j]
		end
	end

	if verbose
		println("Building operator matrices...")
		@time MATS = parallel_to_matrix(F_ARR, SAVELOAD=SAVELOAD, SAVENAME=SAVENAME)
	else
		MATS = parallel_to_matrix(F_ARR, SAVELOAD=SAVELOAD, SAVENAME=SAVENAME)
	end

	if verbose
		println("Calculating commutator ranges...")
		t00 = time()
	end
	@sync @distributed for idx in 1:Int(M*(M-1)/2)
		i,j = ij_dict[idx,:]
		#commutator is anti-hermitian, needs multiplication by -i to become real...
		@time comm_range = sparse_matrix_commutator_range(-1im*MATS[i],MATS[j])
		α_ijs[i,j] = 2*maximum(abs.(comm_range))
	end
	if verbose
		println("Fininshed commutator ranges after $(time() - t00) seconds")
	end

	return sum(α_ijs)
end

function parallel_trotter_αsym(F_ARR, η; verbose=false, SAVELOAD=false, SAVENAME = DATAFOLDER*"PARALLEL_MATS.h5")
	#calculates commutator norm ∑_{nm} ||[Hn,Hm]||
	#must include 1-body fragment as well
	M = length(F_ARR)
	α_ijs = SharedArray(zeros(M, M-1))

	ij_dict = zeros(Int64, Int(M*(M-1)/2), 2)
	idx = 0
	for i in 1:M
		for j in 1:i-1
			idx += 1
			ij_dict[idx,:] = [i,j]
		end
	end

	if verbose
		println("Building operator matrices...")
		@time MATS = parallel_to_matrix(F_ARR, SAVELOAD=SAVELOAD, SAVENAME=SAVENAME)
	else
		MATS = parallel_to_matrix(F_ARR, SAVELOAD=SAVELOAD, SAVENAME=SAVENAME)
	end

	if verbose
		println("Projecting to symmetric subspace...")
		t00 = time()
		
		Uη = Ne_block_diagonalizer(F_ARR[1].N, η)
		for i in 1:M
			@time MATS[i] = matrix_symmetry_block(MATS[i], Uη)
		end
		println("Finished symmetry projections after $(time() - t00) seconds")
	else
		Uη = Ne_block_diagonalizer(F_ARR[1].N, η)
		for i in 1:M
			MATS[i] = matrix_symmetry_block(MATS[i], Uη)
		end
	end
	if verbose
		println("Calculating commutator ranges...")
		t00 = time()
	end
	@sync @distributed for idx in 1:Int(M*(M-1)/2)
		i,j = ij_dict[idx,:]
		#commutator is anti-hermitian, needs multiplication by -i to become real...
		@time comm_range = sparse_matrix_commutator_range(-1im*MATS[i],MATS[j])
		α_ijs[i,j] = 2*maximum(abs.(comm_range))
	end
	if verbose
		println("Fininshed commutator ranges after $(time() - t00) seconds")
	end

	return sum(α_ijs)
end

function parallel_to_matrix(F_ARR; SAVELOAD = false, SAVENAME = DATAFOLDER*"PARALLEL_MATS.h5")
	#returns matrix array from array of objects that were transformed into sparse matrix form by to_matrix()
	#has saveload features for large arrays
	if SAVELOAD
		fid = h5open(SAVENAME, "cw")
		if haskey(fid, "PARALLEL_MATS")
			println("Found save data for parallel matrices, loading...")
			PAR_MATS = fid["PARALLEL_MATS"]
			F_len = read(PAR_MATS, "nfrags")
			if F_len != length(F_ARR)
				error("Loaded matrices do not match length of input array, terminating")
			end
			M = read(PAR_MATS, "M")
			N = read(PAR_MATS,"N")
			COLS = [Vector{Int64}[] for _ in 1:F_len]
			ROWS = [Vector{Int64}[] for _ in 1:F_len]
			NZVALS = [Vector{Float64}[] for _ in 1:F_len]
			for i in 1:F_len
				push!(COLS[i], read(PAR_MATS, "COLS$i"))
				push!(ROWS[i], read(PAR_MATS, "ROWS$i"))
				push!(NZVALS[i], read(PAR_MATS, "NZVALS$i"))
			end

			close(fid)
				
			return [SparseMatrixCSC(M, N, COLS[i][1], ROWS[i][1], NZVALS[i][1]) for i in 1:F_len]
		else
			create_group(fid, "PARALLEL_MATS")
			PAR_MATS = fid["PARALLEL_MATS"]
		end
	end

	F_len = length(F_ARR)

	rm_workers = false
	if F_len < nworkers()
		w_ids = workers()
		w_diff = nworkers() - F_len
		println("Removing $w_diff workers for parallel matrix computation...")
		@show w_ids
		rmprocs(w_ids[end-w_diff+1:end])
		@show F_len
		@show nworkers()
		rm_workers = true
	end

	MS = SharedArray(zeros(Int64, F_len))
	NS = SharedArray(zeros(Int64, F_len))
	COLS = distribute([Vector{Int64}[] for _ in 1:F_len])
	ROWS = distribute([Vector{Int64}[] for _ in 1:F_len])
	NZVALS = distribute([Vector{Float64}[] for _ in 1:F_len])

	curr_run = distribute(Int64[0 for _ in 1:nworkers()])
	@sync @distributed for i in 1:F_len
		localpart(curr_run)[1] += 1
		my_run = localpart(curr_run)[1]
		mat = to_matrix(F_ARR[i])
		MS[i] = mat.m
		NS[i] = mat.n
		push!(localpart(COLS)[my_run], mat.colptr)
		push!(localpart(ROWS)[my_run], mat.rowval)
		push!(localpart(NZVALS)[my_run], mat.nzval)
	end

	if SAVELOAD
		if !prod(MS .== MS[1]) || !prod(NS .== NS[1])
			@warn "Not all matrices have same size, loading saved parallel matrix data might have errors..."
		end
		PAR_MATS["nfrags"] = F_len
		PAR_MATS["M"] = MS[1]
		PAR_MATS["N"] = NS[1]
		for i in 1:F_len
			PAR_MATS["COLS$i"] = COLS[i][1]
			PAR_MATS["ROWS$i"] = ROWS[i][1]
			PAR_MATS["NZVALS$i"] = NZVALS[i][1]
		end
		println("Saved matrices in $SAVENAME")
		close(fid)
	end


	ret_mats = [SparseMatrixCSC(MS[i], NS[i], COLS[i][1], ROWS[i][1], NZVALS[i][1]) for i in 1:F_len]

	return ret_mats
end

function parallel_to_reduced_matrices(F_ARR, Uη; SAVELOAD = false, SAVENAME = DATAFOLDER*"PARALLEL_SYM_RED_MATS.h5", GNAME = "")
	#returns matrix array from array of objects that were transformed into sparse matrix form by to_matrix()
	#also reduces each of these matrices into symmetric subspace by matrix_symmetry_block(mat, Uη)
	#has saveload features for large arrays
	if SAVELOAD
		if length(GNAME) != 0
			group_name = GNAME * "_" * "PARALLEL_SYM_RED_MATS"
		else
			group_name = "PARALLEL_SYM_RED_MATS"
		end
		fid = h5open(SAVENAME, "cw")
		if haskey(fid, group_name)
			println("Found save data for parallel symmetry reduced matrices, loading...")
			PAR_MATS = fid[group_name]
			F_len = read(PAR_MATS, "nfrags")
			if F_len != length(F_ARR)
				error("Loaded matrices do not match length of input array, terminating")
			end
			MATS = [Matrix{Float64}[] for _ in 1:F_len]
			for i in 1:F_len
				push!(MATS[i], read(PAR_MATS, "MAT$i"))
			end

			close(fid)
				
			return [MATS[i][1] for i in 1:F_len]
		end
		close(fid)
	end

	F_len = length(F_ARR)

	rm_workers = false
	if F_len < nworkers()
		w_ids = workers()
		w_diff = nworkers() - F_len
		println("Removing $w_diff workers for parallel matrix computation...")
		@show w_ids
		rmprocs(w_ids[end-w_diff+1:end])
		@show F_len
		@show nworkers()
		rm_workers = true
	end

	SYM_MATS = distribute([Matrix{Float64}[] for _ in 1:F_len])
	curr_run = distribute(Int64[0 for _ in 1:nworkers()])
	@sync @distributed for i in 1:F_len
		@show localpart(curr_run)
		localpart(curr_run)[1] += 1
		my_run = localpart(curr_run)[1]
		@show my_run
		mat = to_matrix(F_ARR[i])
		@show size(mat)
		sym_mat = matrix_symmetry_block(mat, Uη)
		@show size(sym_mat)
		push!(localpart(SYM_MATS)[my_run], sym_mat)
		@show size.(localpart(SYM_MATS)[my_run]...)
	end

	if SAVELOAD
		fid = h5open(SAVENAME, "cw")
		create_group(fid, group_name)
		PAR_MATS = fid[group_name]
		PAR_MATS["nfrags"] = F_len
		for i in 1:F_len
			PAR_MATS["MAT$i"] = SYM_MATS[i][1]
		end
		println("Saved matrices in $SAVENAME under group name $group_name")
		close(fid)
	end

	if rm_workers
		@warn "Removed workers for distributed calculation of symmetry-reduced matrix, rest of calculation will only use $(nworkers()) workers..."
	end

	ret_mats = [SYM_MATS[i][1] for i in 1:F_len]

	return ret_mats
end

function of_parallel_trotter_α(FRAGS :: Array{F_FRAG}; verbose=true)
	#calculates commutator norm ∑_{nm} ||[Hn,Hm]||
	#must include 1-body fragment as well
	M = length(FRAGS)
	α_ijs = SharedArray(zeros(M, M-1))

	ij_dict = zeros(Int64, Int(M*(M-1)/2), 2)
	idx = 0
	for i in 1:M
		for j in 1:i-1
			idx += 1
			ij_dict[idx,:] = [i,j]
		end
	end

	@sync @distributed for idx in 1:Int(M*(M-1)/2)
		i,j = ij_dict[idx,:]
		op_i = -1im*qubit_transform(to_OF(FRAGS[i]))
		op_j = qubit_transform(to_OF(FRAGS[j]))
		comm_range = OF_qubit_op_range(of.commutator(op_i, op_j))
		α_ijs[i,j] = 2*maximum(abs.(comm_range))
	end
	
	return sum(α_ijs)
end

function parallel_trotter_comparer(Hmat, mats, t=1e-5)
	if debug
		mat_diff = sum(abs2.(Hmat - sum(mats)))
		println("Difference between matrix sum and Hmat is $mat_diff")
	end

	Tnum = length(mats)
	Uexact = exp(-1im*t*Hmat)

	Umats = SharedArray(zeros(Complex{Float64}, size(mats[1])..., Tnum))
	@sync @distributed for i in 1:Tnum
		Umats[:,:,i] = exp(-1im*t*mats[i])
	end

	Utrotter = Umats[:,:,1]
	for i in 2:Tnum
		Utrotter *= Umats[:,:,i]
	end

	E,_ = eigen(Utrotter - Uexact)
	spec_err = maximum(abs.(E))

	return spec_err/(t^2)
end

function parallel_trotter_unitaries(Hmat, mats, t=1e-5)
	Tnum = length(mats)
	Uexact = exp(-1im*t*Hmat)

	Umats = SharedArray(zeros(Complex{Float64}, size(mats[1])..., Tnum))
	@sync @distributed for i in 1:Tnum
		Umats[:,:,i] = exp(-1im*t*mats[i])
	end

	Utrotter = Umats[:,:,end]
	for i in 1:Tnum-1
		Utrotter *= Umats[:,:,end-i]
	end

	return Uexact, Utrotter
end

function parallel_autocorr_trotter(Hmat, mats, Ts = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-5, 1e-2, 5e-2])
	num_Ts = length(Ts)

	Eex, Uex = eigen(Hmat)
	ind=sortperm(Eex)
    Eex = Eex[ind]
    Uex=Uex[:,ind]

    fci = Uex[:,1]

    exact_autocorr = zeros(Complex,num_Ts)
    trotter_autocorr = zeros(Complex,num_Ts)
	for i in 1:num_Ts
		Uex, Utrot = parallel_trotter_unitaries(Hmat, mats, Ts[i])
		exact_autocorr[i] = dot(fci, Uex*fci)
		trotter_autocorr[i] = dot(fci, Utrot*fci)
	end

	return abs.(exact_autocorr - trotter_autocorr) ./ (Ts .^2)
end

function parallel_all_trotter(Hmat, mats, Ts = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-5, 1e-2, 5e-2], Tint = [10,100,1000])
	num_Ts = length(Ts)
	int_part = length(Tint)

	Eex, Uex = eigen(Hmat)
	ind=sortperm(Eex)
    Eex = Eex[ind]
    Uex=Uex[:,ind]

    fci = Uex[:,1]

    αψ = zeros(num_Ts)
    αf = zeros(num_Ts)
    αT = zeros(num_Ts, int_part)
    αT4 = zeros(num_Ts, int_part)

	for i in 1:num_Ts
		Uex, Utrot = parallel_trotter_unitaries(Hmat, mats, Ts[i])
		ψex = Uex*fci
		ψtrot = Utrot*fci

		ψdiff = ψex - ψtrot
		αψ[i] = sqrt(dot(ψdiff,ψdiff)) / (Ts[i]^2)

		ex_auto = dot(fci, Uex*fci)
		trot_auto = dot(fci, Utrot*fci)
		αf[i] = abs(ex_auto - trot_auto) / (Ts[i]^2)

		for (j,num_ts) in enumerate(Tint)
			int_val_3 = 0.0
			int_ex = 0.0 + 0im
			int_trot = 0.0 + 0im
			ex_ψ = copy(fci)
			trot_ψ = copy(fci)
			for ts in 1:num_ts
				ex_ψ = Uex * ex_ψ
				trot_ψ = Utrot * trot_ψ
				int_val_3 += abs(dot(fci, ex_ψ) - dot(fci, trot_ψ))
				int_ex += dot(fci, ex_ψ)
				int_trot += dot(fci, trot_ψ)
			end
			αT[i,j] = (int_val_3 * Ts[i] / (Tint[j]^3))
			αT4[i,j] = (abs(int_ex - int_trot) * Ts[i] / (Tint[j]^3))
		end
	end

	return αψ, αf, αT, αT4
end
