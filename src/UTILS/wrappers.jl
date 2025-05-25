function SAVELOAD_HAM(mol_name, FILENAME = DATAFOLDER * mol_name * ".h5", DO_SAVE = SAVING; kwargs...)
	#loads (or generates and saves) Hamiltonian in FILENAME corresponding to mol_name
	if DO_SAVE && isfile(FILENAME*".h5")
		fid = h5open(FILENAME*".h5", "cw")
		if haskey(fid, "MOLECULAR_DATA")
			println("Loading molecular data from $FILENAME.h5")
			MOL_DATA = fid["MOLECULAR_DATA"]
			h_const = read(MOL_DATA,"h_const")
			obt = read(MOL_DATA,"obt")
			tbt = read(MOL_DATA,"tbt")
			η = read(MOL_DATA,"eta")
			close(fid)
			H = F_OP((h_const,obt,tbt))
		else
			H, η = obtain_H(mol_name; kwargs...)
			println("""Saving molecular data in $FILENAME.h5 under group "MOLECULAR_DATA". """)
			if haskey(fid, "MOLECULAR_DATA")
				@warn "Trying to save molecular data to $FILENAME.h5, but MOLECULAR_DATA group already exists. Overwriting and migrating old file..."
				close(fid)
				oldfile(FILENAME*".h5")
				fid = h5open(FILENAME*".h5", "cw")
			end
			create_group(fid, "MOLECULAR_DATA")
			MOL_DATA = fid["MOLECULAR_DATA"]
			MOL_DATA["h_const"] =  H.mbts[1]
			MOL_DATA["obt"] =  H.mbts[2]
			MOL_DATA["tbt"] =  H.mbts[3]
			MOL_DATA["eta"] =  η
			close(fid)
		end
	else 
		H, η = obtain_H(mol_name; kwargs...)
		if DO_SAVE
			println("""Saving molecular data in $FILENAME.h5 under group "MOLECULAR_DATA". """)
			fid = h5open(FILENAME*".h5", "cw")
			if haskey(fid, "MOLECULAR_DATA")
				@warn "Trying to save molecular data to $FILENAME.h5, but MOLECULAR_DATA group already exists."
				close(fid)
				oldfile(FILENAME)
				fid = h5open(FILENAME*".h5", "cw")
			end
			create_group(fid, "MOLECULAR_DATA")
			MOL_DATA = fid["MOLECULAR_DATA"]
			MOL_DATA["h_const"] =  H.mbts[1]
			MOL_DATA["obt"] =  H.mbts[2]
			MOL_DATA["tbt"] =  H.mbts[3]
			MOL_DATA["eta"] =  η
			close(fid)
		end
	end

	return H, η
end

function SAVELOAD_XYZ_HAM(xyz_string, FILENAME = DATAFOLDER * mol_name * ".h5", DO_SAVE = SAVING; kwargs...)
	if DO_SAVE && isfile(FILENAME*".h5")
		fid = h5open(FILENAME*".h5", "cw")
		if haskey(fid, "MOLECULAR_DATA")
			println("Loading molecular data from $FILENAME.h5")
			MOL_DATA = fid["MOLECULAR_DATA"]
			h_const = read(MOL_DATA,"h_const")
			obt = read(MOL_DATA,"obt")
			tbt = read(MOL_DATA,"tbt")
			η = read(MOL_DATA,"eta")
			close(fid)
			H = F_OP((h_const,obt,tbt))
		else
			H, η = H_from_xyz(xyz_string; kwargs...)
			println("""Saving molecular data in $FILENAME.h5 under group "MOLECULAR_DATA". """)
			if haskey(fid, "MOLECULAR_DATA")
				@warn "Trying to save molecular data to $FILENAME.h5, but MOLECULAR_DATA group already exists. Overwriting and migrating old file..."
				close(fid)
				oldfile(FILENAME*".h5")
				fid = h5open(FILENAME*".h5", "cw")
			end
			create_group(fid, "MOLECULAR_DATA")
			MOL_DATA = fid["MOLECULAR_DATA"]
			MOL_DATA["h_const"] =  H.mbts[1]
			MOL_DATA["obt"] =  H.mbts[2]
			MOL_DATA["tbt"] =  H.mbts[3]
			MOL_DATA["eta"] =  η
			close(fid)
		end
	else 
		H, η = H_from_xyz(xyz_string; kwargs...)
		if DO_SAVE
			println("""Saving molecular data in $FILENAME.h5 under group "MOLECULAR_DATA". """)
			fid = h5open(FILENAME*".h5", "cw")
			if haskey(fid, "MOLECULAR_DATA")
				@warn "Trying to save molecular data to $FILENAME.h5, but MOLECULAR_DATA group already exists."
				close(fid)
				oldfile(FILENAME)
				fid = h5open(FILENAME*".h5", "cw")
			end
			create_group(fid, "MOLECULAR_DATA")
			MOL_DATA = fid["MOLECULAR_DATA"]
			MOL_DATA["h_const"] =  H.mbts[1]
			MOL_DATA["obt"] =  H.mbts[2]
			MOL_DATA["tbt"] =  H.mbts[3]
			MOL_DATA["eta"] =  η
			close(fid)
		end
	end

	return H, η
end

function LOCALIZED_XYZ_HAM(xyz_string, FILENAME = DATAFOLDER * mol_name * ".h5", DO_SAVE = SAVING; kwargs...)
	if DO_SAVE && isfile(FILENAME*".h5")
		fid = h5open(FILENAME*".h5", "cw")
		if haskey(fid, "MOLECULAR_DATA")
			println("Loading molecular data from $FILENAME.h5")
			MOL_DATA = fid["MOLECULAR_DATA"]
			h_const = read(MOL_DATA,"h_const")
			obt_hf = read(MOL_DATA,"obt_hf")
			tbt_hf = read(MOL_DATA,"tbt_hf")
			η = read(MOL_DATA,"eta")
			obt_fb = read(MOL_DATA,"obt_fb")
			tbt_fb = read(MOL_DATA,"tbt_fb")
			close(fid)
			Hhf = eri_to_F_OP(obt_hf, tbt_hf, h_const)
			Hfb = eri_to_F_OP(obt_fb, tbt_fb, h_const)
		else
			h_const, obt_hf, tbt_hf, obt_fb, tbt_fb, η = ham.localized_ham_from_xyz(xyz_string; kwargs...)
			h_const = pyconvert(Float64, h_const)
			obt_hf = pyconvert(Array{Float64}, obt_hf)
			tbt_hf = pyconvert(Array{Float64}, tbt_hf)
			obt_fb = pyconvert(Array{Float64}, obt_fb)
			tbt_fb = pyconvert(Array{Float64}, tbt_fb)
			η = pyconvert(Int64, η)
			Hhf = eri_to_F_OP(obt_hf, tbt_hf, h_const)
			Hfb = eri_to_F_OP(obt_fb, tbt_fb, h_const)
			println("""Saving molecular data in $FILENAME.h5 under group "MOLECULAR_DATA". """)
			if haskey(fid, "MOLECULAR_DATA")
				@warn "Trying to save molecular data to $FILENAME.h5, but MOLECULAR_DATA group already exists. Overwriting and migrating old file..."
				close(fid)
				oldfile(FILENAME*".h5")
				fid = h5open(FILENAME*".h5", "cw")
			end
			create_group(fid, "MOLECULAR_DATA")
			MOL_DATA = fid["MOLECULAR_DATA"]
			MOL_DATA["h_const"] =  Hhf.mbts[1]
			MOL_DATA["obt_hf"] =  Hhf.mbts[2]
			MOL_DATA["tbt_hf"] =  Hhf.mbts[3]
			MOL_DATA["eta"] =  η
			MOL_DATA["obt_fb"] =  Hfb.mbts[2]
			MOL_DATA["tbt_fb"] =  Hfb.mbts[3]
			close(fid)
		end
	else 
		h_const, obt_hf, tbt_hf, obt_fb, tbt_fb, η = ham.localized_ham_from_xyz(xyz_string; kwargs...)
		h_const = pyconvert(Float64, h_const)
		obt_hf = pyconvert(Array{Float64}, obt_hf)
		tbt_hf = pyconvert(Array{Float64}, tbt_hf)
		obt_fb = pyconvert(Array{Float64}, obt_fb)
		tbt_fb = pyconvert(Array{Float64}, tbt_fb)
		η = pyconvert(Int64, η)
		Hhf = eri_to_F_OP(obt_hf, tbt_hf, h_const)
		Hfb = eri_to_F_OP(obt_fb, tbt_fb, h_const)
		if DO_SAVE
			println("""Saving molecular data in $FILENAME.h5 under group "MOLECULAR_DATA". """)
			fid = h5open(FILENAME*".h5", "cw")
			if haskey(fid, "MOLECULAR_DATA")
				@warn "Trying to save molecular data to $FILENAME.h5, but MOLECULAR_DATA group already exists."
				close(fid)
				oldfile(FILENAME)
				fid = h5open(FILENAME*".h5", "cw")
			end
			create_group(fid, "MOLECULAR_DATA")
			MOL_DATA = fid["MOLECULAR_DATA"]
			MOL_DATA["h_const"] =  Hhf.mbts[1]
			MOL_DATA["obt_hf"] =  Hhf.mbts[2]
			MOL_DATA["tbt_hf"] =  Hhf.mbts[3]
			MOL_DATA["eta"] =  η
			MOL_DATA["obt_fb"] =  Hfb.mbts[2]
			MOL_DATA["tbt_fb"] =  Hfb.mbts[3]
			close(fid)
		end
	end

	return Hhf, Hfb, η
end

function L1_ROUTINE(H, name; prefix="", dE = true, dE_tol = 1e-1)
	#dE: whether full Hamiltonian is diagonalized for minimum 1-norm
	#runs L1 routine for H, returns array of 1-norms Λ and unitary count Us as:
	#Λ = [ΔE, Pauli, AC, DF, MHC]
	#Us = [Pauli, AC, DF, MHC]

	Λ = zeros(6)
	Us = zeros(Int64, 4)
	println("Obtaining 1-norm lower bound")
	if dE
		fid = h5open(name, "cw")
		if haskey(fid, "dE" * prefix)
			println("Found saved dE for file $name")
			λ_min = read(fid,"dE" * prefix)
		else
			@time λ_min = SQRT_L1(H, tol=dE_tol, verbose=true)
			fid["dE" * prefix] = λ_min
		end
		close(fid)

		Λ[1] = λ_min
	end

	println("Obtaining Fock bound")
	Λ[6] = Fock_bound(H)
	
	println("\n\nCalculating 1-norms...")
	println("1-body:")
	@time λ1 = one_body_L1(H, count=true)

	println("\n\nDoing DF")
	@time DF_FRAGS = DF_decomposition(H, verbose=true)
	println("Finished DF decomposition for 2-body term using $(length(DF_FRAGS)) fragments")
	@time λ2_DF = sum(L1.(DF_FRAGS, count=true))
	λDF = λ1 + λ2_DF
	Λ[4] = λDF[1]
	Us[3] = λDF[2]

	println("\nMHC:")
	@time λ2_MHC = split_schmidt(H.mbts[3], count=true, tol=1e-6)
	λMHC = λ1 + λ2_MHC
	Λ[5] = λMHC[1]
	Us[4] = λMHC[2]
	
	println("\nPauli:")
	@time λPauli = PAULI_L1(H, count=true)
	Λ[2] = λPauli[1]
	Us[1] = λPauli[2]
	
	
	println("\nAnti-commuting:")
	@time λAC, N_AC = AC_group(H, ret_ops=false)
	Λ[3] = λAC
	Us[2] = N_AC
	
	#=
	println("\nOrbital-rotation routine:")
	@time H_rot = ORBITAL_OPTIMIZATION(H, SAVENAME=name)
	λOO_AC, N_OO_AC = AC_group(H_rot, ret_ops=false)
	Λs[i,6] = λOO_AC
	NUM_US[i,5] = N_OO_AC
	# =#

	return Λ, Us
end

function INTERACTION(H; SAVENAME=DATAFOLDER*"INTERACTION.h5",verbose=false)
	H0 = CSA_SD_greedy_decomposition(H :: F_OP, 1, verbose=verbose, SAVENAME=SAVENAME)[1]
	HR = H - H0

	return HR
end

function ORBITAL_OPTIMIZATION(H; verbose=true, SAVELOAD=SAVING, SAVENAME=DATAFOLDER*"OO.h5", do_Givens = OO_GIVENS)
	θmin = false
	if SAVELOAD
		fid = h5open(SAVENAME, "cw")
		if "ORBITAL_OPTIMIZATION" in keys(fid)
			OO = fid["ORBITAL_OPTIMIZATION"]
			if haskey(OO, "theta")
				θmin = read(OO, "theta")
			end
		else
			create_group(fid, "ORBITAL_OPTIMIZATION")
			OO = fid["ORBITAL_OPTIMIZATION"]
		end
	end

	if θmin == false
		H_rot, _, _, θmin = orbital_l1_optimizer(H, verbose=verbose, ret_op=true)
		if SAVELOAD
			OO["theta"] = θmin
		end
	else
		println("Found saved θmin under filename $SAVENAME for orbital optimization, loaded orbital rotation...")
		if do_Givens
			U = givens_real_orbital_rotation(H.N, θmin)
		else
			U = real_orbital_rotation(H.N, θmin)
		end
		H_rot = F_OP_rotation(U, H)
	end

	if SAVELOAD
		close(fid)
	end

	return H_rot
end

function RUN(H; kwargs...)
	@warn "RUN function name has been deprecated, use RUN_L1 instead..."
	return RUN_L1(H; kwargs...)
end

function RUN_L1(H; DO_CSA = true, DO_DF = true, DO_ΔE = true, DO_AC = true, DO_OO = true,
			 DO_SQRT = false, max_frags = 10000, verbose=true, COUNT=false, DO_LANCZOS=DO_LANCZOS, DO_TROTTER = false,
			 DO_MHC = true, DO_MTD_CP4 = true, name = SAVING, SAVELOAD = SAVING, LATEX_PRINT = true, η=0,
			 DO_FC = true, SYM_RED = true, DO_THC = false, FOCK_BOUND=true)
	# Obtain 1-norms for different LCU methods. COUNT=true also counts number of unitaries in decomposition
	# CSA: Cartan sub-algebra decomposition
	# DF: Double Factorization
	# ΔE: Exact lower bound from diagonalization of H
	# FOCK_BOUND: (efficient) computation of lower bound for ΔE using Fock matrix
	# AC: Anticommuting grouping
	# OO: Orbital rotation technique
	# SQRT: obtain square-root lower bound for non-optimal factorization methods (i.e. CSA)
	# TROTTER: obtain α upper-bound for Trotter error
	# MHC: MTD-1ˆ4 with SVD-based MPS
	# MTD_CP4: MTD-1ˆ4
	# η: number of electrons in the wavefunction, necessary for symmetry-projected Trotter bounds
	# FC: fully-commuting grouing
	# SYM_RED: calculate Trotter norms in symmetric subspace, necessary for Trotter bounds
	# name: default name for saving, false means no saving is done

	if name == true
		name = DATAFOLDER*"RUN.h5"
	end

	METHODS = []
	Λs = []

	if DO_ΔE
		println("Obtaining 1-norm lower bound")
		if SAVELOAD
			fid = h5open(name, "cw")
			if haskey(fid, "dE")
				println("Found saved dE for file $name")
				λ_min = read(fid,"dE")
			else
				@time λ_min = SQRT_L1(H)
				fid["dE"] = λ_min
			end
			close(fid)
		end
		@show λ_min
		push!(METHODS, "ΔE/2")
		push!(Λs, λ_min)
	end

	if FOCK_BOUND
		println("Calculating lower bound for ΔE/2 using Fock matrix...")
		@time λ_fock = Fock_bound(H)
		@show λ_fock
		push!(METHODS, "Fock")
		push!(Λs, λ_fock)
	end
	
	if DO_LANCZOS
		println("Estimating ΔE/2 using Lanczos iteration.")
		
		
		#=if !(spin_symmetry==false && rohf==false && (bliss==true || compress==false))
			H_spin=F_OP_space_to_spin(H)
			#tensors=(H_spin.mbts[2],H_spin.mbts[3])
		else
			H_spin=H
			#tensors=(H.mbts[2],H.mbts[3])
		end=#
		H_spin=F_OP_space_to_spin(H)
			
		
		E_max_total,E_min_total = lanczos_total_range(one_body_tensor=H_spin.mbts[2],two_body_tensor=H_spin.mbts[3],core_energy=H_spin.mbts[1][1], steps=5)
    		E_max, E_min= lanczos_range(one_body_tensor=H_spin.mbts[2],two_body_tensor=H_spin.mbts[3],core_energy=H_spin.mbts[1][1], steps=5, num_electrons=η)
    		
    		
		delta_E_total=E_max_total - E_min_total
		delta_E=E_max-E_min
		println("Total spectral gap/2: ", delta_E_total/2)
		println("Spectral gap/2 for ",η," electrons: ", delta_E/2)
	end

	if SYM_RED
		if SAVELOAD
			fid = h5open(name, "cw")
			if haskey(fid, "SYM_RED")
				sym_group = fid["SYM_RED"]
				println("Found saved SYM_RED for file $name")
				Uη = read(sym_group,"Ueta")
				Hmat = read(sym_group,"Hmat")
			else
				create_group(fid, "SYM_RED")
				sym_group = fid["SYM_RED"]
				Uη = Ne_block_diagonalizer(H.N, η, spin_orb=H.spin_orb)
				println("Building reduced H matrix...")
				@time Hmat = matrix_symmetry_block(to_matrix(H), Uη)
				sym_group["Ueta"] = Uη
				sym_group["Hmat"] = Hmat
				println("Saved Hmat and Uη in group SYM_RED in file $name")
			end
			close(fid)
		end
	end

	println("\nPauli:")
	@time λPauli = PAULI_L1(H, count=COUNT)
	@show λPauli
	push!(METHODS, "Pauli")
	push!(Λs, λPauli)

	if DO_OO
		println("\nOrbital-rotation Pauli routine:")
		@time H_rot = ORBITAL_OPTIMIZATION(H, verbose=verbose, SAVENAME=name)
		λOO_Pauli = PAULI_L1(H_rot, count=COUNT)
		@show λOO_Pauli
		push!(METHODS, "OO-Pauli")
		push!(Λs, λOO_Pauli)
	end
	
	if DO_AC
		println("\nAnti-commuting:")
		@time λAC, N_AC = AC_group(H, ret_ops=false)
		if COUNT
			λAC = [λAC, N_AC]
		end
		@show λAC
		push!(METHODS, "AC")
		push!(Λs, λAC)
	end

	if DO_OO && DO_AC
		λOO_AC, N_OO_AC = AC_group(H_rot, ret_ops=false)
		if COUNT
			λOO_AC = [λOO_AC, N_OO_AC]
		end
		@show λOO_AC, N_OO_AC
		push!(METHODS, "OO-AC")
		push!(Λs, λOO_AC)
	end

	println("\n\nCalculating 1-norms...")
	println("1-body:")
	@time λ1 = one_body_L1(H, count=COUNT)
	#@time λ1 = SQRT_L1(OB_extract(H,γ4_contribution=true), count=COUNT)
	@show λ1

	if DO_TROTTER
		ob_frag = to_OBF(H.mbts[2])
	end

	if DO_CSA
		println("Doing CSA")
		max_frags = 100
		@time CSA_FRAGS = CSA_greedy_decomposition(H, max_frags, verbose=verbose, SAVENAME=name)
		println("Finished CSA decomposition for 2-body term using $(length(CSA_FRAGS)) fragments")
		@time λ2_CSA = sum(L1.(CSA_FRAGS, count=COUNT))
		@show λCSA = λ1 + λ2_CSA
		push!(METHODS, "GCSA-F")
		push!(Λs, λCSA)
		if DO_SQRT
			println("Square-root routine...")
			@time λ2_CSA_SQRT = sum(SQRT_L1.(CSA_FRAGS, count=COUNT))
			@show λCSA_SQRT = λ1 + λ2_CSA_SQRT
			push!(METHODS, "GCSA-SR")
			push!(Λs, λCSA_SQRT)
		end
		if DO_TROTTER
			#=
			println("Starting Trotter routine for CSA...")
			CSA_TROT = CSA_FRAGS
			push!(CSA_TROT,ob_frag)
			@time β_CSA = trotter_β(CSA_TROT)
			@show β_CSA
			#@time α_CSA = parallel_trotter_α(CSA_TROT)
			#@show α_CSA
			# =#
		end
	end

	if DO_DF
		println("\n\nDoing DF")
		@time DF_FRAGS = DF_decomposition(H, verbose=verbose)
		println("Finished DF decomposition for 2-body term using $(length(DF_FRAGS)) fragments")
		@time λ2_DF = sum(L1.(DF_FRAGS, count=COUNT))
		@show λDF = λ1 + λ2_DF
		push!(METHODS, "DF")
		push!(Λs, λDF)
		if DO_TROTTER
			println("Starting Trotter routine for DF...")
			DF_OPS = to_OP.(DF_FRAGS) - ob_correction.(DF_FRAGS, return_op = true)
			ob_op = F_OP(H.mbts[2]) + ob_correction(H, return_op=true)
			push!(DF_OPS, ob_op)
			@time DF_mats = parallel_to_reduced_matrices(DF_OPS, Uη, SAVELOAD = true, SAVENAME = name, GNAME = "DF")
			id_mat = Diagonal(ones(size(DF_mats[1])[1]) * H.mbts[1][1])
			push!(DF_mats, id_mat)
			#@time α_DF = parallel_trotter_comparer(Hmat, DF_mats)
			#@show α_DF
			#@time αs_DF = parallel_autocorr_trotter(Hmat, DF_mats)
			#@show αs_DF
			@time αψ, αf, αT, αT4 = parallel_all_trotter(Hmat, DF_mats)
			@show αψ
			@show αf
			println("αT:")
			display(αT)
			println("αT4:")
			display(αT4)
		end
	end


	if DO_MHC
		println("\nMHC:")
		@time λ2_MHC = iterative_schmidt(H.mbts[3], count=COUNT, tol=1e-6)
		@show λMTD = λ1 + λ2_MHC
		push!(METHODS, "MTD-SVD")
		push!(Λs, λMTD)
	end

	if DO_MTD_CP4
		#=
		println("\nMTD_CP4:")
		@time CP4_FRAGS = CP4_decomposition(H, max_frags, verbose=verbose, SAVELOAD = true, SAVENAME = name)
		λ2_CP4_TSR = 4*sum([abs(frag.coeff) for frag in CP4_FRAGS])
		if COUNT == true
			λ2_CP4_TSR = [λ2_CP4_TSR, length(CP4_FRAGS)]
		end
		@show λ1 + λ2_CP4_TSR
		# =#

		println("\nGREEEDY_MTD_CP4:")
		@time CP4_GREEDY_FRAGS = MTD_CP4_greedy_decomposition(H, max_frags, verbose=false, SAVENAME = name, SAVELOAD=true)
		λ2_CP4_GREEDY = 4*sum([abs(frag.coeff) for frag in CP4_GREEDY_FRAGS])
		if COUNT == true
			λ2_CP4_GREEDY = [λ2_CP4_GREEDY, length(CP4_GREEDY_FRAGS)]
		end
		@show λ1 + λ2_CP4_GREEDY
		push!(METHODS, "MTD-1^4")
		push!(Λs, λ1 + λ2_CP4_GREEDY)
	end

	if DO_THC
		println("\n\nTHC routine...")
		#@warn "THC not implemented!"
		#=
		@time λ2_THC = THC_full(H)
		@show λ1 + λ2_THC
		# =#
		
		step_size=40
		@time _, λ2_THC, iterations= THC_tb_lsq(H, step_size)
		@show λ_THC = λ1+λ2_THC
		@show iterations
		@show λ2_THC
		
		#Toffoli_cost
		#λ2_THC=5.61
		#iterations=3
		
		step_cost, toffoli_cost, ancilla_cost= THC_cost(2*H.N, λ2_THC, step_size, iterations)
		@show step_cost, toffoli_cost, ancilla_cost 
		
		push!(METHODS, "THC")
		push!(Λs, λ_THC)
		
	end

	if DO_FC
		println("\nFully-commuting:")
		@time λFC, FC_OPS = FC_group(H, ret_ops=true)
		if COUNT
			@show λFC, length(FC_OPS)
		else
			@show λFC
		end

		if DO_TROTTER
			@time FC_mats = parallel_to_reduced_matrices(FC_OPS, Uη, SAVELOAD = true, SAVENAME = name, GNAME = "FC")
			Hq = Q_OP(H)
			id_mat = Diagonal(ones(size(FC_mats[1])[1]) * Hq.id_coeff)
			push!(FC_mats, id_mat)
			#@time α_FC = parallel_trotter_comparer(Hmat, FC_mats)
			#@show α_FC
			#@time αs_FC = parallel_autocorr_trotter(Hmat, FC_mats)
			#@show αs_FC
			@time αψ, αf, αT, αT4 = parallel_all_trotter(Hmat, FC_mats)
			@show αψ
			@show αf
			println("αT:")
			display(αT)
			println("αT4:")
			display(αT4)
		end
	end
	
	if LATEX_PRINT
		println("\n\n\nFINISHED ROUTINE FOR $name, PRINTING LATEX TABLE...")
		println("#########################################################")
		println("#########################################################")
		println("Printout legend: Parenthesis corresponds to #of unitaries when available")
		methods_string = ""
		num_methods = length(METHODS)
		for i in 1:num_methods
			meth = METHODS[i]
			methods_string = methods_string * meth
			if i < num_methods
				methods_string = methods_string * " & "
			end
		end
		println(methods_string)

		λ_string = ""
		for i in 1:num_methods
			λ_string = λ_string * "$(round(sigdigits=3, Λs[i][1]))"
			if length(Λs[i]) > 1
				λ_string = λ_string * "($(Int(Λs[i][2])))"
			end

			if i < num_methods
				λ_string = λ_string * " & "
			end
		end
		println(λ_string)
		println("#########################################################")
		println("#########################################################")
	end
end

function HUBBARD_RUN(H; DO_CSA = false, DO_DF = true, DO_ΔE = false, DO_AC = true, DO_OO = true,
			 DO_SQRT = false, max_frags = 1000, verbose=true, COUNT=false, DO_TROTTER = false,
			 DO_MHC = true, DO_MTD_CP4 = true, name = SAVING, SAVELOAD = SAVING, LATEX_PRINT = true, η=0,
			 DO_FC = true, SYM_RED = true)
	# Obtain 1-norms for different LCU methods. COUNT=true also counts number of unitaries in decomposition
	# CSA: Cartan sub-algebra decomposition
	# DF: Double Factorization
	# ΔE: Exact lower bound from diagonalization of H
	# AC: Anticommuting grouping
	# OO: Orbital rotation technique
	# SQRT: obtain square-root lower bound for non-optimal factorization methods (i.e. CSA)
	# TROTTER: obtain α upper-bound for Trotter error
	# MHC:L Majorana Hyper-Contraction
	# η: number of electrons in the wavefunction, necessary for symmetry-projected Trotter bounds
	# FC: fully-commuting grouing
	# SYM_RED: calculate Trotter norms in symmetric subspace, necessary for Trotter bounds
	# name: default name for saving, false means no saving is done

	if name == true
		name = DATAFOLDER*"RUN.h5"
	end

	if DO_ΔE
		println("Obtaining 1-norm lower bound")
		if SAVELOAD
			fid = h5open(name, "cw")
			if haskey(fid, "dE")
				println("Found saved dE for file $name")
				λ_min = read(fid,"dE")
			else
				@time λ_min = SQRT_L1(H)
				fid["dE"] = λ_min
			end
			close(fid)
		end
		@show λ_min
	end

	if SYM_RED
		if SAVELOAD
			fid = h5open(name, "cw")
			if haskey(fid, "SYM_RED")
				sym_group = fid["SYM_RED"]
				println("Found saved SYM_RED for file $name")
				Uη = read(sym_group,"Ueta")
				Hmat = read(sym_group,"Hmat")
			else
				create_group(fid, "SYM_RED")
				sym_group = fid["SYM_RED"]
				Uη = Ne_block_diagonalizer(H.N, η, spin_orb=H.spin_orb)
				println("Building reduced H matrix...")
				@time Hmat = matrix_symmetry_block(to_matrix(H), Uη)
				sym_group["Ueta"] = Uη
				sym_group["Hmat"] = Hmat
				println("Saved Hmat and Uη in group SYM_RED in file $name")
			end
			close(fid)
		end
	end

	println("\n\nCalculating 1-norms...")
	println("1-body:")
	@time λ1 = one_body_L1(H, count=COUNT)
	@show λ1

	if DO_TROTTER
		ob_frag = to_OBF(H.mbts[2])
	end

	if DO_CSA
		println("Doing CSA")
		max_frags = 100
		@time CSA_FRAGS = CSA_greedy_decomposition(H, max_frags, verbose=verbose, SAVENAME=name)
		println("Finished CSA decomposition for 2-body term using $(length(CSA_FRAGS)) fragments")
		@time λ2_CSA = sum(L1.(CSA_FRAGS, count=COUNT))
		@show λ1 + λ2_CSA
		if DO_SQRT
			println("Square-root routine...")
			@time λ2_CSA_SQRT = sum(SQRT_L1.(CSA_FRAGS, count=COUNT))
			@show λ1 + λ2_CSA_SQRT
		end
		if DO_TROTTER
			#=
			println("Starting Trotter routine for CSA...")
			CSA_TROT = CSA_FRAGS
			push!(CSA_TROT,ob_frag)
			@time β_CSA = trotter_β(CSA_TROT)
			@show β_CSA
			#@time α_CSA = parallel_trotter_α(CSA_TROT)
			#@show α_CSA
			# =#
		end
	end

	if DO_DF
		println("\n\nDoing DF")
		@time DF_FRAGS = DF_decomposition(H, verbose=verbose)
		println("Finished DF decomposition for 2-body term using $(length(DF_FRAGS)) fragments")
		@time λ2_DF = sum(L1.(DF_FRAGS, count=COUNT))
		@show λ1 + λ2_DF
		if DO_TROTTER
			println("Starting Trotter routine for DF...")
			DF_OPS = to_OP.(DF_FRAGS) - ob_correction.(DF_FRAGS, return_op = true)
			ob_op = F_OP(H.mbts[2]) + ob_correction(H, return_op=true)
			push!(DF_OPS, ob_op)
			@time DF_mats = parallel_to_reduced_matrices(DF_OPS, Uη, SAVELOAD = true, SAVENAME = name, GNAME = "DF")
			id_mat = Diagonal(ones(size(DF_mats[1])[1]) * H.mbts[1][1])
			push!(DF_mats, id_mat)
			#@time α_DF = parallel_trotter_comparer(Hmat, DF_mats)
			#@show α_DF
			@time αs_DF = parallel_autocorr_trotter(Hmat, DF_mats)
			@show αs_DF
		end
	end


	if DO_MHC
		println("\nMHC:")
		@time λ2_MHC = split_schmidt(H.mbts[3], count=COUNT, tol=1e-6)
		@show λ1 + λ2_MHC
	end

	if DO_MTD_CP4
		println("\nMTD_CP4:")
		@time CP4_FRAGS = CP4_decomposition(H, max_frags)
		λ2_CP4_TSR = sum([abs(frag.coeff) for frag in CP4_FRAGS])
		if COUNT == true
			λ2_CP4_TSR = [λ2_CP4_TSR, length(CP4_FRAGS)]
		end
		@show λ1 + λ2_CP4_TSR

		println("\nGREEEDY_MTD_CP4:")
		@time CP4_GREEDY_FRAGS = MTD_CP4_greedy_decomposition(H, max_frags, verbose=verbose, SAVENAME = name)
		λ2_CP4_GREEDY = sum([abs(frag.coeff) for frag in CP4_FRAGS])
		if COUNT == true
			λ2_CP4_GREEDY = [λ2_CP4_GREEDY, length(CP4_GREEDY_FRAGS)]
		end
		@show λ1 + λ2_CP4_GREEDY
	end

	println("\nPauli:")
	Hof = to_OF(H)
	Hq = of.jordan_wigner(Hof)
	n_qubits = H.N
	if H.spin_orb == false
		@warn "Running Hubbard model for spin-orb=false, be wary of results!"
		n_qubits *= 2
	end

	of_pw_list, of_coefs = qub.get_pauliword_list(Hq)

    num_paulis = length(of_pw_list)
    bin_vecs = zeros(Bool, 2*n_qubits, num_paulis)
    coeffs = zeros(Complex, num_paulis)

    pws = pauli_word[]
    for i in 1:num_paulis
    	bin_vecs[:,i] = of_pauli_word_to_binary_vector(of_pw_list[i], n_qubits)
    	coeffs[i] = of_coefs[i]
    	push!(pws, pauli_word(bin_vecs[:,i], coeffs[i]))
    end

    H_Q = Q_OP(pws)

	@time λPauli = PAULI_L1(H_Q, count=COUNT)
	@show λPauli
	
	if DO_AC
		println("\nAnti-commuting:")
		@time λAC, N_AC = AC_group(H_Q, ret_ops=false)
		if COUNT
			@show λAC, N_AC
		else
			@show λAC
		end
	end

	if DO_FC
		println("\nFully-commuting:")
		@time λFC, FC_OPS = FC_group(H_Q, ret_ops=true)
		if COUNT
			@show λFC, length(FC_OPS)
		else
			@show λFC
		end

		if DO_TROTTER
			@time FC_mats = parallel_to_reduced_matrices(FC_OPS, Uη, SAVELOAD = true, SAVENAME = name, GNAME = "FC")
			id_mat = Diagonal(ones(size(FC_mats[1])[1]) * H_Q.id_coeff)
			push!(FC_mats, id_mat)
			#@time α_FC = parallel_trotter_comparer(Hmat, FC_mats)
			#@show α_FC
			@time αs_FC = parallel_autocorr_trotter(Hmat, FC_mats)
			@show αs_FC
		end
	end

	if DO_OO
		error("Orbital optimization not defined for Hubbard model, or in general for spin-orb=true!")
		#=
		println("\nOrbital-rotation routine:")
		@time H_rot = ORBITAL_OPTIMIZATION(H, verbose=verbose, SAVENAME=name)
		λOO_Pauli = PAULI_L1(H_rot, count=COUNT)
		@show λOO_Pauli
		if DO_AC
			λOO_AC, N_OO_AC = AC_group(H_rot, ret_ops=false)
			if COUNT
				@show λOO_AC, N_OO_AC
			else
				@show λOO_AC
			end
		end
		# =#
	end

	λFCSA = λ1 + λ2_CSA
	
	if LATEX_PRINT
		println("\n\n\nFINISHED ROUTINE FOR $name, PRINTING LATEX TABLE...")
		println("#########################################################")
		println("#########################################################")
		println("Printout legend: Parenthesis corresponds to #of unitaries when available")
		println("ΔE/2 & λPauli & λAC & λGCSA")
		println("$(round(sigdigits=3,λ_min)) & $(round(sigdigits=3,λPauli[1]))($(Int(λPauli[2]))) & $(round(sigdigits=3,λAC))($N_AC) & $(round(sigdigits=3,λFCSA[1]))($(Int(λFCSA[2])))")
		println("#########################################################")
		println("#########################################################")
	end
end
