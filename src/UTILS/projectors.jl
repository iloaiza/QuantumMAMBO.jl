#projection of operators into symmetric subspaces of interest
function Ne_block_diagonalizer(N, η; η_tol = 1e-12, verbose=false, debug=false, spin_orb=false)
	Ne = Ne_builder(N, spin_orb)
	Ne_mat = collect(to_matrix(Ne))

	if verbose
		println("Diagonalizing Ne...")
		@time Neigs, U = eigen(Ne_mat)
	else
		Neigs, U = eigen(Ne_mat)
	end
	#Ne_mat = U * Diagonal(Neigs) * U'
	
	if verbose
		println("Obtaining indices of right symmetry...")
		t00 = time()
	end
	η_idxs = Int64[]
	for i in 1:length(Neigs)
		if abs(Neigs[i] - η) < η_tol
			push!(η_idxs, i)
		end
	end

	if verbose
		println("Finished obtaining indices after $(time() - t00) seconds")
		println("Symmetry subspace dimension is $(length(η_idxs))")
	end

	Uη = U[:, η_idxs]


	if debug
		η_red  = Diagonal(η*ones(length(η_idxs)))
		Ndbg = Uη' * Ne_mat * Uη
		diff = sum(abs.(Ndbg - η_red))
		println("Difference between symmetry-reduced N and full N is $diff (should be ≈0!)")
	end

	return Uη
end

function matrix_symmetry_block(mat, Uη)
	return Uη' * mat * Uη
end