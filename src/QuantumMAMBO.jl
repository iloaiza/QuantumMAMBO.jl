module QuantumMAMBO
	using Distributed, LinearAlgebra, Einsum, Optim, SharedArrays, JuMP, Arpack, SparseArrays

	src_dir = @__DIR__
	UTILS_dir = src_dir * "/UTILS/"
	if !(@isdefined CONFIG_LOADED) #only include config file one time so constants can be later redefined
		include("config.jl")
	end

	include(UTILS_dir * "symplectic.jl")
	include(UTILS_dir * "structures.jl")
	include(UTILS_dir * "unitaries.jl")
	include(UTILS_dir * "fermionic.jl")
	include(UTILS_dir * "cost.jl")
	include(UTILS_dir * "gradient.jl")
	include(UTILS_dir * "decompose.jl")
	include(UTILS_dir * "symmetries.jl")
	include(UTILS_dir * "linprog.jl")
	include(UTILS_dir * "guesses.jl")
	include(UTILS_dir * "lcu.jl")
	function __init__()
		include(UTILS_dir * "py_utils.jl")
	end
	include(UTILS_dir * "majorana.jl")
	include(UTILS_dir * "qubit.jl")
	include(UTILS_dir * "orbitals.jl")
	include(UTILS_dir * "bliss.jl")
	include(UTILS_dir * "trotter.jl")
	include(UTILS_dir * "projectors.jl")
	include(UTILS_dir * "schmidt.jl")

	if @isdefined myid
		include(UTILS_dir * "parallel.jl")
	end

	include(UTILS_dir * "planted.jl")
	include(UTILS_dir * "wrappers.jl")
	include(UTILS_dir * "hubbard.jl")


	if !(@isdefined SAVING_LOADED) && SAVING #only include saving file one time if saving option is on
		include(UTILS_dir * "saving.jl")
		global SAVING_LOADED = true
	end
end
