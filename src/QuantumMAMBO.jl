module QuantumMAMBO
	using Distributed, LinearAlgebra, Einsum, Optim, SharedArrays, JuMP, Arpack, SparseArrays

	if !(@isdefined CONFIG_LOADED) #only include config file one time so constants can be later redefined
		include("config.jl")
	end

	if !(@isdefined SAVING_LOADED) && SAVING #only include saving file one time if saving option is on
		include("UTILS/saving.jl")
		global SAVING_LOADED = true
	end

	include("UTILS/symplectic.jl")
	include("UTILS/structures.jl")
	include("UTILS/unitaries.jl")
	include("UTILS/fermionic.jl")
	include("UTILS/cost.jl")
	include("UTILS/decompose.jl")
	include("UTILS/symmetries.jl")
	include("UTILS/linprog.jl")
	include("UTILS/guesses.jl")
	include("UTILS/lcu.jl")
	include("UTILS/py_utils.jl")
	include("UTILS/majorana.jl")
	include("UTILS/qubit.jl")
	include("UTILS/orbitals.jl")
	include("UTILS/bliss.jl")
	include("UTILS/trotter.jl")
	include("UTILS/projectors.jl")

	if @isdefined myid
		include("UTILS/parallel.jl")
	end

	include("UTILS/wrappers.jl")
end
