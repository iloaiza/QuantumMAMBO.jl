using Test, SafeTestsets

@time begin
    @time @safetestset "Lanczos Tests" include("lanczos_tests.jl")
	@time @safetestset "LP BLISS Tests" include("lpbliss_tests.jl")
    @time @safetestset "LP BLISS Interface Tests" include("lpbliss_interface_tests.jl")
end # @time