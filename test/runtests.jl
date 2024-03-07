using Test, SafeTestsets

@time begin
    @time @safetestset "Lanczos Tests" include("lanczos_tests.jl")
end # @time
