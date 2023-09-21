import Pkg

Pkg.activate("./") # uncomment for using local QuantumMAMBO installation
import QuantumMAMBO as QM


U_arr = [0.01, 0.1, 0.2, 0.5, 1, 5, 10, 100]
J = 1
nsites = 6


ΔEs = zeros(length(U_arr))
ΔEs_bliss = zeros(length(U_arr))

λs = zeros(length(U_arr))
λs_bliss = zeros(length(U_arr))

for (i,U) in enumerate(U_arr)
	println("Starting calculations for U=$U")
	println("Bulding Hubbard Hamiltonian and getting measures...")
	H = QM.build_1D_hubbard(U, J, nsites)
	ΔEs[i] = QM.SQRT_L1(H)
	λs[i] = QM.PAULI_L1(H, count=false)

	η = nsites
	sz = 0
	s2 = 0.25 * nsites * (nsites + 2) #S^2 eigenvalues = s*(s+1), for electrons s=1/2

	println("Obtaining BLISS shift and getting measures...")
	@time H_bliss = QM.hubbard_bliss_optimizer(H, η, sz, s2)
	ΔEs_bliss[i] = QM.SQRT_L1(H_bliss)
	λs_bliss[i] = QM.PAULI_L1(H_bliss, count=false)
end

@show ΔEs
@show ΔEs_bliss
@show λs
@show λs_bliss