#Wrapper for Openfermion THC



ENV["JULIA_CONDAPKG_BACKEND"] = PY_BACKEND

using PythonCall
np = pyimport("numpy")
scipy = pyimport("scipy")
sympy = pyimport("sympy")
of = pyimport("openfermion")
of_thc = pyimport("openfermion.resource_estimates.thc")

UTILS_DIR = @__DIR__
sys = pyimport("sys")
sys.path.append(UTILS_DIR)
ham = pyimport("ham_utils")
fermionic = pyimport("ferm_utils")
qub = pyimport("py_qubits")
thc_utils = pyimport("thc_utils")

of_simplify(OP) = of.reverse_jordan_wigner(of.jordan_wigner(OP))


function OF_THC(F::F_OP, M)
	eri_thc,thc_leaf, thc_central, info= of_thc.thc_via_cp3(F.mbts[3], M)
	@show info, thc_leaf, thc_central
end

