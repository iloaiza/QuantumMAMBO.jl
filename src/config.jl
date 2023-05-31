#CONFIGURATION FILE
#Sets defaults for code
const F2Q_map = "jw"
const SAVING = true #save/load quantities as they are generated

#Decomposition options
const DECOMPOSITION_PRINT = false #verbose during decompositions (useful for very large systems). false for no printout
								 #integer number N for print every N steps
const SVD_for_CSA = true #starting conditions for each CSA step are taken from SVD solution
const SVD_for_CSA_SD = true #starting conditions for each CSA_SD step are taken from SVD solution, only considers two-body
const CSA_GIVENS = true #whether unitary rotations for CSA are calculated as products of Givens or directly from e.g. SO(N) algebra exponential
const DF_GIVENS = false #same as CSA_GIVENS but for Double-Factorization
const OO_GIVENS = true #whether orbital-rotation optimization generates unitaries as Givens of SO(N) directly
const OO_reps = 10 #how many parallel repetitions are done for orbital optimization routine

#Tolerances and constants
const ϵ = 1e-6 #decomposition fermionic 2-norm tolerance
const ϵ_Givens = 1e-12 #tolerance for Givens decomposition using maximal torus theorem L2-norm
const SVD_tiny = 1e-8 #tolerance for symmetry-enforcement of SVD fragments
const SVD_tol = 1e-8 #cut-off for SVD coefficients
const PAULI_TOL = 1e-8 #cut-off Pauli terms when building Q_OP qubit operators with magnitude smaller than this tolerance
const LCU_tol = 1e-6 #cut-off for counting unitaries with magnitude smaller than this tolerance

#Additional definitions
const CONFIG_LOADED = true #flag that tracks config already being loaded, useful for redefining constants and being able to load build.jl