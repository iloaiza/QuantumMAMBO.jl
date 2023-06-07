# STRUCTURES FOR FLAVOUR-AGNOSTIC IMPLEMENTATIONS

abstract type OP_FLAVOUR 
	#marks representation of operator, whether qubit, majorana or fermionic
end

struct QUBIT <: OP_FLAVOUR
end

struct MAJORANA <: OP_FLAVOUR
end

struct FERMIONIC <: OP_FLAVOUR
end

abstract type OPERATOR 
	#basic class containing arbitrary operators
	#Idea is to output in this format and have easy connection to openfermion
	#Obtaining operator from electronic structure integrals for easy connection with SCF calculations
	#Representation should be able to be changed as e.g. JW transform
end

struct F_OP <: OPERATOR
	#fermionic operator
	Nbods :: Int64 #how many bodies it covers (e.g. 0, 1, 2, ..., e.g. Ham = 2)
	mbts :: Tuple #mbts[i] correponds to (i-1)-body tensor
	#mbts[i][a1,b1,a2,b2,...,a(i-1),b(i-1)] corresponds to coefficient of Eˆ(a1)_(b1)Eˆ(a2)_(b2)... operator
	filled :: Vector{Bool} #filled[i] = false implies mbts[i] = 0. Saves [0] for type stability
	spin_orb :: Bool #false means tensors correspond to spin-orbitals, true means spin symmetry
	#spin symmetry corresponds to working with F_ij = \sum_(σ) Eˆ(iσ)_(j_σ) instead of E's
	N :: Int64 #number of orbitals/spin-orbitals
end

function F_OP(mbts :: Tuple, spin_orb=false)
	#spin_orb = false by default
	Nbods = length(mbts) - 1
	filled = zeros(Bool,Nbods + 1)
	N = 0
	for i in 1:Nbods + 1
		if mbts[i] != [0]
			filled[i] = true
			N = size(mbts[i])[1]
		end
	end
	
	return F_OP(Nbods, mbts, filled, spin_orb, N)
end

struct M_OP <: OPERATOR
	#Majorana operator
	Nmajs :: Int64 #highest polynomial power of majoranas (e.g. Ham = 4)
	mbts :: Tuple #Nmajs+1, corresponds e.g. mbts[2][i] → γi, mbts[5][i,j,k,l] → γi*γj*γk*γl
	t_coeffs :: Array #Nmajs+1 coefficients multiplying with each tensor, can be used for making mbts real if all multiplied by e.g. i/2
	# OP = sum_i t_coeffs[i] * (mbts[i] .* mtbs_operators) (e.g. mbts_operator[3][3,5] => γ3*γ5)
	filled :: Vector{Bool} #Nmajs+1 corresponding to whether mbts[i] is filled
	body_sym :: Bool #false means all γin γjm are available, true only γi0 γj1 terms appear
	#if true, this makes each mbts[k] dimension be N, otherwise it's 2N
	#true also means Majorana words with odd-number of Majoranas can't appear
	#false -> SO(2N), true -> SO(N)
	spin_orb :: Bool #whether it comes from spin-orb symmetry
	#false -> same coefficients as fermionic h_ij and g_ijkl, constructs majorana operators corresponding to m operators 
	N :: Int64 #number of orbitals/spin-orbitals
end

struct Q_OP <: OPERATOR
	#qubit operator
	N :: Int64 #number of qubits
	n_paulis :: Int64 #number of Pauli terms, doesn't include identity
	id_coeff #coefficient multiplying identity term
	paulis :: Array{pauli_word} #Pauli word structure defined in symplectic.jl
end

function Q_OP(paulis :: Array{pauli_word})
	N = Int(paulis[1].size)
	n_paulis = length(paulis)

	return Q_OP(N, n_paulis, 0.0, paulis)
end

function Q_OP(pauli :: pauli_word)
	N = Int(pauli.size)
	n_paulis = 1

	return Q_OP(N, n_paulis, 0.0, [pauli])
end

struct I_OP <: OPERATOR
	#identity operator, representation independent. Can be used for defining zero operator
	coeff :: Number
end

abstract type PARAMETERS
	#this type carries parameters from which operators can be built
end

abstract type UNITARY <: PARAMETERS
	#a unitary operator can be built from these parameters
end

abstract type CARTANS <: PARAMETERS
	#holds parameters that correspond to cartan polynomial
end

struct cartan_1b <: CARTANS
	#parameters for fermionic cartan 1-body polynomial:
	# C(λ) = ∑_i λi m_i
	# where m_i = n_iα + n_iβ for spin_orb = false, and m_i = n_i for spin_orb = true
	spin_orb :: Bool #whether it acts on orbitals or spin-orbitals
	λ :: Array{Float64, 1} #vector with cartan coefficients
	N :: Int64 #number of orbitals/spin-orbitals
end

function cartan_1b(spin_orb, λ)
	return cartan_1b(spin_orb, λ, length(λ))
end

struct cartan_2b <: CARTANS
	#parameters for fermionic cartan 2-body polynomial:
	# C(λ) = ∑_i≥j λ(ind(i,j)) (mi mj + mj mi)
	# ind(1,1) = 1, ind(1,2) = 2, ind(1,3) = 3, ... ind(1, N) = N, ind(2,2) = N+1, ..., ind(N,N) = N(N+1)/2
	spin_orb :: Bool
	λ :: Array{Float64, 1}
	N :: Int64 #number of orbitals/spin-orbitals
end

function cartan_2b(spin_orb, λ)
	#L = length(λ) -> L = N(N+1)/2
	N = Int(sqrt(1+8*length(λ)) - 1)*0.5
	return cartan_2b(spin_orb, λ, N)
end

struct cartan_m1 <: CARTANS
	#corresponds to m1 = ∑_σ n1σ operators
	#m1 ~ z1α + z1β plus identity terms
end

struct cartan_SD <: CARTANS
	spin_orb :: Bool
	λ1 :: Array{Float64, 1}
	λ2 :: Array{Float64, 1}
	N :: Int64
end

abstract type F_UNITARY <: UNITARY
end

abstract type Q_UNITARY <: UNITARY
end

abstract type M_UNITARY <: UNITARY
end

abstract type REAL_F_UNITARY <: F_UNITARY
	#fermionic unitaries defined only using real numbers/operators (i.e. SO(N) algebra)
end

struct givens_real_orbital_rotation <: REAL_F_UNITARY
	#rotates molecular orbitals (symmetric for both spins)
	N :: Int64 #number of orbitals
	θs :: Array{Float64, 1} #N(N-1)/2 coeffs
	# U(θs) = ∏_(i) θs[i] Gi
	# with G1 = ∑_σ E^(1σ)_(2σ) - E^(2σ)_(1σ), G2 => E^1_3-E^3_1, ..., G(N-1) => E^1_N, GN => E^2_3, ...,  G(N(N-1)/2) => E^(N-1)_N
end

struct real_orbital_rotation <: REAL_F_UNITARY
	#rotates molecular orbitals (symmetric for both spins)
	N :: Int64 #number of orbitals
	θs :: Array{Float64, 1} #N(N-1)/2 coeffs
	# U(θs) = e^(∑_(i) θs[i] Gi)
	# with G1 = ∑_σ E^(1σ)_(2σ) - E^(2σ)_(1σ), G2 => E^1_3-E^3_1, ..., G(N-1) => E^1_N, GN => E^2_3, ...,  G(N(N-1)/2) => E^(N-1)_N
end

struct givens_real_spin_orbital_rotation <: REAL_F_UNITARY
	#SO(N) unitary built using Givens rotations
	N :: Int64 #number of spin-orbitals
	θs :: Array{Float64, 1} #(N(N-1)/2 coeffs
	# U(θs) = ∏_i θs[i] Gi
	# with G1 = E^1_2 - E^2_1, G2 => E^1_3, ..., G(N-1) => E^1_N, GN => E^2_3, ..., G(N(N-1)/2) => E^(N-1)_N
end

struct restricted_orbital_rotation <: REAL_F_UNITARY
	#corresponds to THC rotations, only rotates w/r to 1st orbital
	# U(θs) = ∏_i θs[i] Gi
	# with G1 = E^1_2 - h.c., G2 => E^1_3, ..., G(N-1) => E^1_N
	N :: Int64 #number of orbitals
	θs :: Array{Float64, 1} #N-1 coefficients
end

struct single_majorana_rotation <: REAL_F_UNITARY
	#corresponds to THC rotations as well, but represented as
	#γ_u⃗ = ∑_n u_n γ_n, with ∑_n|u_n|ˆ2 = 1
	# u1 = cos(2θ_1), u2 = sin(2θ_2)cos(2θ_1), ..., ui = cos(2θ_i)∏_{j<i}sin(2θ_j) (for i < N), ..., uN = ∏_{i} sin(2θ_i)
	N :: Int64 #number of orbitals
	θs :: Array{Float64, 1} #N-1 coeffs
end

struct single_orbital_rotation <: REAL_F_UNITARY
	#corresponds to THC rotations as well, but represented as
	#γ_u⃗ = ∑_n c_n γ_n, with ∑_n|c_n|ˆ2 = 1
	N :: Int64 #number of orbitals
	cns :: Array{Float64, 1} #N coeffs, must be normalized
end

struct givens_orbital_rotation  <: F_UNITARY
	#rotates molecular orbitals, includes both real and imaginary generators
	N :: Int64 #number of orbitals
	θs :: Array{Float64, 1} #N^2 coeffs
	# U(θs) = ∏_(i) θs[i] Gi
	# with G1 = ∑_σ E^(1σ)_(2σ) - E^(2σ)_(1σ), G2 => E^1_3-E^3_1, ..., G(N-1) => E^1_N, GN => E^2_3, ...,  G(N(N-1)/2) => E^(N-1)_N
	# G(N(N-1)/2+1) => i(∑_σ E^(1σ)_(2σ) + h.c.), ..., G(N(N-1)) => i(E^(N-1)_N + h.c.)
	# G(N(N-1) + 1) => i∑_σ E^(1σ)_(1σ), ..., G(N^2) => E^N_N
end

struct f_matrix_rotation <: F_UNITARY
	#rotation that is given in matrix form directly
	N :: Int64 #number of orbitals
	mat :: Array{Number, 2}
end

abstract type FRAGMENT <: OPERATOR
	#subtype of operator, corresponds to a single fragment
end

abstract type FRAGMENTATION_TECH
	#has fragmentation technique inside, can be used for building unitarization technique of the fragment
	#e.g. double factorization for fermionic fragment -> unitarized using complete-square Chebyshev encoding
end

struct CSA <: FRAGMENTATION_TECH
	#CSA fragmentation, uses cartan_2b and orbital rotation unitaries
	# OP = U * cartan_2b * U = U†(θs) ∑_(i≥j) λ(ind(i,j)) mi mj U(θ)
end

struct CSA_SD <: FRAGMENTATION_TECH
	#CSA fragmentation for both one+two-body, uses cartan_SD and orbital rotation unitaries
	# OP = U * cartan_SD * U; cartan_SD = cartan_1b + cartan_2b
end

struct DF <: FRAGMENTATION_TECH
	# OP = U' * cartan_1bˆ2 * U = U†(θs) ∑_(i,j) λi λj mi mj U(θ)
end

struct THC <: FRAGMENTATION_TECH
	# OP = U' * m1 * U * V' * m1 * V + h.c. = U[1]'* cartan_m1 * U[1] * U[2]' * cartan_m1 * U[2] + h.c. 
	#(hermitized THC)
	# with m1 = ∑_σ n_1σ
end

struct OBF <: FRAGMENTATION_TECH
	#one-body fragment, can be written as
	#OP = U' * cartan_1b * U
end

struct MTD_CP4 <: FRAGMENTATION_TECH
	# Fragments of CP4 decomposition
	# OP = ∑_στ U[1]' * γ1σ,0 * U[1] * U[2]' * γ1σ,1 * U[2] * U[3]' * γ1τ,0 * U[3] * U[4]' * γ1τ,1 * U[4]
end

struct MTD_PARAFAC <: FRAGMENTATION_TECH
	# Fragments of PARAFAC decomposition
	# same as CP4, but orbital rotations are given by c_n coefficients instead of θs 
	#(i.e. see single_orbital_rotation class vs restricted_orbital_rotation)
	# OP = ∑_στ ∑_{ijkl} c_i * γiσ,0 * c_j * γjσ,1 * c_k * γ1τ,0 * c_l * γ1τ,1
end

struct F_FRAG <: FRAGMENT
	nUs :: Int64 #number of unitaries defining the fragment
	U :: NTuple #tuple containing nUs F_UNITARY objects
	TECH :: FRAGMENTATION_TECH
	C :: CARTANS
	N :: Int64 # number of orbitals/spin-orbitals
	spin_orb :: Bool
	coeff :: Float64 #coefficient multiplying the fragment
	has_coeff :: Bool #whether coefficient is different than 1, for faster implementation
end

function F_FRAG(U::NTuple,TECH::FRAGMENTATION_TECH,C::CARTANS,N::Int64,spin_orb::Bool)
	return F_FRAG(length(U),U,TECH,C,N,spin_orb,1,false)
end

function F_FRAG(nUs::Int64,U::NTuple,TECH::FRAGMENTATION_TECH,C::CARTANS,N::Int64,spin_orb::Bool)
	return F_FRAG(nUs,U,TECH,C,N,spin_orb,1,false)
end

function F_FRAG(U::F_UNITARY,TECH::FRAGMENTATION_TECH,C::CARTANS,N::Int64,spin_orb::Bool)
	return F_FRAG(1,tuple(U),TECH,C,N,spin_orb,1,false)
end

struct AC <: FRAGMENTATION_TECH
	#anti-commuting grouping tech, assumes all Paulis inside anti-commute
end

struct Q_FRAG <: FRAGMENT
	#under construction, can be used for e.g. anti-commuting grouping with rotations corresponding to SO(2N) diagonalization
	#where N is the number of Paulis in anti-commuting group
end

struct M_FRAG <: FRAGMENT
	#under construction, can be used for fragments built from majorana operators
end