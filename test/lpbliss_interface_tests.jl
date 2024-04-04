using Test
using QuantumMAMBO
using PythonCall

atol_test = 1E-6
rtol_test = 1E-6

@testset "lpbliss_interface" begin
# Inputs
######
data_file_path = "../examples/data/fcidump.36_1ru_II_2pl"
lpbliss_hdf5_output_loading_file_path = "data/36_1ru_II_2pl_BLISS.h5"
lpbliss_fcidump_output_file_path = "data/fcidump.36_1ru_II_2pl_BLISS"
#If the ouptut file already exists, bliss_linprog will load the tensors from the h5 file 
# Not a restart, just load and return operator

# NOTE: The tensors in the lpbliss_hdf5_output_loading_file_path hdf5 file assume the Hamiltonian is in the form:
# H = E_0 + h_ij a†_i a_j + g_ijkl a†_i a_j a†_k a_l 
# NOTE: The tensors written to the FCIDUMP file assume the Hamiltonian is in the form:
# H = E_0 + h_ij a†_i a_j + 0.5*g_ijkl a†_i a†_k a_l a_j


# Read FCIDUMP file
######
(
    one_body_tensor,
    two_body_tensor,
    core_energy,
    num_orbitals,
    num_spin_orbitals,
    num_electrons,
    two_S,
    two_Sz,
    orb_sym,
    extra_attributes,
) = QuantumMAMBO.load_tensors_from_fcidump(data_file_path=data_file_path)
# The tensors stored in the FCIDUMP file are assumed to fit the following definition of the Hamiltonian:
# H = E_0 + h_ij a†_i a_j + 0.5*g_ijkl a†_i a†_k a_l a_j
# where i,j,k,l are indices for the spatial orbitals (NOT spin orbitals)
# The full g_ijkl tensor, not a symmetry-compressed version, is returned.
# "C1" point group symmetry is assumed

println("Number of orbitals: ", num_orbitals)
println("Number of spin orbitals: ", num_spin_orbitals)
println("Number of electrons: ", num_electrons)
println("Two S: ", two_S)
println("Two Sz: ", two_Sz)
println("Orbital symmetry: ", orb_sym)
println("Extra attributes: ", extra_attributes)



# Convert to QuantumMAMBO fermion operator
######
H_orig = QuantumMAMBO.eri_to_F_OP(one_body_tensor, two_body_tensor, core_energy, spin_orb=false)
println("Fermionic operator generated.")

one_body_tensor_reverse, two_body_tensor_reverse = QuantumMAMBO.F_OP_to_eri(H_orig)
@test all(isapprox.(one_body_tensor_reverse, one_body_tensor, atol=atol_test, rtol=rtol_test))
# # Run LPBLISS
# ######
# H_bliss,_=QuantumMAMBO.bliss_linprog(H_orig, 
#     num_electrons,
#     model="highs", # Default , alternative is "ipopt" (what does this do?)
#     verbose=true,
#     SAVELOAD = true, 
#     SAVENAME=lpbliss_hdf5_output_loading_file_path) #What is the second output (i.e. what ends up in "_" - killer operator)
# println("BLISS optimization complete.")
# # highs is faster, both give the same answer. They are different solvers for the same problem.
# # Retrieve the tensors from the fermionic operator
# ######
# # These tensors assume the Hamiltonian is in the form:
# # H = E_0 + h_ij a†_i a_j + 0.5*g_ijkl a†_i a†_k a_l a_j
# one_body_tensor_bliss, two_body_tensor_bliss = QuantumMAMBO.F_OP_to_eri(H_bliss)
# core_energy_bliss = core_energy #LPBLISS does not change the core energy
# println("Tensors retrieved from the fermionic operator.")
# println("one_body_tensor_bliss: ", one_body_tensor_bliss)
# println("two_body_tensor_bliss: ", two_body_tensor_bliss)
# println("one_body_tensor_bliss shape: ", size(one_body_tensor_bliss))
# println("two_body_tensor_bliss shape: ", size(two_body_tensor_bliss))

# # Save the tensors to a FCIDUMP file
# ######
# # The tensors written assume the Hamiltonian is in the form:
# # H = E_0 + h_ij a†_i a_j + 0.5*g_ijkl a†_i a†_k a_l a_j
# pyscf = pyimport("pyscf")
# np = pyimport("numpy")
# pyscf.tools.fcidump.from_integrals(filename=lpbliss_fcidump_output_file_path, 
#     h1e=np.array(one_body_tensor_bliss), 
#     h2e=np.array(two_body_tensor_bliss), 
#     nmo=num_orbitals, 
#     nelec=num_electrons, 
#     nuc=core_energy_bliss, 
#     ms=two_S, 
#     orbsym=nothing, # "C1" point group symmetry is assumed, so orbsym will be written as [1,1,1,...,1]
#     tol=1e-15, # All |h_ij| or |g_ijkl| less than tol will be discarded
#     float_format=" %.16g"
#     )
# println("LPBLISS-modified tensors written to FCIDUMP file.")

# pyscf.tools.fcidump.from_integrals(filename="fcidump.testme", 
#     h1e=np.array(one_body_tensor), 
#     h2e=np.array(two_body_tensor), 
#     nmo=num_orbitals, 
#     nelec=num_electrons, 
#     nuc=core_energy_bliss, 
#     ms=two_S, 
#     orbsym=nothing, # "C1" point group symmetry is assumed, so orbsym will be written as [1,1,1,...,1]
#     tol=1e-15, # All |h_ij| or |g_ijkl| less than tol will be discarded
#     float_format=" %.16g"
#     )
# println("LPBLISS-modified tensors written to FCIDUMP file.")
end