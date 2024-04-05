import Pkg

Pkg.activate("./")
Pkg.instantiate()

using QuantumMAMBO
using PythonCall


# Inputs
######
data_file_path = "examples/data/fcidump.36_1ru_II_2pl"
lpbliss_hdf5_output_loading_file_path = "examples/data/36_1ru_II_2pl_BLISS.h5"
lpbliss_fcidump_output_file_path = "examples/data/fcidump.36_1ru_II_2pl_BLISS"
#If lpbliss_hdf5_output_loading_file_path already exists, 
# bliss_linprog will load tensors from the h5 file and return the operator

# NOTE: The tensors in the lpbliss_hdf5_output_loading_file_path hdf5 file assume the Hamiltonian is in the form:
# H = E_0 + h_ij a†_i a_j + g_ijkl a†_i a_j a†_k a_l 
# NOTE: The tensors read from and written to the FCIDUMP files assume the Hamiltonian is in the form:
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

# Run LPBLISS
######
H_bliss_cmp,K_operator=QuantumMAMBO.bliss_linprog(H_orig_cmp, 
    num_electrons,
    model="highs", # LP solver used by Gurobi; "highs" or "ipopt". Both give the same answer, while "highs" is faster.
    verbose=true,
    SAVELOAD=true, 
    SAVENAME=lpbliss_hdf5_output_loading_file_path)
println("BLISS optimization/operator retrieval complete.")


# Retrieve the tensors from the fermionic operator
######
# These tensors assume the Hamiltonian is in the form:
# H = E_0 + h_ij a†_i a_j + 0.5*g_ijkl a†_i a†_k a_l a_j
one_body_tensor_bliss, two_body_tensor_bliss = QuantumMAMBO.F_OP_to_eri(H_bliss)
core_energy_bliss = core_energy #LPBLISS does not change the core energy
println("Tensors retrieved from the fermionic operator.")

# Save the tensors to a FCIDUMP file
######
# The tensors written assume the Hamiltonian is in the form:
# H = E_0 + h_ij a†_i a_j + 0.5*g_ijkl a†_i a†_k a_l a_j
pyscf = pyimport("pyscf")
np = pyimport("numpy")
pyscf.tools.fcidump.from_integrals(filename=lpbliss_fcidump_output_file_path, 
    h1e=np.array(one_body_tensor_bliss), 
    h2e=np.array(two_body_tensor_bliss), 
    nmo=num_orbitals, 
    nelec=num_electrons, 
    nuc=core_energy_bliss, 
    ms=two_S, 
    orbsym=nothing, # "C1" point group symmetry is assumed, so orbsym will be written as [1,1,1,...,1]
    tol=1e-15, # All |h_ij| or |g_ijkl| less than tol will be discarded
    float_format=" %.16g"
    )
println("LPBLISS-modified tensors written to FCIDUMP file.")


# Calculate halfbandwidths, which are the lower bound on the L1 norm of the Hamiltonian
################################################################################################

#Original Hamiltonian
println("Calculating halfbandwidths for the original Hamiltonian")
one_body_tensor_chemist_spatial_orbitals = H_orig.mbts[2]
two_body_tensor_chemist_spatial_orbitals = H_orig.mbts[3]
println("one_body_tensor_chemist_spatial_orbitals shape: ", size(one_body_tensor_chemist_spatial_orbitals))
println("two_body_tensor_chemist_spatial_orbitals shape: ", size(two_body_tensor_chemist_spatial_orbitals))


QuantumMAMBO.eliminate_small_values!(one_body_tensor_chemist_spatial_orbitals, 1e-8)
QuantumMAMBO.eliminate_small_values!(two_body_tensor_chemist_spatial_orbitals, 1e-8)
println("Small values eliminated.")
@time begin
E_max_orig, E_min_orig= QuantumMAMBO.lanczos_total_range(one_body_tensor=one_body_tensor_chemist_spatial_orbitals, 
                                                        two_body_tensor=two_body_tensor_chemist_spatial_orbitals, 
                                                        core_energy=core_energy, 
                                                        initial_states=[],
                                                        num_electrons_list=[], 
                                                        steps=25, #Increase this for more accurate results
                                                        multiprocessing=false, #Setting to true may cause a segfault
                                                        spin_orbitals=false
                                                        )
end
delta_E_div_2_orig = (E_max_orig - E_min_orig) / 2
println("Original Hamiltonian:")
println("E_max_orig: ", E_max_orig)
println("E_min_orig: ", E_min_orig)
println("delta_E_div_2_orig: ", delta_E_div_2_orig)

#LPBLISS-modified Hamiltonian
println("Calculating halfbandwidths for the LPBLISS-modified Hamiltonian")
one_body_tensor_bliss_spatial_orbitals = H_bliss.mbts[2]    
two_body_tensor_bliss_spatial_orbitals = H_bliss.mbts[3]
println("one_body_tensor_bliss_spatial_orbitals shape: ", size(one_body_tensor_bliss_spatial_orbitals))
println("two_body_tensor_bliss_spatial_orbitals shape: ", size(two_body_tensor_bliss_spatial_orbitals))

QuantumMAMBO.eliminate_small_values!(one_body_tensor_bliss_spatial_orbitals, 1e-8)
QuantumMAMBO.eliminate_small_values!(two_body_tensor_bliss_spatial_orbitals, 1e-8)
println("Small values eliminated.")
@time begin
E_max_bliss, E_min_bliss= QuantumMAMBO.lanczos_total_range(one_body_tensor=one_body_tensor_bliss_spatial_orbitals, 
                                                            two_body_tensor=two_body_tensor_bliss_spatial_orbitals, 
                                                            core_energy=core_energy, 
                                                            initial_states=[],
                                                            num_electrons_list=[], 
                                                            steps=25, #Increase this for more accurate results
                                                            multiprocessing=false, #Setting to true may cause a segfault
                                                            spin_orbitals=false
                                                            )
end 
delta_E_div_2_bliss = (E_max_bliss - E_min_bliss) / 2
println("LPBLISS-modified Hamiltonian:")
println("E_max_bliss: ", E_max_bliss)
println("E_min_bliss: ", E_min_bliss)
println("delta_E_div_2_bliss: ", delta_E_div_2_bliss)
