using Test
using QuantumMAMBO
using PythonCall
pyscf = pyimport("pyscf")
feru = pyimport("ferm_utils")
open_ferm = pyimport("openfermion")
sp_linalg = pyimport("scipy.sparse.linalg")
sdstate_lanczos = pyimport("module_sdstate.lanczos_utils")

atol_lanczos_test = 1E-10
rtol_lanczos_test = 1E-10

# TEMPFOLDER = "test/tmp"

# function calc_energy_range_all_num_electrons(;H_to_use,num_spin_orbitals,num_electrons_list,spin_preserving,nuc_rep_energy)
#     # Get sparse matrix from openfermion
#     H_orig_sparseMat = open_ferm.get_number_preserving_sparse_operator(
#         fermion_op=open_ferm.normal_ordered(H_to_use),
#         num_qubits=num_spin_orbitals,
#         num_electrons=,
#         spin_preserving=true,
#         reference_determinant=nothing,
#         excitation_level=nothing,
#     )

#     # Solve eigenvalue problem for GS energy
#     GS_orig_FCI_eig_val = sp_linalg.eigs(
#         A=H_orig_sparseMat,
#         k=1,
#         M=nothing,
#         sigma=nothing,
#         which="SR",  # Smallest real part eigenvalues.
#         v0=nothing,
#         ncv=nothing,
#         maxiter=1000,
#         tol=0,  # 0 implies machine precision
#         return_eigenvectors=false,
#         Minv=nothing,
#         OPinv=nothing,
#         # mode="normal", # not applied as sigma is None
#     )

#     GS_orig_FCI_eig_val_final = real(pyconvert(Complex, GS_orig_FCI_eig_val[0]) + nuc_rep_energy)

#     # Solve eigenvalue problem for largest energy
#     max_orig_FCI_eig_val = sp_linalg.eigs(
#         A=H_orig_sparseMat,
#         k=1,
#         M=nothing,
#         sigma=nothing,
#         which="LR",  # Smallest real part eigenvalues.
#         v0=nothing,
#         ncv=nothing,
#         maxiter=1000,
#         tol=0,  # 0 implies machine precision
#         return_eigenvectors=false,
#         Minv=nothing,
#         OPinv=nothing,
#         # mode="normal", # not applied as sigma is None
#     )

#     max_orig_FCI_eig_val_final = real(pyconvert(Complex, max_orig_FCI_eig_val[0]) + nuc_rep_energy)
# end

function calc_lanczos_test_energies(; basis::String, geometry::String)
    mol = pyscf.gto.Mole()
    mol.basis = basis
    mol.atom = geometry
    mol.build()

    #Get FCI energies, code mostly from https://pyscf.org/user/ci.html
    ##################

    myhf = mol.RHF().run()
    #
    # create an FCI solver based on the SCF object
    #
    cisolver = pyscf.fci.FCI(myhf)
    E_FCI_HF = pyconvert(Float64, cisolver.kernel()[0])

    #
    # create an FCI solver based on the SCF object
    #
    myuhf = mol.UHF().run()
    cisolver = pyscf.fci.FCI(myuhf)
    E_FCI_UHF = pyconvert(Float64, cisolver.kernel()[0])

    #
    # create an FCI solver based on the given orbitals and the num. electrons and
    # spin of the mol object
    #
    cisolver = pyscf.fci.FCI(mol, myhf.mo_coeff)
    E_FCI_orb = pyconvert(Float64, cisolver.kernel()[0])

    # Get metadata
    num_electrons = pyconvert(Int64,mol.nelectron)
    num_orb = pyconvert(Int64,mol.nao_nr())

    # Get one and two body tensors
    # Based on the approach used in Block2 in get_rhf_integrals
    # of pyblock2/_pyscf/ao2mo/integrals.py at https://github.com/block-hczhai/block2-preview
    ##################
    mo_coeff = myhf.mo_coeff
    mo_coeff_julia = pyconvert(Array{Float64}, mo_coeff)
    h_core_julia = pyconvert(Array{Float64}, myhf.get_hcore())
    one_body_tensor = mo_coeff_julia' * h_core_julia * mo_coeff_julia


    g2e = pyscf.ao2mo.full(myhf._eri, mo_coeff)
    two_body_tensor = pyscf.ao2mo.restore(symmetry="s1", eri=g2e, norb=num_orb)

    nuc_rep_energy = pyconvert(Float64, mol.energy_nuc())

    #Get openfermion version of the Hamiltonian
    ##################
    two_body_fermi_op = feru.get_ferm_op_two(tbt=0.5 * two_body_tensor, spin_orb=false)
    one_body_fermi_op = feru.get_ferm_op_one(obt=one_body_tensor, spin_orb=false)

    # Get the one body operator given the two body terms are now in chemist notation 
    # Put into normal order to spawn the negative of the extra one body terms
    temp_ferm_op = open_ferm.normal_ordered(two_body_fermi_op)

    # Get the one body terms generated and negate them
    extra_terms = feru.get_one_body_terms(temp_ferm_op)

    # Add the one body terms to the original one body terms
    one_body_fermi_op += -1 * extra_terms

    H_to_use = two_body_fermi_op + one_body_fermi_op
    num_spin_orbitals = 2 * size(one_body_tensor)[1]

    # Get one body tensor for the two body term being in chemist notation
    one_body_tensor_chemist = pyconvert(Array{Float64,2},feru.get_obt(H=one_body_fermi_op, n=2 * num_orb, spin_orb=true))

    # Get the FCI energy based on Lanczos from Scipy
    # For both the ground state and the largest energy
    ##################

    # Get sparse matrix from openfermion
    H_orig_sparseMat = open_ferm.get_number_preserving_sparse_operator(
        fermion_op=open_ferm.normal_ordered(H_to_use),
        num_qubits=num_spin_orbitals,
        num_electrons=num_electrons,
        spin_preserving=true,
        reference_determinant=nothing,
        excitation_level=nothing,
    )

    # Solve eigenvalue problem for GS energy
    GS_orig_FCI_eig_val = sp_linalg.eigs(
        A=H_orig_sparseMat,
        k=1,
        M=nothing,
        sigma=nothing,
        which="SR",  # Smallest real part eigenvalues.
        v0=nothing,
        ncv=nothing,
        maxiter=1000,
        tol=0,  # 0 implies machine precision
        return_eigenvectors=false,
        Minv=nothing,
        OPinv=nothing,
        # mode="normal", # not applied as sigma is None
    )

    GS_orig_FCI_eig_val_final = real(pyconvert(Complex, GS_orig_FCI_eig_val[0]) + nuc_rep_energy)

    # Solve eigenvalue problem for largest energy
    max_orig_FCI_eig_val = sp_linalg.eigs(
        A=H_orig_sparseMat,
        k=1,
        M=nothing,
        sigma=nothing,
        which="LR",  # Smallest real part eigenvalues.
        v0=nothing,
        ncv=nothing,
        maxiter=1000,
        tol=0,  # 0 implies machine precision
        return_eigenvectors=false,
        Minv=nothing,
        OPinv=nothing,
        # mode="normal", # not applied as sigma is None
    )

    max_orig_FCI_eig_val_final = real(pyconvert(Complex, max_orig_FCI_eig_val[0]) + nuc_rep_energy)

    # Get the energies as calculated by module_sdstate
    ##################

    # Get the two body tensor in spin orbital form from the openfermion operator
    two_body_tensor_spin_orb = zeros((num_spin_orbitals, num_spin_orbitals, num_spin_orbitals, num_spin_orbitals))
    for dict_item in H_to_use.terms.items()
        term, val = dict_item
        val = pyconvert(Float64, val)
        term = pyconvert(Union{NTuple{4,Tuple{Int64,Int64}},NTuple{2,Tuple{Int64,Int64}}}, term)
        if length(term) == 4
            two_body_tensor_spin_orb[term[1][1]+1, term[2][1]+1, term[3][1]+1, term[4][1]+1] = val
        end
    end

    # #Get the energies using the Lanczos method from module_sdstate
    tensors = (one_body_tensor_chemist, two_body_tensor_spin_orb)

    E_max_sdstate_lanczos, E_min_sdstate_lanczos = sdstate_lanczos.lanczos_range(Hf=tensors, steps=30, state=nothing, ne=num_electrons)
    # E_max_sdstate_lanczos, E_min_sdstate_lanczos = QuantumMAMBO.lanczos_range(
    #     one_body_tensor=one_body_tensor_chemist,
    #     two_body_tensor=two_body_tensor_spin_orb,
    #     num_electrons=num_electrons,
    #     initial_state=nothing,
    #     steps=30
    # )
    E_max_lanczo_final = pyconvert(Float64, E_max_sdstate_lanczos) + nuc_rep_energy
    E_min_lanczo_final = pyconvert(Float64, E_min_sdstate_lanczos) + nuc_rep_energy

    #Get the energies using the total Lanczos method from module_sdstate
    E_max_sdstate_lanczos_total, E_min_sdstate_lanczos_total = sdstate_lanczos.lanczos_total_range(Hf=tensors, steps=30, states=[], e_nums=[num_electrons], multiprocessing=false)
    # E_max_sdstate_lanczos_total, E_min_sdstate_lanczos_total = QuantumMAMBO.lanczos_total_range(one_body_tensor=one_body_tensor_chemist,
    #     two_body_tensor=two_body_tensor_spin_orb,
    #     initial_states=[],
    #     e_nums=[],
    #     steps=30,
    #     multiprocessing=false
    # )

    E_max_lanczo_final_total = pyconvert(Float64, E_max_sdstate_lanczos_total) + nuc_rep_energy
    E_min_lanczo_final_total = pyconvert(Float64, E_min_sdstate_lanczos_total) + nuc_rep_energy

    return E_FCI_HF, E_FCI_UHF, E_FCI_orb, GS_orig_FCI_eig_val_final, E_min_lanczo_final, E_min_lanczo_final_total, max_orig_FCI_eig_val_final, E_max_lanczo_final, E_max_lanczo_final_total
end

@testset "lanczos_fci_lih" begin
    basis = "sto3g"
    bond_length = 1.0
    geometry = "H 0 0 0; Li 0 0 $bond_length"

    E_FCI_HF, E_FCI_UHF, E_FCI_orb, GS_orig_FCI_eig_val_final, E_min_lanczo_final, E_min_lanczo_final_total, max_orig_FCI_eig_val_final, E_max_lanczo_final, E_max_lanczo_final_total = calc_lanczos_test_energies(basis=basis, geometry=geometry)

    # Check that the ground state energies are the same
    @test isapprox(E_FCI_HF, E_FCI_UHF, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, E_FCI_orb, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, GS_orig_FCI_eig_val_final, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, E_min_lanczo_final, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, E_min_lanczo_final_total, atol=atol_lanczos_test, rtol=rtol_lanczos_test)

    # Check that the highest state energies are the same
    @test isapprox(max_orig_FCI_eig_val_final, E_max_lanczo_final, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(max_orig_FCI_eig_val_final, E_max_lanczo_final_total, atol=atol_lanczos_test, rtol=rtol_lanczos_test) broken = true



end