using Test
using QuantumMAMBO
using PythonCall
pyscf = pyimport("pyscf")
feru = pyimport("ferm_utils")
open_ferm = pyimport("openfermion")
sp_linalg = pyimport("scipy.sparse.linalg")
sp_orig_linalg = pyimport("scipy.linalg")
sdstate_lanczos = pyimport("module_sdstate.lanczos_utils")
numpy = pyimport("numpy")

atol_lanczos_test = 1E-3#1E-6
rtol_lanczos_test = 1E-3#1E-6
# TEMPFOLDER = "test/tmp"

function calc_energy_range_all_num_electrons(; H_to_use, num_spin_orbitals, num_electrons_list, nuc_rep_energy, spin_preserving=false)
    max_energy_list = []
    min_energy_list = []

    for num_electrons in num_electrons_list
        # Get sparse matrix from openfermion
        H_orig_sparseMat = open_ferm.get_number_preserving_sparse_operator(
            fermion_op=open_ferm.normal_ordered(H_to_use),
            num_qubits=num_spin_orbitals,
            num_electrons=num_electrons,
            spin_preserving=spin_preserving,
            reference_determinant=nothing,
            excitation_level=nothing,
        )

        # if num_electrons != num_spin_orbitals

        GS_orig_FCI_eig_val_final = nothing
        max_orig_FCI_eig_val_final = nothing
        try
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
                which="LR",  # Largest real part eigenvalues.
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

        catch error_val
            println("Switching to diagonalization of dense matrix.")
            println("Matrix shape: ", H_orig_sparseMat.shape)
            @time eigvals = sp_orig_linalg.eigvals(H_orig_sparseMat.toarray())
            # println("Error: ", error_val)
            # println("num_electrons: ", num_electrons)
            # println("num_spin_orbitals: ", num_spin_orbitals)
            # println("matrix: ", H_orig_sparseMat)
            # println("matrix: ", H_orig_sparseMat.shape)

            # GS_orig_FCI_eig_val_final = Inf
            # max_orig_FCI_eig_val_final = Inf
            eigvals = real(pyconvert(Array{Complex{Float64}}, eigvals)) .+ nuc_rep_energy
            # println("eigvals: ", eigvals)
            energy = minimum(eigvals)

            GS_orig_FCI_eig_val_final = energy
            energy = maximum(eigvals)
            max_orig_FCI_eig_val_final = energy
        end

        # else

        # end

        max_energy_list = push!(max_energy_list, max_orig_FCI_eig_val_final)
        min_energy_list = push!(min_energy_list, GS_orig_FCI_eig_val_final)
    end
    max_energy = maximum(max_energy_list)
    min_energy = minimum(min_energy_list)

    return max_energy, min_energy, max_energy_list, min_energy_list
end

function calc_lanczos_test_energies(; basis::String, geometry::String, spin::Int64=0, charge::Int64=0, multiplicity::Int64=1)
    println("----------------------------------------------------------------")
    println("basis: ", basis)
    println("geometry: ", geometry)
    println("spin: ", spin)
    println("charge: ", charge)
    println("multiplicity: ", multiplicity)

    mol = pyscf.gto.Mole()
    mol.basis = basis
    mol.atom = geometry
    mol.spin = spin
    mol.charge = charge
    mol.multiplicity = multiplicity
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

    println("E_FCI_HF: ", E_FCI_HF)
    println("E_FCI_UHF: ", E_FCI_UHF)
    println("E_FCI_orb: ", E_FCI_orb)

    # Get metadata
    num_electrons = pyconvert(Int64, mol.nelectron)
    num_orb = pyconvert(Int64, mol.nao_nr())

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
    # println("H_to_use: ", H_to_use)
    # println("----------------------------------------------------------------")
    num_spin_orbitals = 2 * size(one_body_tensor)[1]

    println("num_spin_orbitals: ", num_spin_orbitals)
    

    # Get one body tensor for the two body term being in chemist notation
    one_body_tensor_chemist = pyconvert(Array{Float64,2}, feru.get_obt(H=one_body_fermi_op, n=2 * num_orb, spin_orb=false))
    two_body_tensor_chemist = pyconvert(Array{Float64,4}, 0.5*two_body_tensor)
    QuantumMAMBO.eliminate_small_values!(one_body_tensor_chemist, 1e-8)
    QuantumMAMBO.eliminate_small_values!(two_body_tensor_chemist, 1e-8)
    println("one_body_tensor type: ", typeof(one_body_tensor))
    println("two_body_tensor type: ", typeof(two_body_tensor))
    println("one_body_tensor shape: ", size(one_body_tensor))
    println("two_body_tensor shape: ", two_body_tensor.shape)
    
    println(" one_body_tensor_chemist type: ", typeof(one_body_tensor_chemist))
    println(" two_body_tensor_chemist type: ", typeof(two_body_tensor_chemist))
    println("one_body_tensor_chemist shape: ", size(one_body_tensor_chemist))
    println("two_body_tensor_chemist shape: ", size(two_body_tensor_chemist))

    # println("one_body_tensor_chemist: ", one_body_tensor_chemist)
    # println("two_body_tensor_chemist: ", two_body_tensor_chemist)
    # println("one_body_tensor: ", one_body_tensor)
    # println("two_body_tensor: ", two_body_tensor)
    

    # Get the FCI energy based on Lanczos from Scipy
    # For both the ground state and the largest energy
    ##################
    num_electrons_list=collect(range(0, stop=num_spin_orbitals, step=1))
    # num_electrons_list = Array([0, num_electrons, num_spin_orbitals])
    @time max_scipy_FCI_energy_all_electrons, min_scipy_FCI_energy_all_electrons, max_energy_list, min_energy_list = calc_energy_range_all_num_electrons(
        H_to_use=H_to_use,
        num_spin_orbitals=num_spin_orbitals,
        num_electrons_list=num_electrons_list,
        nuc_rep_energy=nuc_rep_energy,
        spin_preserving=false)
    println("max_energy_list: ", max_energy_list)
    println("min_energy_list: ", min_energy_list)
    println("num_electrons_list: ", num_electrons_list)
    println("num_electrons: ", num_electrons)

    neutral_charge_max_scipy_FCI_energy = max_energy_list[num_electrons+1]
    println("neutral_charge_max_scipy_FCI_energy: ", neutral_charge_max_scipy_FCI_energy)
    # neutral_charge_max_scipy_FCI_energy = max_energy_list[2]
    # println("max_energy_list: ", max_energy_list)
    # println("min_energy_list: ", min_energy_list)
    # println("num_electrons_list: ", num_electrons_list)

    # Get the energies as calculated by module_sdstate
    ##################

    # Get the two body tensor in spin orbital form from the openfermion operator
    # two_body_tensor_spin_orb = zeros((num_spin_orbitals, num_spin_orbitals, num_spin_orbitals, num_spin_orbitals))
    # for dict_item in H_to_use.terms.items()
    #     term, val = dict_item
    #     val = pyconvert(Float64, val)
    #     term = pyconvert(Union{NTuple{4,Tuple{Int64,Int64}},NTuple{2,Tuple{Int64,Int64}}}, term)
    #     if length(term) == 4
    #         two_body_tensor_spin_orb[term[1][1]+1, term[2][1]+1, term[3][1]+1, term[4][1]+1] = val
    #     end
    # end

    # #Get the energies using the Lanczos method from module_sdstate
    E_max_lanczos_final, E_min_lanczos_final = QuantumMAMBO.lanczos_range(one_body_tensor=one_body_tensor_chemist, 
                                                                        two_body_tensor=two_body_tensor_chemist, 
                                                                        core_energy=nuc_rep_energy, 
                                                                        num_electrons=num_electrons, 
                                                                        initial_state=nothing, 
                                                                        steps=25,
                                                                        spin_orbitals=false
                                                                        )
                                                                        
    # tensors = (one_body_tensor_chemist, two_body_tensor_spin_orb)
    # E_max_sdstate_lanczos, E_min_sdstate_lanczos = sdstate_lanczos.lanczos_range(Hf=tensors, steps=25, state=nothing, ne=num_electrons)
    
    # E_max_sdstate_lanczos_temp, E_min_sdstate_lanczos_temp = sdstate_lanczos.lanczos_range(Hf=tensors, steps=25, state=nothing, ne=0)
    # E_max_sdstate_lanczos_temp2, E_min_sdstate_lanczos_temp2 = sdstate_lanczos.lanczos_range(Hf=tensors, steps=25, state=nothing, ne=num_spin_orbitals)
    # E_max_sdstate_lanczos_total = maximum([pyconvert(Float64, E_max_sdstate_lanczos_temp), pyconvert(Float64, E_max_sdstate_lanczos_temp2)])
    # E_min_sdstate_lanczos_total = minimum([pyconvert(Float64, E_min_sdstate_lanczos_temp), pyconvert(Float64, E_min_sdstate_lanczos_temp2), pyconvert(Float64, E_min_sdstate_lanczos)])
    
    # Below wrapper seg faults for unknown reason
    # E_max_sdstate_lanczos, E_min_sdstate_lanczos = QuantumMAMBO.lanczos_range(
    #     one_body_tensor=one_body_tensor_chemist,
    #     two_body_tensor=two_body_tensor_spin_orb,
    #     num_electrons=num_electrons,
    #     initial_state=nothing,
    #     steps=30
    # )
    # E_max_lanczo_final = pyconvert(Float64, E_max_sdstate_lanczos) + nuc_rep_energy
    # E_min_lanczo_final = pyconvert(Float64, E_min_sdstate_lanczos) + nuc_rep_energy

    println("E_max_lanczo_final: ", E_max_lanczos_final)
    println("E_min_lanczo_final: ", E_min_lanczos_final)

    #Get the energies using the total Lanczos method from module_sdstate
    E_max_lanczo_final_total, E_min_lanczo_final_total = QuantumMAMBO.lanczos_total_range(one_body_tensor=one_body_tensor_chemist, 
                                                                                        two_body_tensor=two_body_tensor_chemist, 
                                                                                        core_energy=nuc_rep_energy, 
                                                                                        initial_states=[],
                                                                                        num_electrons_list=[], 
                                                                                        steps=25,
                                                                                        multiprocessing=false,
                                                                                        spin_orbitals=false
                                                                                        )
                                                                                        
    
    # E_max_sdstate_lanczos_total, E_min_sdstate_lanczos_total = sdstate_lanczos.lanczos_total_range(Hf=tensors, steps=25, states=[], e_nums=[], multiprocessing=false)
    
    # Below wrapper seg faults for unknown reason
    # E_max_sdstate_lanczos_total, E_min_sdstate_lanczos_total = QuantumMAMBO.lanczos_total_range(one_body_tensor=one_body_tensor_chemist,
    #     two_body_tensor=two_body_tensor_spin_orb,
    #     initial_states=[],
    #     e_nums=[],
    #     steps=30,
    #     multiprocessing=false
    # )

    # E_max_lanczo_final_total = pyconvert(Float64, E_max_sdstate_lanczos_total) + nuc_rep_energy
    # E_min_lanczo_final_total = pyconvert(Float64, E_min_sdstate_lanczos_total) + nuc_rep_energy
    # E_max_lanczo_final_total = E_max_sdstate_lanczos_total + nuc_rep_energy
    # E_min_lanczo_final_total = E_min_sdstate_lanczos_total + nuc_rep_energy
    println("E_max_lanczo_final_total: ", E_max_lanczo_final_total)
    println("E_min_lanczo_final_total: ", E_min_lanczo_final_total)

    return (E_FCI_HF, 
        E_FCI_UHF, 
        E_FCI_orb, 
        min_scipy_FCI_energy_all_electrons, 
        E_min_lanczos_final, 
        E_min_lanczo_final_total, 
        max_scipy_FCI_energy_all_electrons, 
        E_max_lanczos_final, 
        E_max_lanczo_final_total, 
        neutral_charge_max_scipy_FCI_energy
        )
end

@testset "lanczos_fci_h2" begin
    basis = "sto3g" # 4 spin orbitals
    bond_length = 1.0
    geometry = "H 0 0 0; H 0 0 $bond_length"

    E_FCI_HF, E_FCI_UHF, E_FCI_orb, min_scipy_FCI_energy_all_electrons, E_min_lanczo_final, E_min_lanczo_final_total, max_scipy_FCI_energy_all_electrons, E_max_lanczo_final, E_max_lanczo_final_total, neutral_charge_max_scipy_FCI_energy = calc_lanczos_test_energies(basis=basis, geometry=geometry)

    # Check that the ground state energies are the same
    @test isapprox(E_FCI_HF, E_FCI_UHF, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, E_FCI_orb, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, min_scipy_FCI_energy_all_electrons, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, E_min_lanczo_final, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, E_min_lanczo_final_total, atol=atol_lanczos_test, rtol=rtol_lanczos_test)

    # Check that the highest state energies are the same
    @test isapprox(neutral_charge_max_scipy_FCI_energy, E_max_lanczo_final, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(max_scipy_FCI_energy_all_electrons, E_max_lanczo_final_total, atol=atol_lanczos_test, rtol=rtol_lanczos_test)

end

@testset "lanczos_fci_h2_sto6g" begin
    basis = "sto6g" # 4 spin orbitals
    bond_length = 1.0
    geometry = "H 0 0 0; H 0 0 $bond_length"

    E_FCI_HF, E_FCI_UHF, E_FCI_orb, min_scipy_FCI_energy_all_electrons, E_min_lanczo_final, E_min_lanczo_final_total, max_scipy_FCI_energy_all_electrons, E_max_lanczo_final, E_max_lanczo_final_total, neutral_charge_max_scipy_FCI_energy = calc_lanczos_test_energies(basis=basis, geometry=geometry)

    # Check that the ground state energies are the same
    @test isapprox(E_FCI_HF, E_FCI_UHF, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, E_FCI_orb, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, min_scipy_FCI_energy_all_electrons, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, E_min_lanczo_final, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, E_min_lanczo_final_total, atol=atol_lanczos_test, rtol=rtol_lanczos_test)

    # Check that the highest state energies are the same
    @test isapprox(neutral_charge_max_scipy_FCI_energy, E_max_lanczo_final, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(max_scipy_FCI_energy_all_electrons, E_max_lanczo_final_total, atol=atol_lanczos_test, rtol=rtol_lanczos_test)

end

@testset "lanczos_fci_h2neg_sto6g" begin
    basis = "sto6g" # 4 spin orbitals
    bond_length = 1.0
    geometry = "H 0 0 0; H 0 0 $bond_length"
    charge = -1
    spin = 1
    multiplicity = 2

    (E_FCI_HF, 
    E_FCI_UHF, 
    E_FCI_orb, 
    min_scipy_FCI_energy_all_electrons, 
    E_min_lanczo_final, 
    E_min_lanczo_final_total, 
    max_scipy_FCI_energy_all_electrons, 
    E_max_lanczo_final, 
    E_max_lanczo_final_total, 
    neutral_charge_max_scipy_FCI_energy) = calc_lanczos_test_energies(basis=basis, 
                                                                        geometry=geometry, 
                                                                        spin=spin, 
                                                                        charge=charge, 
                                                                        multiplicity=multiplicity)

    # Check that the ground state energies are the same
    @test isapprox(E_FCI_HF, E_FCI_UHF, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, E_FCI_orb, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    # @test isapprox(E_min_lanczo_final, min_scipy_FCI_energy_all_electrons, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, E_min_lanczo_final, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(min_scipy_FCI_energy_all_electrons, E_min_lanczo_final_total, atol=atol_lanczos_test, rtol=rtol_lanczos_test)

    # Check that the highest state energies are the same
    @test isapprox(neutral_charge_max_scipy_FCI_energy, E_max_lanczo_final, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(max_scipy_FCI_energy_all_electrons, E_max_lanczo_final_total, atol=atol_lanczos_test, rtol=rtol_lanczos_test)

end

# @testset "lanczos_fci_h2_cc-pVDZ" begin
#     basis = "cc-pVDZ" # 20 spin orbitals
#     bond_length = 1.0
#     geometry = "H 0 0 0; H 0 0 $bond_length"

#     E_FCI_HF, E_FCI_UHF, E_FCI_orb, min_scipy_FCI_energy_all_electrons, E_min_lanczo_final, E_min_lanczo_final_total, max_scipy_FCI_energy_all_electrons, E_max_lanczo_final, E_max_lanczo_final_total, neutral_charge_max_scipy_FCI_energy = calc_lanczos_test_energies(basis=basis, geometry=geometry)

#     # Check that the ground state energies are the same
#     @test isapprox(E_FCI_HF, E_FCI_UHF, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
#     @test isapprox(E_FCI_HF, E_FCI_orb, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
#     @test isapprox(E_FCI_HF, min_scipy_FCI_energy_all_electrons, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
#     @test isapprox(E_FCI_HF, E_min_lanczo_final, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
#     @test isapprox(E_FCI_HF, E_min_lanczo_final_total, atol=atol_lanczos_test, rtol=rtol_lanczos_test)

#     # Check that the highest state energies are the same
#     @test isapprox(neutral_charge_max_scipy_FCI_energy, E_max_lanczo_final, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
#     @test isapprox(max_scipy_FCI_energy_all_electrons, E_max_lanczo_final_total, atol=atol_lanczos_test, rtol=rtol_lanczos_test)

# end

# @testset "lanczos_fci_hneg_aug-cc-pVDZ" begin
#     basis = "aug-cc-pVDZ" # 18 spin orbitals
#     # bond_length = 1.0
#     geometry = "H 0 0 0"
#     charge = -1

#     (E_FCI_HF,
#         E_FCI_UHF,
#         E_FCI_orb,
#         min_scipy_FCI_energy_all_electrons,
#         E_min_lanczo_final,
#         E_min_lanczo_final_total,
#         max_scipy_FCI_energy_all_electrons,
#         E_max_lanczo_final,
#         E_max_lanczo_final_total,
#         neutral_charge_max_scipy_FCI_energy) = calc_lanczos_test_energies(basis=basis,
#         geometry=geometry,
#         spin=0,
#         charge=charge,
#         multiplicity=1)

#     # Check that the ground state energies are the same
#     @test isapprox(E_FCI_HF, E_FCI_UHF, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
#     @test isapprox(E_FCI_HF, E_FCI_orb, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
#     @test isapprox(E_FCI_HF, min_scipy_FCI_energy_all_electrons, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
#     @test isapprox(E_FCI_HF, E_min_lanczo_final, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
#     @test isapprox(E_FCI_HF, E_min_lanczo_final_total, atol=atol_lanczos_test, rtol=rtol_lanczos_test)

#     # Check that the highest state energies are the same
#     @test isapprox(neutral_charge_max_scipy_FCI_energy, E_max_lanczo_final, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
#     @test isapprox(max_scipy_FCI_energy_all_electrons, E_max_lanczo_final_total, atol=atol_lanczos_test, rtol=rtol_lanczos_test)

# end

@testset "lanczos_fci_hneg_cc-pVDZ" begin
    basis = "cc-pVDZ" #  spin orbitals
    # bond_length = 1.0
    geometry = "H 0 0 0"
    charge = -1

    (E_FCI_HF,
        E_FCI_UHF,
        E_FCI_orb,
        min_scipy_FCI_energy_all_electrons,
        E_min_lanczo_final,
        E_min_lanczo_final_total,
        max_scipy_FCI_energy_all_electrons,
        E_max_lanczo_final,
        E_max_lanczo_final_total,
        neutral_charge_max_scipy_FCI_energy) = calc_lanczos_test_energies(basis=basis,
        geometry=geometry,
        spin=0,
        charge=charge,
        multiplicity=1)

    # Check that the ground state energies are the same
    @test isapprox(E_FCI_HF, E_FCI_UHF, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, E_FCI_orb, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    # @test isapprox(E_min_lanczo_final, min_scipy_FCI_energy_all_electrons, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, E_min_lanczo_final, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(min_scipy_FCI_energy_all_electrons, E_min_lanczo_final_total, atol=atol_lanczos_test, rtol=rtol_lanczos_test)

    # Check that the highest state energies are the same
    @test isapprox(neutral_charge_max_scipy_FCI_energy, E_max_lanczo_final, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(max_scipy_FCI_energy_all_electrons, E_max_lanczo_final_total, atol=atol_lanczos_test, rtol=rtol_lanczos_test)

end

# @testset "lanczos_fci_lih_cc-pVDZ" begin
#     basis = "cc-pVDZ" # 38 spin orbitals
#     bond_length = 1.0
#     geometry = "H 0 0 0; Li 0 0 $bond_length"

#     E_FCI_HF, E_FCI_UHF, E_FCI_orb, min_scipy_FCI_energy_all_electrons, E_min_lanczo_final, E_min_lanczo_final_total, max_scipy_FCI_energy_all_electrons, E_max_lanczo_final, E_max_lanczo_final_total, neutral_charge_max_scipy_FCI_energy = calc_lanczos_test_energies(basis=basis, geometry=geometry)

#     # Check that the ground state energies are the same
#     @test isapprox(E_FCI_HF, E_FCI_UHF, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
#     @test isapprox(E_FCI_HF, E_FCI_orb, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
#     @test isapprox(E_FCI_HF, min_scipy_FCI_energy_all_electrons, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
#     @test isapprox(E_FCI_HF, E_min_lanczo_final, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
#     @test isapprox(E_FCI_HF, E_min_lanczo_final_total, atol=atol_lanczos_test, rtol=rtol_lanczos_test)

#     # Check that the highest state energies are the same
#     @test isapprox(neutral_charge_max_scipy_FCI_energy, E_max_lanczo_final, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
#     @test isapprox(max_scipy_FCI_energy_all_electrons, E_max_lanczo_final_total, atol=atol_lanczos_test, rtol=rtol_lanczos_test)

# end

@testset "lanczos_fci_lih" begin
    basis = "sto3g"
    bond_length = 1.0
    geometry = "H 0 0 0; Li 0 0 $bond_length"

    E_FCI_HF, E_FCI_UHF, E_FCI_orb, min_scipy_FCI_energy_all_electrons, E_min_lanczo_final, E_min_lanczo_final_total, max_scipy_FCI_energy_all_electrons, E_max_lanczo_final, E_max_lanczo_final_total, neutral_charge_max_scipy_FCI_energy = calc_lanczos_test_energies(basis=basis, geometry=geometry)

    # Check that the ground state energies are the same
    @test isapprox(E_FCI_HF, E_FCI_UHF, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, E_FCI_orb, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, min_scipy_FCI_energy_all_electrons, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, E_min_lanczo_final, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, E_min_lanczo_final_total, atol=atol_lanczos_test, rtol=rtol_lanczos_test)

    # Check that the highest state energies are the same
    @test isapprox(neutral_charge_max_scipy_FCI_energy, E_max_lanczo_final, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(max_scipy_FCI_energy_all_electrons, E_max_lanczo_final_total, atol=atol_lanczos_test, rtol=rtol_lanczos_test)



end

@testset "lanczos_fci_beh2" begin
    basis = "sto3g"
    bond_length = 1.0
    geometry = "Be 0 0 0; H 0 0 $bond_length; H 0 0 -$bond_length"

    E_FCI_HF, E_FCI_UHF, E_FCI_orb, min_scipy_FCI_energy_all_electrons, E_min_lanczo_final, E_min_lanczo_final_total, max_scipy_FCI_energy_all_electrons, E_max_lanczo_final, E_max_lanczo_final_total, neutral_charge_max_scipy_FCI_energy = calc_lanczos_test_energies(basis=basis, geometry=geometry)

    # Check that the ground state energies are the same
    @test isapprox(E_FCI_HF, E_FCI_UHF, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, E_FCI_orb, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, min_scipy_FCI_energy_all_electrons, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, E_min_lanczo_final, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, E_min_lanczo_final_total, atol=atol_lanczos_test, rtol=rtol_lanczos_test)

    # Check that the highest state energies are the same
    @test isapprox(neutral_charge_max_scipy_FCI_energy, E_max_lanczo_final, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(max_scipy_FCI_energy_all_electrons, E_max_lanczo_final_total, atol=atol_lanczos_test, rtol=rtol_lanczos_test)



end

@testset "lanczos_fci_h2o" begin
    basis = "sto3g"
    bond_length = 1.0

    angle = deg2rad(107.6 / 2)
    xDistance = bond_length * sin(angle)
    yDistance = bond_length * cos(angle)
    geometry = "O 0 0 0; H -$xDistance $yDistance 0; H $xDistance $yDistance 0"

    E_FCI_HF, E_FCI_UHF, E_FCI_orb, min_scipy_FCI_energy_all_electrons, E_min_lanczo_final, E_min_lanczo_final_total, max_scipy_FCI_energy_all_electrons, E_max_lanczo_final, E_max_lanczo_final_total, neutral_charge_max_scipy_FCI_energy = calc_lanczos_test_energies(basis=basis, geometry=geometry)

    # Check that the ground state energies are the same
    @test isapprox(E_FCI_HF, E_FCI_UHF, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, E_FCI_orb, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, min_scipy_FCI_energy_all_electrons, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, E_min_lanczo_final, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(E_FCI_HF, E_min_lanczo_final_total, atol=atol_lanczos_test, rtol=rtol_lanczos_test)

    # Check that the highest state energies are the same
    @test isapprox(neutral_charge_max_scipy_FCI_energy, E_max_lanczo_final, atol=atol_lanczos_test, rtol=rtol_lanczos_test)
    @test isapprox(max_scipy_FCI_energy_all_electrons, E_max_lanczo_final_total, atol=atol_lanczos_test, rtol=rtol_lanczos_test)

end 
