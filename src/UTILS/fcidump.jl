# pyscf_tools_fcidump = pyimport("pyscf.tools.fcidump") ############NOT HERE###############

function load_tensors_from_fcidump(; data_file_path, molpro_orbsym_convention=true)
    """
    Load tensors from a FCIDUMP file.
    FCIDUMP files from Chemistry Benchmark appear to have the molpro orbital symmetries convention.
    """
    println("Loading tensors from FCIDUMP file: ", data_file_path)
    # readline()
    println("fci_data to be loaded")
    # readline()
    pyscf_tools_fcidump = pyimport("pyscf.tools.fcidump") ############PUT HERE###################
    pyscf= pyimport("pyscf") ############PUT HERE###################
    fci_data = pyscf.tools.fcidump.read(
        data_file_path, molpro_orbsym_convention
    )
    println("fci_data loaded")

    # dict_keys(['NORB', 'NELEC', 'MS2', 'ORBSYM', 'ISYM', 'ECORE', 'H1', 'H2'])
    # pyconvert(Float64, cisolver.kernel()[0])

    num_orbitals = pyconvert(Int64, fci_data["NORB"])
    num_spin_orbitals = 2 * num_orbitals
    num_electrons = pyconvert(Int64, fci_data["NELEC"])
    two_S = pyconvert(Int64, fci_data["MS2"])
    two_Sz = pyconvert(Int64, fci_data["MS2"])
    orb_sym = pyconvert(Array{Int64}, fci_data["ORBSYM"])
    nuc_rep_energy = pyconvert(Float64, fci_data["ECORE"])
    one_body_tensor = pyconvert(Array{Float64},fci_data["H1"])
    two_body_tensor_symmetrized = fci_data["H2"]
    
    
    two_body_tensor = pyconvert(Array{Float64}, pyscf.ao2mo.restore(
        "s1", two_body_tensor_symmetrized, num_orbitals
    ))

    extra_attributes = Dict("ISYM" => pyconvert(Int64,fci_data["ISYM"]))
    return (
        one_body_tensor,
        two_body_tensor,
        nuc_rep_energy,
        num_orbitals,
        num_spin_orbitals,
        num_electrons,
        two_S,
        two_Sz,
        orb_sym,
        extra_attributes,
    )
end 