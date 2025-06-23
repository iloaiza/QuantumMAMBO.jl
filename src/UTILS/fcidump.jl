# #FUNCTIONS FOR INTERFACING WITH PYTHON
# ENV["JULIA_CONDAPKG_BACKEND"] = PY_BACKEND

# using PythonCall
# np = pyimport("numpy")
# scipy = pyimport("scipy")
# sympy = pyimport("sympy")
# of = pyimport("openfermion")

# UTILS_DIR = @__DIR__
# sys = pyimport("sys")
# sys.path.append(UTILS_DIR)


function load_tensors_from_fcidump(; data_file_path, molpro_orbsym_convention=true)
    """
    Load tensors from a FCIDUMP file.
    FCIDUMP files from Chemistry Benchmark appear to have the molpro orbital symmetries convention.
    """
    println("Loading tensors from FCIDUMP file: ", data_file_path)

    println("fci_data to be loaded")

    pyscf_tools_fcidump = pyimport("pyscf.tools.fcidump") 
    fci_data = pyscf_tools_fcidump.read(
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
    
    pyscf_ao2mo = pyimport("pyscf.ao2mo")
    two_body_tensor = pyconvert(Array{Float64}, pyscf_ao2mo.restore(
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


function compress_tensors(one_body_tensor, two_body_tensor, num_orbitals )
    # #FUNCTIONS FOR INTERFACING WITH PYTHON
    # ENV["JULIA_CONDAPKG_BACKEND"] = PY_BACKEND

    # # using PythonCall
    # UTILS_DIR = @__DIR__
    # sys = pyimport("sys")
    # sys.path.append(UTILS_DIR)
    perm_sym = pyimport("permutation_symmetries" )
    two_body_tensor_python = Py(two_body_tensor).to_numpy()
    four_fold_symmetry_bool = perm_sym.check_permutation_symmetries_complex_orbitals(Py(one_body_tensor).to_numpy(), two_body_tensor_python)
    eight_fold_symmetry_bool = perm_sym.check_permutation_symmetries_real_orbitals(Py(one_body_tensor).to_numpy(), two_body_tensor_python)
    
    four_fold_symmetry_bool = pyconvert(Bool, four_fold_symmetry_bool)
    eight_fold_symmetry_bool = pyconvert(Bool, eight_fold_symmetry_bool)
    
    pyscf_ao2mo = pyimport("pyscf.ao2mo")
    if eight_fold_symmetry_bool
        println("Eight-fold permutation symmetry detected.")
        two_body_tensor_bliss_compressed = pyscf_ao2mo.restore("s8", two_body_tensor_python, num_orbitals)
    elseif four_fold_symmetry_bool
        println("Four-fold permutation symmetry detected.")
        two_body_tensor_bliss_compressed = pyscf_ao2mo.restore("s4", two_body_tensor_python, num_orbitals)
    else
        throw(ArgumentError("No permutation symmetry detected. At least four-fold symmetry is required."))
    end

    return two_body_tensor_bliss_compressed
end