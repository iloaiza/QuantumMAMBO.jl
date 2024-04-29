using PythonCall

function list_of_2sz_values(num_electrons,num_orbitals)
    # Calculate the possible values of the Sz quantum number for the given number of electrons and orbitals
    # Only positive values are returned, assuming that the negative values are degenerate.
    println("num_electrons: ", num_electrons)
    println("num_orbitals: ", num_orbitals)
    if num_electrons <= num_orbitals
        if num_electrons % 2 == 0
            println("collect(0:2:num_electrons): ", collect(0:2:num_electrons))
            return collect(0:2:num_electrons)
        else
            println("collect(1:2:num_electrons): ", collect(1:2:num_electrons))
            return collect(1:2:num_electrons)
        end
    else
        if num_electrons % 2 == 0
            println("collect(0:2:(2*num_orbitals-num_electrons)): ", collect(0:2:(2*num_orbitals-num_electrons)))
            return collect(0:2:(2*num_orbitals-num_electrons))
        else
            println("collect(1:2:(2*num_orbitals-num_electrons)): ", collect(1:2:(2*num_orbitals-num_electrons)))
            return collect(1:2:(2*num_orbitals-num_electrons))
        end
    end
        
end

function pyscf_full_ci(one_body_tensor::Matrix{Float64},
                        two_body_tensor::Array{Float64, 4},
                        ecore::Float64,
                        num_electrons::Int64,
                        max_cycle::Int64,
                        conv_tol::Float64)
    #Hamiltonian form assumed to be:
	# H = E_0 + h_ij a†_i a_j + 0.5*g_ijkl a†_i a†_k a_l a_j
    # with no permutation symmetry compression for two_body_tensor
    # Will probably want a convergence tolerance (conv_tol) around chemical accuracy (1e-3), 
    # or perhaps looser if the calculations are too slow.
    

    # Check real orbital permutation symmetries as direct_spin1.FCI() assumes these symmetries
    perm_sym = pyimport("permutation_symmetries")
    two_body_tensor_python = Py(two_body_tensor).to_numpy()
    one_body_tensor_python = Py(one_body_tensor).to_numpy()
    # four_fold_symmetry_bool = perm_sym.check_permutation_symmetries_complex_orbitals(one_body_tensor_python, two_body_tensor_python)
    eight_fold_symmetry_bool = perm_sym.check_permutation_symmetries_real_orbitals(one_body_tensor_python, two_body_tensor_python)
    
    # four_fold_symmetry_bool = pyconvert(Bool, four_fold_symmetry_bool)
    eight_fold_symmetry_bool = pyconvert(Bool, eight_fold_symmetry_bool)

    if !eight_fold_symmetry_bool
        throw(ErrorException("8-fold symmetry not satisfied."))
    end

    num_orbitals = size(one_body_tensor)[1]
    num_electrons_list = collect(0:2*num_orbitals)
    E_min_FS = Inf # min over whole Fock space
    E_max_FS = -Inf # max over whole Fock space
    E_min_ENS = Inf # min over electron number subspace
    E_max_ENS = -Inf # max over electron number subspace
    pyscf_fci = pyimport("pyscf.fci")

    println("num_electrons_list: ", num_electrons_list)
    for loop_num_electrons in num_electrons_list
        println("num_electrons: ", loop_num_electrons)
        println("num_orbitals: ", num_orbitals)
        println("---------")
        if loop_num_electrons == 0
            println("Skipping for zero electrons.")
            e_min = ecore
            e_max = ecore
            if e_min < E_min_FS
                E_min_FS = e_min
            end
            if e_max > E_max_FS
                E_max_FS = e_max
            end
            continue
        end
        println("Entering loop for non-zero electrons.")

        # Also loop over the possible multiplicities by looping over the possible values of the Sz quantum number
        two_sz_vals = list_of_2sz_values(loop_num_electrons,num_orbitals)
        println("two_sz_vals: ", two_sz_vals)
        for two_sz_val in two_sz_vals
            num_alpha_electrons = (loop_num_electrons + two_sz_val) ÷ 2
            num_beta_electrons = loop_num_electrons - num_alpha_electrons
            if (num_alpha_electrons + num_beta_electrons) != loop_num_electrons
                throw(ErrorException("num_alpha_electrons + num_beta_electrons != loop_num_electrons"))
            end
            # Below based on example from https://github.com/pyscf/pyscf/blob/master/examples/fci/01-given_h1e_h2e.py
        
            cisolver = pyscf_fci.direct_spin1.FCI()
            cisolver.max_cycle = max_cycle
            cisolver.conv_tol = conv_tol
            println("num_electrons: ", loop_num_electrons)
            println("num_orbitals: ", num_orbitals)
            println("num_alpha_electrons: ", num_alpha_electrons)
            println("num_beta_electrons: ", num_beta_electrons)
            
            println("two_sz_val: ", two_sz_val)

            e_min, fcivec = cisolver.kernel(one_body_tensor_python, two_body_tensor_python, num_orbitals, (num_alpha_electrons,num_beta_electrons), ecore=ecore)
            e_min = pyconvert(Float64, e_min)
            println("e_min: ", e_min)
            neg_e_max, fcivec = cisolver.kernel(-1*one_body_tensor_python, -1*two_body_tensor_python, num_orbitals, (num_alpha_electrons,num_beta_electrons), ecore=-1*ecore)
            neg_e_max = pyconvert(Float64, neg_e_max)
            e_max = -1*neg_e_max
            println("e_max: ", e_max)
            if e_min < E_min_FS
                E_min_FS = e_min
            end
            if e_max > E_max_FS
                E_max_FS = e_max
            end

            if (loop_num_electrons == num_electrons) #&& (two_sz_val == two_sz)
                if e_min < E_min_ENS
                    E_min_ENS = e_min
                end
                if e_max > E_max_ENS
                    E_max_ENS = e_max
                end
                # E_min_ENS = e_min
                # E_max_ENS = e_max
                println("For num_electrons: ", loop_num_electrons, " and two_sz_val: ", two_sz_val, " now:")
                println("E_min_ENS: ", E_min_ENS)
                println("E_max_ENS: ", E_max_ENS)
            end
        
        end
        
    end
    return E_min_FS, E_max_FS, E_min_ENS, E_max_ENS

end