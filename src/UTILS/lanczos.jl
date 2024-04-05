using PythonCall


"""
    lanczos_total_range(one_body_tensor::Array{Float64,2}, two_body_tensor::Array{Float64,4}, states=[], e_nums=[num_electrons], steps::Int=2, multiprocessing::Bool=false)
    Calculate the range of energies across all possible numbers of electrons.
    The tensors are defined such that
    ``H = \\sum_{ij} h_{ij} a_i^† a_j + \\sum_{ijkl} g_ijkl a_i^† a_j a_k^† a_l``
    where the one body tensor is ``h_{ij}`` and the two body tensor is ``g_{ijkl}``.
    Note the absence of the factor of 0.5 in the two body term.

    Args:
        one_body_tensor (Array{Float64,2}): The one body tensor.
        two_body_tensor (Array{Float64,4}): The two body tensor, chemist notation.
        core_energy (Float64): The core energy (aka nuclear repulsion energy)
        initial_states (Array{Any,1}): The initial_states for the Lanczos algorithm.
        num_electrons_list (Array{Int,1}): List of the number of electrons for multiprocessing.
                                            If [], then the range is calculated across all possible 
                                            numbers of electrons: 0 to number of spin orbitals.
        steps (Int): The number of iterations to use in the Lanczos algorithm.
        multiprocessing (Bool): Whether to use multiprocessing. Segfaults if true in unit tests.
        spin_orbitals (Bool): Whether the tensor indices ijkl refer to spin orbitals or not. Default is true.
    Returns:
        E_max_final (Float64): The maximum energy over num_electrons_list.
        E_min_final (Float64): The minimum energy over num_electrons_list.
"""
function lanczos_total_range(; one_body_tensor::Array{Float64,2},
    two_body_tensor::Array{Float64,4},
    core_energy::Float64,
    initial_states=[],
    num_electrons_list=[],
    steps::Int=2,
    multiprocessing::Bool=false,
    spin_orbitals::Bool=true)

    if !spin_orbitals
        two_body_tensor_so  = Py(tbt_orb_to_so(two_body_tensor)).to_numpy()
        one_body_tensor_so = Py(obt_orb_to_so(one_body_tensor)).to_numpy()


    else
        two_body_tensor_so = two_body_tensor
        one_body_tensor_so = one_body_tensor

    end

    sdstate_lanczos = pyimport("module_sdstate.lanczos_utils")
    E_max, E_min = sdstate_lanczos.lanczos_total_range(Hf=(one_body_tensor_so, two_body_tensor_so),
        steps=steps,
        states=initial_states,
        e_nums=num_electrons_list,
        multiprocessing=multiprocessing)

    E_max_final = pyconvert(Float64,E_max)+core_energy
    E_min_final = pyconvert(Float64,E_min)+core_energy

    return E_max_final, E_min_final

end

"""
    lanczos_range(one_body_tensor::Array{Float64,2}, two_body_tensor::Array{Float64,4}, num_electrons::Int, initial_state=nothing, steps::Int=2)
    Calculate the range of energies for a given number of electrons.
    The tensors are defined such that
    ``H = \\sum_{ij} h_{ij} a_i^† a_j + \\sum_{ijkl} g_ijkl a_i^† a_j a_k^† a_l``
    where the one body tensor is ``h_{ij}`` and the two body tensor is ``g_{ijkl}``.
    Note the absence of the factor of 0.5 in the two body term.

    Args:
        one_body_tensor (Array{Float64,2}): The one body tensor.
        two_body_tensor (Array{Float64,4}): The two body tensor, chemist notation.
        num_electrons (Int): The number of electrons.
        initial_state (Any): The initial_state for the Lanczos algorithm.
        steps (Int): The number of iterations to use in the Lanczos algorithm.
        spin_orbitals (Bool): Whether the tensor indices ijkl refer to spin orbitals or not. Default is true.
    Returns:
        E_max_final (Float64): The maximum energy.
        E_min_final (Float64): The minimum energy.
"""
function lanczos_range(; one_body_tensor::Array{Float64,2},
    two_body_tensor::Array{Float64,4},
    core_energy::Float64,
    num_electrons::Int,
    initial_state=nothing,
    steps::Int=2,
    spin_orbitals::Bool=true)

    if !spin_orbitals
        two_body_tensor_so = Py(tbt_orb_to_so(two_body_tensor)).to_numpy()
        one_body_tensor_so = Py(obt_orb_to_so(one_body_tensor)).to_numpy()

    else
        two_body_tensor_so = two_body_tensor
        one_body_tensor_so = one_body_tensor

    end

    sdstate_lanczos = pyimport("module_sdstate.lanczos_utils")
    E_max, E_min = sdstate_lanczos.lanczos_range(Hf=(one_body_tensor_so, two_body_tensor_so),
        steps=steps, state=initial_state, ne=num_electrons)

    E_max_final = pyconvert(Float64,E_max)+core_energy
    E_min_final = pyconvert(Float64,E_min)+core_energy

    return E_max_final, E_min_final

end

"""
    Eliminate small values in a tensor in place.
"""
function eliminate_small_values!(tensor::Array{Float64}, threshold::Float64=1e-8)
    tensor[abs.(tensor) .< threshold] .= 0.0
end