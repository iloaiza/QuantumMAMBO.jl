using PythonCall


"""
    lanczos_total_range(one_body_tensor::Array{Float64,2}, two_body_tensor::Array{Float64,4}, states=[], e_nums=[num_electrons], steps::Int=2, multiprocessing::Bool=false)
    Calculate the range of energies across all possible numbers of electrons.
    The tensors are defined such that
    ``H = \\sum_{ij} h_{ij} a_i^† a_j + \\sum_{ijkl} g_ijkl a_i^† a_j a_k^† a_l``
    where the one body tensor is ``h_{ij}`` and the two body tensor is ``g_{ijkl}``.
    Note the absence of the factor of 0.5 in the two body term.

    **NOTE**: This function is currently not passing the tests and seg faults.
    It is recommended to use sdstate_lanczos.lanczos_range; see lanczos_range in this file
    The python function itself passes the unittests for small systems and no multiprocessing: 
        sdstate_lanczos.lanczos_total_range
    Args:
        one_body_tensor (Array{Float64,2}): The one body tensor.
        two_body_tensor (Array{Float64,4}): The two body tensor, chemist notation.
        initial_states (Array{Any,1}): The initial_states for the Lanczos algorithm.
        e_nums (Array{Int,1}): The number of electrons for multiprocessing.
        steps (Int): The number of iterations to use in the Lanczos algorithm.
        multiprocessing (Bool): Whether to use multiprocessing.
    Returns:
        E_max (Float64): The maximum energy.
        E_min (Float64): The minimum energy.
"""
function lanczos_total_range(; one_body_tensor::Array{Float64,2},
    two_body_tensor::Array{Float64,4},
    initial_states=[],
    e_nums=[],
    steps::Int=2,
    multiprocessing::Bool=false)
    println("!!!!!!!!!!!!!!!!!!!lanczos_total_range!!!!!!!!!!!!!!!!!!!!!")
    sdstate_lanczos = pyimport("module_sdstate.lanczos_utils")
    E_max, E_min = sdstate_lanczos.lanczos_total_range(Hf=(one_body_tensor, two_body_tensor),
        steps=steps,
        states=initial_states,
        e_nums=e_nums,
        multiprocessing=multiprocessing)

    return E_max, E_min

end

"""
    lanczos_range(one_body_tensor::Array{Float64,2}, two_body_tensor::Array{Float64,4}, num_electrons::Int, initial_state=nothing, steps::Int=2)
    Calculate the range of energies for a given number of electrons.
    The tensors are defined such that
    ``H = \\sum_{ij} h_{ij} a_i^† a_j + \\sum_{ijkl} g_ijkl a_i^† a_j a_k^† a_l``
    where the one body tensor is ``h_{ij}`` and the two body tensor is ``g_{ijkl}``.
    Note the absence of the factor of 0.5 in the two body term.
    **NOTE**: This function is currently not passing the tests and seg faults.
    Please use the python function directly; this function passes the unit tests: 
        sdstate_lanczos.lanczos_range
    Args:
        one_body_tensor (Array{Float64,2}): The one body tensor.
        two_body_tensor (Array{Float64,4}): The two body tensor, chemist notation.
        num_electrons (Int): The number of electrons.
        initial_state (Any): The initial_state for the Lanczos algorithm.
        steps (Int): The number of iterations to use in the Lanczos algorithm.
    Returns:
        E_max (Float64): The maximum energy.
        E_min (Float64): The minimum energy.
"""
function lanczos_range(; one_body_tensor::Array{Float64,2},
    two_body_tensor::Array{Float64,4},
    num_electrons::Int,
    initial_state=nothing,
    steps::Int=2
)
    sdstate_lanczos = pyimport("module_sdstate.lanczos_utils")
    E_max, E_min = sdstate_lanczos.lanczos_range(Hf=(one_body_tensor, two_body_tensor),
        steps=steps, state=initial_state, ne=num_electrons)

    return E_max, E_min

end
