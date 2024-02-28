using PythonCall
sdstate_lanczos = pyimport("module_sdstate.lanczos_utils")


function lanczos_total_range(;one_body_tensor::Array{Float64,2},
    two_body_tensor::Array{Float64,4},
    num_electrons::Int,
    states=[],
    e_nums=[num_electrons],
    steps::Int=2,
    multiprocessing::Bool=false)

    E_max, E_min = sdstate_lanczos.lanczos_total_range(Hf=(one_body_tensor, two_body_tensor),
        steps=steps,
        states=states,
        e_nums=e_nums,
        multiprocessing=multiprocessing)

    return E_max, E_min

end