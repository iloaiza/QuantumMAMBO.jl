# Linear programming routines for 1-norm optimization of symmetry shifts
# everything works on orbitals, NOT spin-orbitals. Maintains right symmetries so smaller tensors can be used throughout the optimizations
import HiGHS
import Ipopt


function cartan_tbt_to_vec(cartan_tbt)
    #transforms a two-body tensor corresponding to cartan polynomial into λ vector
    #such that cartan_2b(λ) = cartan_tbt
    n = size(cartan_tbt)[1]

    ν_len = Int(n*(n+1)/2)
    λ_vec = zeros(ν_len)
    idx = 0
    for i in 1:n
        for j in 1:i
            idx += 1
            if i == j
                λ_vec[idx] = cartan_tbt[i,i,i,i]
            else
                λ_vec[idx] = (cartan_tbt[i,i,j,j] + cartan_tbt[j,j,i,i])/2
            end
        end
    end
    
    return λ_vec
end

function τ_mat_builder(SYM_ARR)
    n = size(SYM_ARR[1])[1]
    ν_len = Int(n*(n+1)/2)
    num_syms = length(SYM_ARR)
    τ_mat = zeros(ν_len,num_syms)
    
    for i in 1:num_syms
        τ_mat[:,i] = cartan_tbt_to_vec(SYM_ARR[i])
    end

    return τ_mat
end

function L1_linprog_optimizer_tbt(cartan_tbt :: Array{Float64,4}, τ_mat, verbose=false, model="highs")
    if model == "highs"
        L1_OPT = Model(HiGHS.Optimizer)
    elseif model == "ipopt"
        L1_OPT = Model(Ipopt.Optimizer)
    else
        error("Not defined for model = $model")
    end
    
    if verbose == false
        set_silent(L1_OPT)
    end

    λ_vec = cartan_tbt_to_vec(cartan_tbt)
    ν_len,num_syms = size(τ_mat)

    @variables(L1_OPT, begin
        s[1:num_syms]
        t[1:ν_len]
    end)

    @objective(L1_OPT, Min, sum(t))

    @constraint(L1_OPT, low, τ_mat*s - t - λ_vec .<= 0)
    @constraint(L1_OPT, high, τ_mat*s + t - λ_vec .>= 0)

    optimize!(L1_OPT)

    return value.(s)
end


function L1_linprog_one_body_Ne(obt, verbose=false, model="highs")
    #input: one-body tensor obt
    #obt = U'DU
    #minimizes 1-norm = ∑_(i) |D[i]- s|, corresponds to λ_min[obt - s*Ne]

    if model == "highs"
        L1_OPT = Model(HiGHS.Optimizer)
    elseif model == "ipopt"
        L1_OPT = Model(Ipopt.Optimizer)
    else
        error("Not defined for model = $model")
    end
    
    if verbose == false
        set_silent(L1_OPT)
    end

    ν_len = size(obt)[1]
    D,_ = eigen(obt)

    @variables(L1_OPT, begin
        s
        t[1:ν_len]
    end)

    @objective(L1_OPT, Min, sum(t))

    τ_mat = ones(ν_len)
    @constraint(L1_OPT, low, τ_mat*s - t - D .<= 0)
    @constraint(L1_OPT, high, τ_mat*s + t - D .>= 0)

    optimize!(L1_OPT)

    return value.(s)[1]
end
