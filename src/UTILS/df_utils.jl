num_calls(λ,ϵ=qpe_error) = Int(ceil(π*λ/2ϵ))

function qrom_cost(a,b,c,d,e)
    n = log2(((a + b + c) / d) ^ 0.5)
    k = [2 ^ floor(n), 2 ^ ceil(n)]
    cost = ceil.((a + b) ./ k) + ceil.(c ./ k) + d .* (k .+ e)

    return Int(cost[argmin(cost)]), Int(k[argmin(cost)])
end

function df_unitary_cost(n, rank_r, rank_m, rank_max, br=7, alpha=10, beta=20)
    rank_rm = rank_r * rank_m

    # eta is computed based on step 1.(a) in page 030305-41 of PRX Quantum 2, 030305 (2021)
    eta = [log2(n) for n in range(1, rank_r + 1) if rank_r % n == 0]
    eta = Int(maximum([n for n in eta if n % 1 == 0]))

    nxi = ceil(log2(rank_max))  # Eq. (C14) of PRX Quantum 2, 030305 (2021)
    nl = ceil(log2(rank_r + 1))  # Eq. (C14) of PRX Quantum 2, 030305 (2021)
    nlxi = ceil(log2(rank_rm + n / 2))  # Eq. (C15) of PRX Quantum 2, 030305 (2021)

    bp1 = nl + alpha  # Eq. (C27) of PRX Quantum 2, 030305 (2021)
    bo = nxi + nlxi + br + 1  # Eq. (C29) of PRX Quantum 2, 030305 (2021)
    bp2 = nxi + alpha + 2  # Eq. (C31) of PRX Quantum 2, 030305 (2021)

    # cost is computed using Eq. (C39) of PRX Quantum 2, 030305 (2021)
    cost = (
        9 * nl - 6 * eta + 12 * br + 34 * nxi + 8 * nlxi + 9 * alpha + 3 * n * beta - 6 * n - 43
    )
    cost += qrom_cost(rank_r, 1, 0, bp1, -1)[1]
    cost += qrom_cost(rank_r, 1, 0, bo, -1)[1]
    cost += qrom_cost(rank_r, 1, 0, 1, 0)[1] * 2
    cost += qrom_cost(rank_rm, n / 2, rank_rm, n * beta, 0)[1]
    cost += qrom_cost(rank_rm, n / 2, rank_rm, 2, 0)[1] * 2
    cost += qrom_cost(rank_rm, n / 2, rank_rm, 2 * bp2, -1)[1]

    return Int(cost)
end

function df_gate_cost(n, lamb, error, rank_r, rank_m, rank_max, br=7, alpha=10, beta=20)
	e_cost = num_calls(lamb, error)
	u_cost = df_unitary_cost(n, rank_r, rank_m, rank_max, br, alpha, beta)

	return Int(e_cost * u_cost)
end

function df_qubit_cost(n, lamb, error, rank_r, rank_m, rank_max, br=7, alpha=10, beta=20)
    rank_rm = rank_r * rank_m

    nxi = ceil(log2(rank_max))  # Eq. (C14) of PRX Quantum 2, 030305 (2021)
    nl = ceil(log2(rank_r + 1))  # Eq. (C14) of PRX Quantum 2, 030305 (2021)
    nlxi = ceil(log2(rank_rm + n / 2))  # Eq. (C15) of PRX Quantum 2, 030305 (2021)

    bo = nxi + nlxi + br + 1  # Eq. (C29) of PRX Quantum 2, 030305 (2021)
    bp2 = nxi + alpha + 2  # Eq. (C31) of PRX Quantum 2, 030305 (2021)
    # kr is taken from Eq. (C39) of PRX Quantum 2, 030305 (2021)
    kr = qrom_cost(rank_rm, n / 2, rank_rm, n * beta, 0)[2]

    # the cost is computed using Eq. (C40) of PRX Quantum 2, 030305 (2021)
    e_cost = num_calls(lamb, error)
    cost = n + 2 * nl + nxi + 3 * alpha + beta + bo + bp2
    cost += kr * n * beta / 2 + 2 * ceil(log2(e_cost + 1)) + 7

    return Int(cost)
end

function df_frags_analysis(df_frags; trunc_thresh = 1e-5)
    rank_r = length(df_frags)
    Lxi = 0
    rank_max = 0
    for frag in df_frags
        e_vec = sqrt(abs(frag.coeff)) .* frag.C.λ
        truncation = sum(abs.(e_vec)) * abs.(e_vec)
        idx = [t > trunc_thresh for t in truncation]
        Lxi += sum(idx)
        rank_max = maximum([rank_max, sum(idx)])
    end
    
    rank_m = Lxi / rank_r

    return rank_r, rank_m, rank_max
end

function df_be_qubit_cost(n, rank_r, rank_m, rank_max, br=7, alpha=10, beta=20)
    rank_rm = rank_r * rank_m

    nxi = ceil(log2(rank_max))  # Eq. (C14) of PRX Quantum 2, 030305 (2021)
    nl = ceil(log2(rank_r + 1))  # Eq. (C14) of PRX Quantum 2, 030305 (2021)
    nlxi = ceil(log2(rank_rm + n / 2))  # Eq. (C15) of PRX Quantum 2, 030305 (2021)

    bo = nxi + nlxi + br + 1  # Eq. (C29) of PRX Quantum 2, 030305 (2021)
    bp2 = nxi + alpha + 2  # Eq. (C31) of PRX Quantum 2, 030305 (2021)
    # kr is taken from Eq. (C39) of PRX Quantum 2, 030305 (2021)
    kr = qrom_cost(rank_rm, n / 2, rank_rm, n * beta, 0)[2]

    # the cost is computed using Eq. (C40) of PRX Quantum 2, 030305 (2021), removed log2(I+1) coming from QPE procedure
    cost = n + 2 * nl + nxi + 3 * alpha + beta + bo + bp2
    cost += kr * n * beta / 2 + 7

    return Int(cost)
end

function df_be_cost(df_frags; trunc_thresh = 1e-5, br=7, alpha=10, beta=20, kwargs...)
    rank_r, rank_m, rank_max = df_frags_analysis(df_frags, trunc_thresh=trunc_thresh)
    n = df_frags[1].N * 2

    be_cost = df_unitary_cost(n, rank_r, rank_m, rank_max, br, alpha, beta) * 4 #multiplication by 4 to go to T-gates from Toffolis
    qubit_cost = df_be_qubit_cost(n, rank_r, rank_m, rank_max, br, alpha, beta)

    return be_cost, qubit_cost
end

function get_df_resources(λ, df_frags; trunc_thresh=1e-5, error=0.001, kwargs...)
    rank_r, rank_m, rank_max = df_frags_analysis(df_frags, trunc_thresh=trunc_thresh)
    n = df_frags[1].N * 2

    gate_cost = df_gate_cost(n, λ, error, rank_r, rank_m, rank_max; kwargs...)
    qubit_cost = df_qubit_cost(n, λ, error, rank_r, rank_m, rank_max; kwargs...)

    return gate_cost, qubit_cost
end

function df_fragment_bliss(frag)
    eps_vec = frag.C.λ

    function cost(x)
        return (sum(abs.(eps_vec .- x))) ^ 2
    end

    x0 = [0.0]
    sol = optimize(cost, x0, BFGS())
    C_bliss = cartan_1b(frag.spin_orb, eps_vec .- sol.minimizer , frag.N)

    return F_FRAG(1, frag.U, DF(), C_bliss, frag.N, frag.spin_orb, frag.coeff, frag.has_coeff), sol.minimizer
end

function ob_fragment_bliss(frag)
    eps_vec = frag.C.λ

    function cost(x)
        return (sum(abs.(eps_vec .- x)))
    end

    x0 = [0.0]
    sol = optimize(cost, x0, BFGS())
    C_bliss = cartan_1b(frag.spin_orb, eps_vec .- sol.minimizer , frag.N)

    return F_FRAG(1, frag.U, OBF(), C_bliss, frag.N, frag.spin_orb, frag.coeff, frag.has_coeff)
end

function df_bliss(df_frags)
    bliss_frags = []
    ϕs = []

    for frag in df_frags
        new_frag, ϕ = df_fragment_bliss(frag)
        push!(bliss_frags, new_frag)
        push!(ϕs, ϕ)
    end

    return bliss_frags, ϕs
end