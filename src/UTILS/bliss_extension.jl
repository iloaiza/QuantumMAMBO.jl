"""
We assume size(F.mbts[2], 1) == size(F.mbts[3], 1)
"""
function bliss_linprog_extension(F::F_OP, η; model="highs", verbose=true, SAVELOAD=SAVING, SAVENAME=DATAFOLDER * "BLISS.h5", num_threads::Int=1)
  if F.spin_orb
    error("BLISS not defined for spin-orb=true!")
  end

  if model == "highs"
    L1_OPT = Model(HiGHS.Optimizer)
  elseif model == "ipopt"
    L1_OPT = Model(Ipopt.Optimizer)
  else
    error("Not defined for model = $model")
  end

  if num_threads > 1
    HiGHS.Highs_resetGlobalScheduler(1)
    set_attribute(L1_OPT, JuMP.MOI.NumberOfThreads(), num_threads)
  end

  if verbose == false
    set_silent(L1_OPT)
  end

  ### Saving part begins ###

  if SAVELOAD
    fid = h5open(SAVENAME, "cw")
    if haskey(fid, "BLISS")
      BLISS_group = fid["BLISS"]
      if haskey(BLISS_group, "ovec")
        println("Loading results for BLISS optimization from $SAVENAME")
        ovec = read(BLISS_group, "ovec")
        t1 = read(BLISS_group, "t1")
        t2 = read(BLISS_group, "t2")
        t3 = read(BLISS_group, "t3")
        t_opt = [t1, t2, t3]
        O = zeros(F.N, F.N)
        idx = 1
        for i = 1:F.N
          for j = 1:F.N
            O[i, j] = ovec[idx]
            idx += 1
          end
        end
        @show t_opt
        @show O
        ham = fid["BLISS_HAM"]
        F_new = F_OP((read(ham, "h_const"), read(ham, "obt"), read(ham, "tbt"), read(ham, "threebt")))
        # println("The L1 cost of symmetry treated fermionic operator is: ", PAULI_L1(F_new))
        close(fid)
        return F_new, F - F_new
      end
    end
    close(fid)
  end

  ### Saving part ends ###


  ovec_len = Int(F.N * (F.N + 1) / 2)

  ν1_len = F.N^2
  ν2_len = F.N^4
  ν3_len = Int((F.N * (F.N - 1) / 2)^2)
  v4_len = F.N^2 * (F.N - 1) * (F.N - 2)

  v6_len = (F.N * (F.N - 1) * (F.N - 2))^2

  @variables(L1_OPT, begin
    t[1:3]
    obt[1:ν1_len]
    tbt1[1:ν2_len]
    tbt2[1:ν3_len]
    tbt3[1:v4_len]
    tbt4[1:v4_len]
    threebt[1:v6_len]
    omat[1:F.N^2]
  end)

  @objective(L1_OPT, Min, sum(obt) + sum(tbt1) + sum(tbt2) + sum(tbt3) + sum(tbt4) + sum(threebt))

  ### Constraint ###
  obt_corr = ob_correction(F)
  # h_ij + 2\sum_k g_ijkk 
  λ1 = zeros(ν1_len)
  idx = 0
  for i in 1:F.N
    for j in 1:F.N
      idx += 1
      λ1[idx] = F.mbts[2][i, j] + obt_corr[i, j] + ob_correction_3bd_1(F, i, j) + ob_correction_3bd_2(F, i, j) + ob_correction_3bd_3(F, i, j) + ob_correction_3bd_4(F, i, j)
    end
  end

  # \mu_1\delta_ij t[1] corresponds to \mu_1
  τ_11 = zeros(ν1_len)
  idx = 0
  for i in 1:F.N
    for j in 1:F.N
      idx += 1
      if i == j
        τ_11[idx] = 1
      end
    end
  end

  # 2N \mu_2\delta_ij t[2] corresponds to \mu_2
  τ_12 = zeros(ν1_len)
  idx = 0
  for i in 1:F.N
    for j in 1:F.N
      idx += 1
      if i == j
        τ_12[idx] = 2 * F.N
      end
    end
  end

  τ_13 = zeros(ν1_len)
  idx = 0
  for i in 1:F.N
    for j in 1:F.N
      idx += 1
      if i == j
        τ_13[idx] = 1 / 8 * (3 * F.N + 3) + 3 / 2 * F.N^2
      end
    end
  end

  # \xi part omat corresponds to \xi
  T1 = zeros(ν1_len, ν1_len)
  T1 += Diagonal((η - F.N) * ones(ν1_len))
  idx1 = 0
  for i in 1:F.N
    for j in 1:F.N
      idx1 += 1
      idx2 = 0
      for k in 1:F.N
        for l in 1:F.N
          idx2 += 1
          if i == j && k == l
            T1[idx1, idx2] -= 1
          end
        end
      end
    end
  end


  @constraint(L1_OPT, low_1, λ1 - τ_11 * t[1] - τ_12 * t[2] - τ_13 * t[3] + T1 * omat - obt .<= 0)
  @constraint(L1_OPT, high_1, λ1 - τ_11 * t[1] - τ_12 * t[2] - τ_13 * t[3] + T1 * omat + obt .>= 0)

  # This is the 1/2 g_ijkl part
  tbt_corr_1 = tb_correction_1(F)
  λ2 = zeros(ν2_len)
  idx = 0
  for i in 1:F.N
    for j in 1:F.N
      for k in 1:F.N
        for l in 1:F.N
          idx += 1
          λ2[idx] = 0.5 * F.mbts[3][i, j, k, l] + 0.5 * tbt_corr_1[i, j, k, l]
        end
      end
    end
  end

  # this is the \mu_2\delta_ij\delta_kl part
  τ_21 = zeros(ν2_len)
  idx = 0
  for i in 1:F.N
    for j in 1:F.N
      for k in 1:F.N
        for l in 1:F.N
          idx += 1
          if i == j && k == l
            τ_21[idx] = 0.5
          end
        end
      end
    end
  end

  # this is the \mu_2\delta_ij\delta_kl part
  τ_22 = zeros(ν2_len)
  idx = 0
  for i in 1:F.N
    for j in 1:F.N
      for k in 1:F.N
        for l in 1:F.N
          idx += 1
          if i == j && k == l
            τ_22[idx] = 1.5 * F.N
          end
        end
      end
    end
  end

  # This is the 1/2 (\xi_ij\delta_kl + \delta_ij\xi_kl) part
  T2 = zeros(ν2_len, ν1_len)
  idx = 0
  idx_ij = 0
  for i in 1:F.N
    for j in 1:F.N
      idx_ij += 1
      idx_kl = 0
      for k in 1:F.N
        for l in 1:F.N
          idx += 1
          idx_kl += 1
          if i == j
            T2[idx, idx_kl] += 0.5
          end
          if k == l
            T2[idx, idx_ij] += 0.5
          end
        end
      end
    end
  end

  @constraint(L1_OPT, low_2, λ2 - τ_21 * t[2] - τ_22 * t[3] - 0.5 * T2 * omat - tbt1 .<= 0)
  @constraint(L1_OPT, high_2, λ2 - τ_21 * t[2] - τ_22 * t[3] - 0.5 * T2 * omat + tbt1 .>= 0)

  T_dict = zeros(Int64, F.N, F.N)
  idx = 0
  for i in 1:F.N
    for j in 1:F.N
      idx += 1
      T_dict[i, j] = idx
    end
  end

  tbt_corr_2 = tb_correction_2(F)
  # g_ijkl - g_ilkj
  λ3 = zeros(ν3_len)
  idx = 0
  for i in 1:F.N
    for j in 1:F.N
      for k in 1:i-1
        for l in 1:j-1
          idx += 1
          λ3[idx] = F.mbts[3][i, j, k, l] - F.mbts[3][i, l, k, j] + tbt_corr_2[i, j, k, l] - tbt_corr_2[i, l, k, j]
        end
      end
    end
  end

  # \mu_2(\delta_ij\delta_kl - \delta_il\delta_kj)
  τ_31 = zeros(ν3_len)
  idx = 0
  for i in 1:F.N
    for j in 1:F.N
      for k in 1:i-1
        for l in 1:j-1
          idx += 1
          if i == j && k == l
            τ_31[idx] += 1
          end
          if i == l && k == j
            τ_31[idx] -= 1
          end
        end
      end
    end
  end

  τ_32 = zeros(ν3_len)
  idx = 0
  for i in 1:F.N
    for j in 1:F.N
      for k in 1:i-1
        for l in 1:j-1
          idx += 1
          if i == j && k == l
            τ_32[idx] += 3 * F.N
          end
          if i == l && k == j
            τ_32[idx] -= 3 * F.N
          end
        end
      end
    end
  end



  T3 = zeros(ν3_len, ν1_len)
  idx = 0
  for i in 1:F.N
    for j in 1:F.N
      for k in 1:i-1
        for l in 1:j-1
          idx += 1

          idx_ij = T_dict[i, j]
          if k == l
            T3[idx, idx_ij] += 0.5
          end

          idx_kl = T_dict[k, l]
          if i == j
            T3[idx, idx_kl] += 0.5
          end

          idx_il = T_dict[i, l]
          if k == j
            T3[idx, idx_il] -= 0.5
          end

          idx_kj = T_dict[k, j]
          if i == l
            T3[idx, idx_kj] -= 0.5
          end
        end
      end
    end
  end

  @constraint(L1_OPT, low_3, λ3 - τ_31 * t[2] - τ_32 * t[3] - T3 * omat - tbt2 .<= 0)
  @constraint(L1_OPT, high_3, λ3 - τ_31 * t[2] - τ_32 * t[3] - T3 * omat + tbt2 .>= 0)

  λ4 = zeros(v4_len)
  idx = 0
  for i in 1:F.N
    for k in 1:F.N
      for m in 1:F.N
        for j in 1:F.N
          if i != k && k != m && i != m
            idx += 1
            λ4[idx] += 1 / 8 * (sum(F.mbts[4][i, l, k, j, m, l] for l in 1:F.N) + sum(F.mbts[4][i, l, k, l, m, j] for l in 1:F.N) + sum(F.mbts[4][i, j, k, l, m, l] for l in 1:F.N))
            λ4[idx] += 1 / 8 * F.mbts[4][i, j, k, j, m, j]
          end
        end
      end
    end
  end

  τ_41 = zeros(v4_len)
  idx = 0
  for i in 1:F.N
    for k in 1:F.N
      for m in 1:F.N
        for j in 1:F.N
          if i != k && k != m && i != m
            idx += 1
            if k == j
              τ_41[idx] += 1
            end
            if m == j
              τ_41[idx] += 1
            end
            if i == j
              τ_41[idx] += 1
            end
            if i == j && k == j && m == j
              τ_41[idx] -= 1
            end
          end
        end
      end
    end
  end



  @constraint(L1_OPT, low_4, λ4 - 1 / 8 * τ_41 * t[3] - tbt3 .<= 0)
  @constraint(L1_OPT, high_4, λ4 - 1 / 8 * τ_41 * t[3] + tbt3 .>= 0)

  λ5 = zeros(v4_len)
  idx = 0
  for j in 1:F.N
    for l in 1:F.N
      for n in 1:F.N
        for i in 1:F.N
          if j != l && l != n && j != n
            idx += 1
            λ5[idx] += 1 / 8 * (-sum(F.mbts[4][k, j, i, l, k, n] for k in 1:F.N) + sum(F.mbts[4][k, j, k, l, i, n] for k in 1:F.N) + sum(F.mbts[4][i, j, k, l, k, n] for k in 1:F.N))
            λ5[idx] += 1 / 8 * F.mbts[4][i, j, i, l, i, n]
          end
        end
      end
    end
  end

  τ_51 = zeros(v4_len)
  idx = 0
  for j in 1:F.N
    for l in 1:F.N
      for n in 1:F.N
        for i in 1:F.N
          if j != l && l != n && j != n
            idx += 1
            if l == i
              τ_51[idx] += 1
            end
            if n == i
              τ_51[idx] += 1
            end
            if j == i
              τ_51[idx] += 1
            end
            if j == i && l == i && n == i
              τ_51[idx] -= 1
            end
          end
        end
      end
    end
  end



  @constraint(L1_OPT, low_5, λ5 - 1 / 8 * τ_51 * t[3] - tbt4 .<= 0)
  @constraint(L1_OPT, high_5, λ5 - 1 / 8 * τ_51 * t[3] + tbt4 .>= 0)




  λ6 = zeros(v6_len)
  idx = 0
  for i in 1:F.N
    for j in 1:F.N
      for k in 1:F.N
        for l in 1:F.N
          for m in 1:F.N
            for n in 1:F.N
              if i != k && k != m && i != m && j != l && l != n && j != n
                idx += 1
                λ6[idx] = 1 / 8 * F.mbts[4][i, j, k, l, m, n]
              end
            end
          end
        end
      end
    end
  end

  τ_61 = zeros(v6_len)
  idx = 0
  for i in 1:F.N
    for j in 1:F.N
      for k in 1:F.N
        for l in 1:F.N
          for m in 1:F.N
            for n in 1:F.N
              if i != k && k != m && i != m && j != l && l != n && j != n
                idx += 1
                if i == j && k == l && m == n
                  τ_61[idx] = 1 / 8
                end
              end
            end
          end
        end
      end
    end
  end

  @constraint(L1_OPT, low_6, λ6 - τ_61 * t[3] - threebt .<= 0)
  @constraint(L1_OPT, high_6, λ6 - τ_61 * t[3] + threebt .>= 0)


  JuMP.optimize!(L1_OPT)

  t_opt = value.(t)
  o_opt = value.(omat)
  @show t_opt
  idx = 1
  O = zeros(F.N, F.N)
  for i = 1:F.N
    for j = 1:F.N
      O[i, j] = o_opt[idx]
      idx += 1
    end
  end
  @show O
  O = (O + O') / 2


  Ne, Ne2, Ne3 = symmetry_builder_extension(F)


  s_threebt = t_opt[3] * Ne3.mbts[4]
  s2_tbt = t_opt[2] * Ne2.mbts[3]
  for i in 1:F.N
    for j in 1:F.N
      for k in 1:F.N
        s2_tbt[i, j, k, k] += O[i, j]
        s2_tbt[k, k, i, j] += O[i, j]
      end
    end
  end
  s3 = F_OP(([0], [0], [0], s_threebt))
  s2 = F_OP(([0], [0], s2_tbt))

  s1_obt = t_opt[1] * Ne.mbts[2] - 2η * O
  s1 = F_OP(([-t_opt[1] * η - t_opt[2] * η^2 - t_opt[3] * η^3], s1_obt))

  F_new = F - s1 - s2 - s3
  println("h_const Bliss:", F_new.mbts[1])
  if SAVELOAD
    fid = h5open(SAVENAME, "cw")
    create_group(fid, "BLISS")
    BLISS_group = fid["BLISS"]
    println("Saving results of BLISS optimization to $SAVENAME")
    BLISS_group["ovec"] = o_opt
    BLISS_group["t1"] = t_opt[1]
    BLISS_group["t2"] = t_opt[2]
    BLISS_group["t3"] = t_opt[3]
    BLISS_group["N"] = F.N
    BLISS_group["Ne"] = η
    create_group(fid, "BLISS_HAM")
    MOL_DATA = fid["BLISS_HAM"]
    MOL_DATA["h_const"] = F_new.mbts[1]
    MOL_DATA["obt"] = F_new.mbts[2]
    MOL_DATA["tbt"] = F_new.mbts[3]
    MOL_DATA["threebt"] = F_new.mbts[4]
    MOL_DATA["eta"] = η
    close(fid)
  end

  println("")

  #println("The L1 cost of symmetry treated fermionic operator is: ", PAULI_L1(F_new))
  return F_new, s1 + s2 + s3
end