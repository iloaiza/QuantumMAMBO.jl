######## RUNNING CODE
import Pkg
Pkg.activate("./")
using QuantumMAMBO: L1_ROUTINE, DATAFOLDER, LOCALIZED_XYZ_HAM, symmetry_treatment, bliss_optimizer

using HDF5
###### SAVELOAD ROUTINES FOR MOLECULAR HAMILTONIAN #######
CHAINS = collect(2:2:24)

num_chains = length(CHAINS)

function H_CHAIN(n, r = 1.4)
	xyz = String[]
	for i in 1:n
		my_r = i*r
		push!(xyz,"H 0.0 0.0 $my_r\n")
	end

	return LOCALIZED_XYZ_HAM(xyz, DATAFOLDER * "chain_H$n", true)
end

#1 goes to HF, 2 to FB
Λs = zeros(num_chains, 5, 2, 3) #ΔE, DF, Pauli, AC, MHC
Us = zeros(Int64, num_chains, 4, 2, 3) #DF, Pauli, AC, MHC
for i in 1:num_chains
	name = DATAFOLDER * "chain_H$(CHAINS[i])"
	println("Starting calculations for chain length = $(CHAINS[i])...")
	t00 = time()
	Hhf, Hfb, η = H_CHAIN(CHAINS[i])

	Λs[i,:,1,1], Us[i,:,1,1] = L1_ROUTINE(Hhf, name * ".h5", prefix="HF", dE=false)
	Λs[i,:,2,1], Us[i,:,2,1] = L1_ROUTINE(Hfb, name * "_FB.h5", prefix="FB", dE=false)

	println("Obtaining symmetry shifts...")
	name = DATAFOLDER * "chain_H$(CHAINS[i])_SYM"
	H_SYM_hf, _ = symmetry_treatment(Hhf, verbose=false, SAVENAME=name * ".h5")
	H_SYM_fb, _ = symmetry_treatment(Hfb, verbose=false, SAVENAME=name * "_FB.h5")
	Λs[i,:,1,2], Us[i,:,1,2] = L1_ROUTINE(H_SYM_hf, name * ".h5", prefix="HF", dE=false)
	Λs[i,:,2,2], Us[i,:,2,2] = L1_ROUTINE(H_SYM_fb, name * "_FB.h5", prefix="FB", dE=false)

	println("Obtaining BLISS...")
	name = DATAFOLDER * "chain_H$(CHAINS[i])_BLISS"
	@time H_bliss_hf = bliss_optimizer(Hhf, η, verbose=false, SAVENAME=name * ".h5")
	@time H_bliss_fb = bliss_optimizer(Hfb, η, verbose=false, SAVENAME=name * "_FB.h5")
	Λs[i,:,1,3], Us[i,:,1,3] = L1_ROUTINE(H_bliss_hf, name * ".h5", prefix="HF", dE=false)
	Λs[i,:,2,3], Us[i,:,2,3] = L1_ROUTINE(H_bliss_fb, name * "_FB.h5", prefix="FB", dE=false)

	
	println("Finished chain length = $(CHAINS[i]) after $(time() - t00) seconds!")
	@show Λs
	@show Us
	println("#################\n#################\n#################\n#################\n\n\n\n\n\n")
end

@show Λs
@show Us

using Plots
using Plots.PlotMeasures
gr()
using DataFrames, GLM
function log_regression(X, Y; return_b = false, fit_range = collect(1:length(X)))
	Xlog = log10.(X[fit_range])
	Ylog = log10.(Y[fit_range])
	data = DataFrame(X=Xlog, Y=Ylog)
	ols = lm(@formula(Y~X), data)
	α = round(coef(ols)[2], digits=3)
	β = coef(ols)[1]
	R = round(r2(ols)[1], digits=4)
	Yfit = (X .^ α) * 10^β
	if return_b
		return R, Yfit, α, β
	else
		return R, Yfit, α
	end
end
FONT=font(40)
L_FONT = font(18)
SIZE=[1980,1020]
L_MARG=[15mm 0mm]
B_MARG=[20mm 0mm]
XTICKS = [2, 4, collect(10:10:maximum(CHAINS))...]
XLABS = string.(XTICKS)
XLIMS = (7.8, 24.2)

CALC_LAST = 5 #over how many points should the fits be done, num_chains-1 for considering all points

P = scatter(CHAINS, Λs[:,2,1,1], label="Pauli (CMOs)", m=(10.0, :square, :red, stroke(0)))
scatter!(CHAINS, Λs[:,2,2,1], label="Pauli (FB)", m=(10.0, :square, :cyan, stroke(0)))
scatter!(CHAINS, Λs[:,3,1,1], label="AC (CMOs)", m=(10.0, :dtriangle, :green, stroke(0)))
scatter!(CHAINS, Λs[:,3,2,1], label="AC (FB)", m=(10.0, :dtriangle, :gray, stroke(0)))
scatter!(CHAINS, Λs[:,4,1,1], label="DF", m=(10.0, :utriangle, :black, stroke(0)))


plot!(xscale=:log, yscale=:log, xlabel="Chain dimension N", ylabel="1-norm λ", legend=false,
	xtickfont = FONT,xguidefont=FONT,ytickfont = FONT,yguidefont=FONT,size=SIZE,legendfont=L_FONT,
	left_margin=L_MARG,bottom_margin=B_MARG, xticks=(XTICKS, XLABS), xlims=XLIMS)


f_range = collect(num_chains - CALC_LAST:num_chains)

println("Starting log fits for full Hamiltonian...")
R_DF, Y_DF, α_DF, β_DF = log_regression(CHAINS, Λs[:, 4, 1, 1], fit_range = f_range, return_b = true)
R_Pauli_HF, Y_Pauli_HF, α_Pauli_HF, β_Pauli_HF = log_regression(CHAINS, Λs[:, 2, 1, 1], fit_range = f_range, return_b = true)
R_Pauli_FB, Y_Pauli_FB, α_Pauli_FB, β_Pauli_FB = log_regression(CHAINS, Λs[:, 2, 2, 1], fit_range = f_range, return_b = true)
R_AC_HF, Y_AC_HF, α_AC_HF, β_AC_HF = log_regression(CHAINS, Λs[:, 3, 1, 1], fit_range = f_range, return_b = true)
R_AC_FB, Y_AC_FB, α_AC_FB, β_AC_FB = log_regression(CHAINS, Λs[:, 3, 2, 1], fit_range = f_range, return_b = true)

plot!(CHAINS, Y_DF, line=(0.5, :solid, :black), label=false)#, label="λDF")
plot!(CHAINS, Y_Pauli_HF, line=(0.5, :solid, :red), label=false)#, label="λPauli_HF")
plot!(CHAINS, Y_Pauli_FB, line=(0.5, :dash, :cyan), label=false)#, label="λPauli_FB")
plot!(CHAINS, Y_AC_HF, line=(0.5, :solid, :green), label=false)#, label="λAC_HF")
plot!(CHAINS, Y_AC_FB, line=(0.5, :dash, :gray), label=false)#, label="λAC_FB")

display(P)

savefig(P, "H_CHAIN_FULL.png")
plot!(legend=true)
savefig(P, "LEGEND_FULL.png")

println("α_DF = $α_DF, β_DF=$β_DF, R_DF = $R_DF")
println("α_Pauli_HF = $α_Pauli_HF, β_Pauli_HF=$β_Pauli_HF, R_Pauli_HF = $R_Pauli_HF")
println("α_Pauli_FB = $α_Pauli_FB, β_Pauli_FB=$β_Pauli_FB, R_Pauli_FB = $R_Pauli_FB")
println("α_AC_HF = $α_AC_HF, β_AC_HF=$β_AC_HF, R_AC_HF = $R_AC_HF")
println("α_AC_FB = $α_AC_FB, β_AC_FB=$β_AC_FB, R_AC_FB = $R_AC_FB")

P = scatter(CHAINS, Λs[:,2,1,2], label="Pauli (CMOs)", m=(10.0, :square, :red, stroke(0)))
scatter!(CHAINS, Λs[:,2,2,2], label="Pauli (FB)", m=(10.0, :square, :cyan, stroke(0)))
scatter!(CHAINS, Λs[:,3,1,2], label="AC (CMOs)", m=(10.0, :dtriangle, :green, stroke(0)))
scatter!(CHAINS, Λs[:,3,2,2], label="AC (FB)", m=(10.0, :dtriangle, :gray, stroke(0)))
scatter!(CHAINS, Λs[:,4,1,2], label="DF", m=(10.0, :utriangle, :black, stroke(0)))


plot!(xscale=:log, yscale=:log, xlabel="Chain dimension N", ylabel="1-norm λ", legend=false,
	xtickfont = FONT,xguidefont=FONT,ytickfont = FONT,yguidefont=FONT,size=SIZE,legendfont=L_FONT,
	left_margin=L_MARG,bottom_margin=B_MARG, xticks=(XTICKS, XLABS), xlims=XLIMS)


f_range = collect(num_chains - CALC_LAST:num_chains)

println("Starting log fits for symmetry shifted Hamiltonian...")
R_DF, Y_DF, α_DF, β_DF = log_regression(CHAINS, Λs[:, 4, 1, 2], fit_range = f_range, return_b = true)
R_Pauli_HF, Y_Pauli_HF, α_Pauli_HF, β_Pauli_HF = log_regression(CHAINS, Λs[:, 2, 1, 2], fit_range = f_range, return_b = true)
R_Pauli_FB, Y_Pauli_FB, α_Pauli_FB, β_Pauli_FB = log_regression(CHAINS, Λs[:, 2, 2, 2], fit_range = f_range, return_b = true)
R_AC_HF, Y_AC_HF, α_AC_HF, β_AC_HF = log_regression(CHAINS, Λs[:, 3, 1, 2], fit_range = f_range, return_b = true)
R_AC_FB, Y_AC_FB, α_AC_FB, β_AC_FB = log_regression(CHAINS, Λs[:, 3, 2, 2], fit_range = f_range, return_b = true)

plot!(CHAINS, Y_DF, line=(0.5, :solid, :black), label=false)#, label="λDF")
plot!(CHAINS, Y_Pauli_HF, line=(0.5, :solid, :red), label=false)#, label="λPauli_HF")
plot!(CHAINS, Y_Pauli_FB, line=(0.5, :dash, :cyan), label=false)#, label="λPauli_FB")
plot!(CHAINS, Y_AC_HF, line=(0.5, :solid, :green), label=false)#, label="λAC_HF")
plot!(CHAINS, Y_AC_FB, line=(0.5, :dash, :gray), label=false)#, label="λAC_FB")

display(P)

savefig(P, "H_CHAIN_SYM.png")
plot!(legend=true)
savefig(P, "LEGEND_SYM.png")

println("α_DF = $α_DF, β_DF=$β_DF, R_DF = $R_DF")
println("α_Pauli_HF = $α_Pauli_HF, β_Pauli_HF=$β_Pauli_HF, R_Pauli_HF = $R_Pauli_HF")
println("α_Pauli_FB = $α_Pauli_FB, β_Pauli_FB=$β_Pauli_FB, R_Pauli_FB = $R_Pauli_FB")
println("α_AC_HF = $α_AC_HF, β_AC_HF=$β_AC_HF, R_AC_HF = $R_AC_HF")
println("α_AC_FB = $α_AC_FB, β_AC_FB=$β_AC_FB, R_AC_FB = $R_AC_FB")

P = scatter(CHAINS, Λs[:,2,1,3], label="Pauli (CMOs)", m=(10.0, :square, :red, stroke(0)))
scatter!(CHAINS, Λs[:,2,2,3], label="Pauli (FB)", m=(10.0, :square, :cyan, stroke(0)))
scatter!(CHAINS, Λs[:,3,1,3], label="AC (CMOs)", m=(10.0, :dtriangle, :green, stroke(0)))
scatter!(CHAINS, Λs[:,3,2,3], label="AC (FB)", m=(10.0, :dtriangle, :gray, stroke(0)))
scatter!(CHAINS, Λs[:,4,1,3], label="DF", m=(10.0, :utriangle, :black, stroke(0)))


plot!(xscale=:log, yscale=:log, xlabel="Chain dimension N", ylabel="1-norm λ", legend=false,
	xtickfont = FONT,xguidefont=FONT,ytickfont = FONT,yguidefont=FONT,size=SIZE,legendfont=L_FONT,
	left_margin=L_MARG,bottom_margin=B_MARG, xticks=(XTICKS, XLABS), xlims=XLIMS)


f_range = collect(num_chains - CALC_LAST:num_chains)

println("Starting log fits for BLISS Hamiltonian...")
R_DF, Y_DF, α_DF, β_DF = log_regression(CHAINS, Λs[:, 4, 1, 3], fit_range = f_range, return_b = true)
R_Pauli_HF, Y_Pauli_HF, α_Pauli_HF, β_Pauli_HF = log_regression(CHAINS, Λs[:, 2, 1, 3], fit_range = f_range, return_b = true)
R_Pauli_FB, Y_Pauli_FB, α_Pauli_FB, β_Pauli_FB = log_regression(CHAINS, Λs[:, 2, 2, 3], fit_range = f_range, return_b = true)
R_AC_HF, Y_AC_HF, α_AC_HF, β_AC_HF = log_regression(CHAINS, Λs[:, 3, 1, 3], fit_range = f_range, return_b = true)
R_AC_FB, Y_AC_FB, α_AC_FB, β_AC_FB = log_regression(CHAINS, Λs[:, 3, 2, 3], fit_range = f_range, return_b = true)

plot!(CHAINS, Y_DF, line=(0.5, :solid, :black), label=false)#, label="λDF")
plot!(CHAINS, Y_Pauli_HF, line=(0.5, :solid, :red), label=false)#, label="λPauli_HF")
plot!(CHAINS, Y_Pauli_FB, line=(0.5, :dash, :cyan), label=false)#, label="λPauli_FB")
plot!(CHAINS, Y_AC_HF, line=(0.5, :solid, :green), label=false)#, label="λAC_HF")
plot!(CHAINS, Y_AC_FB, line=(0.5, :dash, :gray), label=false)#, label="λAC_FB")

display(P)

savefig(P, "H_CHAIN_BLISS.png")
plot!(legend=true)
savefig(P, "LEGEND_BLISS.png")

println("α_DF = $α_DF, β_DF=$β_DF, R_DF = $R_DF")
println("α_Pauli_HF = $α_Pauli_HF, β_Pauli_HF=$β_Pauli_HF, R_Pauli_HF = $R_Pauli_HF")
println("α_Pauli_FB = $α_Pauli_FB, β_Pauli_FB=$β_Pauli_FB, R_Pauli_FB = $R_Pauli_FB")
println("α_AC_HF = $α_AC_HF, β_AC_HF=$β_AC_HF, R_AC_HF = $R_AC_HF")
println("α_AC_FB = $α_AC_FB, β_AC_FB=$β_AC_FB, R_AC_FB = $R_AC_FB")