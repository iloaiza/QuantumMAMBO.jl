using Pkg
Pkg.activate("./")

Pkg.instantiate()
Pkg.resolve()

# Read the file line by line
function extract_routine(input_str::String)
    starting_idx = findfirst("Starting", input_str)[end]
    routine_idx = findfirst("routine", input_str)[1]
    variable_part = input_str[starting_idx + 2:routine_idx - 2]

    return variable_part
end

function read_estimate_values(filename::String)
    T_estimates = Float64[]
    Q_estimates = Float64[]
    λ_estimates = Float64[]
    method_list = String[]


    is_double = [false]
    #@show filename
    open(filename) do file
        for line in eachline(file)
            if contains(line, "Starting") && contains(line, "routine")
                curr_routine = extract_routine(line)
                if curr_routine == "CP4" || curr_routine == "SVD"
                    push!(method_list, curr_routine * "(T)")
                    push!(method_list, curr_routine * "(Q)")
                    is_double[1] = true
                else
                    push!(method_list, curr_routine)
                    is_double[1] = false
                end
            end

            if contains(line, "QM.quantum_estimate")
                starting_idx = findfirst("= (", line)[end]
                resource = eval(Meta.parse(line[starting_idx:end]))

                #@show line
                #@show is_double
                if is_double[1] == false
                    push!(T_estimates, resource[1][1])
                    push!(Q_estimates, resource[1][2])
                    push!(λ_estimates, resource[2])
                else
                    push!(T_estimates, resource[1][1])
                    push!(T_estimates, resource[2][1])
                    push!(Q_estimates, resource[1][2])
                    push!(Q_estimates, resource[2][2])
                    push!(λ_estimates, resource[3])
                    push!(λ_estimates, resource[3])
                end
            end
        end
    end

    return method_list,T_estimates, Q_estimates, λ_estimates
end

HCHAINS = collect(8:2:30)

@show HCHAINS

METHODS = []
TS = []
QS = []
ΛS = []

for (i,hnum) in enumerate(HCHAINS)
    fname = "TEXT/FULL_h$(hnum).txt"
    method_list,T_estimates, Q_estimates, λ_estimates = read_estimate_values(fname)
    push!(METHODS, method_list)
    push!(TS, T_estimates)
    push!(QS, Q_estimates)
    push!(ΛS, λ_estimates)
end


all_methods = METHODS[1]
num_methods = length(all_methods)
num_chains = length(HCHAINS)


Hardness4plot = zeros(num_methods, num_chains)
Qubs4plot = zeros(Int64, num_methods, num_chains)
isInPlot = zeros(Bool, num_methods, num_chains)


for i in 1:num_chains
    HARDNESS = TS[i] .* ΛS[i]
    for (i_meth,method) in enumerate(METHODS[i])
        meth_index = findfirst(item -> item == method, all_methods)

        if isnothing(meth_index)
            error("METHOD NOT FOUND IN LIST!")
        end

        Hardness4plot[meth_index, i] = HARDNESS[i_meth]
        Qubs4plot[meth_index, i] = QS[i][i_meth]
        isInPlot[meth_index, i] = true
    end
end

@show Hardness4plot
@show Qubs4plot
@show isInPlot

all_methods

@show size(Hardness4plot)
@show size(Qubs4plot)

DF_H = [105389.725227179,191619.59756611817,309665.32546998473,464205.6944783411,667234.9676974929,919658.6217900944,1.2146067898344118e6,1.5712804976699324e6,2.0099310474158293e6,2.5070987912147706e6,3.103895069460392e6,3.770391109251299e6]
DF_Q = [226,266,301,327,587,649,709,771,834,894,950,1010]
Hardness4plot[11,:] = DF_H
Qubs4plot[11,:] = DF_Q

RM_LIST = ["SVD(Q)","CP4(Q)"]

thc_len=6
i_thc=12
#THC_H=[244339.2,400982.4,612437.28,651881.2,863520.32, 1113924.24]
THC_H=[99360, 161467.02, 246468.04, 342777.46, 453263.33, 583693.11]
THC_Q=[303, 309, 348, 542, 550, 554]




using GLM
using DataFrames

function just_fit(X,Y)
    num_pts = length(Y)
    data = DataFrame(X=log10.(X), Y=log10.(Y))
    
    ols = lm(@formula(Y ~ X), data)
    A,B = GLM.coef(ols)
    
    return A,B
end

function do_line(A,B)
    return 10^A * HCHAINS.^B
end

#using Pkg
#Pkg.add("LaTexStrings")
#using LaTexStrings

using Plots
gr()
using Plots.PlotMeasures
using ColorSchemes
FONT=font(40)
SIZE=[2150,980]
L_MARG=[15mm 0mm]
B_MARG=[10mm 0mm]
COLOR_LIST = [:black,:red,:blue,:green,:grey,:red,:magenta,:grey,:saddlebrown,:indigo,:brown,:darkgreen]
MARK_LIST = [:circle,:rtriangle,:ltriangle,:dtriangle,:utriangle,:hexagon,:pentagon,:star4,:star7,:star5,:square,:heptagon]
MARKER_SIZE = 20

Phardness = plot()
plot!(xlabel="Number of atoms in chain\n\n",ylabel="Hardness measure", xscale=:log, yscale=:log);
plot!(xtickfont = FONT,xguidefont=FONT,ytickfont = FONT,yguidefont=FONT,size=SIZE,legendfont=FONT,
                    left_margin=L_MARG,bottom_margin=B_MARG, legend=false,xticks=([10,20,30],["10","20","30"]),
                    ylims=(1e4,2e10),yticks=([1e5,1e6,1e7,1e8,1e9,1e10]))

@show all_methods
for (i_meth, method) in enumerate(all_methods)
    if (method in RM_LIST) == false
        meth_len = sum(isInPlot[i_meth, :])

        Hmeth = Hardness4plot[i_meth, 1:meth_len]

        if method == "orbital-optimized sparse"
            NAME = "OO-Pauli"
        elseif method == "orbital-optimized AC"
            NAME = "OO-AC"
        elseif method == "sparse"
            NAME = "Pauli"
        elseif method == "DF_openfermion"
            NAME = "Optimized DF"
        elseif method == "SVD(T)"
            NAME = "SVD"
        elseif method == "CP4(T)"
            NAME = "CP4"
        else
            NAME = method
        end
        
        if NAME == "CP4"
            #NAME = L"$L^4$-CP4"
            A,B = just_fit(HCHAINS[6:meth_len], Hmeth[6:end])
            scatter!(HCHAINS[6:meth_len], Hmeth[6:end], label=false, marker=(MARKER_SIZE,COLOR_LIST[i_meth],MARK_LIST[i_meth],stroke(0)))
        elseif NAME == "SVD"
            #NAME = L"$L^4$-SVD"
            A,B = just_fit(HCHAINS[1:meth_len], Hmeth)
            scatter!(HCHAINS[1:meth_len], Hmeth, label=false, marker=(MARKER_SIZE,COLOR_LIST[i_meth],MARK_LIST[i_meth],stroke(0)))
        elseif NAME == "MPS"
            #NAME = L"$L^4$-MPS"
            A,B = just_fit(HCHAINS[1:meth_len], Hmeth)
            scatter!(HCHAINS[1:meth_len], Hmeth, label=false, marker=(MARKER_SIZE,COLOR_LIST[i_meth],MARK_LIST[i_meth],stroke(0)))
        else
            A,B = just_fit(HCHAINS[1:meth_len], Hmeth)
            scatter!(HCHAINS[1:meth_len], Hmeth, label=false, marker=(MARKER_SIZE,COLOR_LIST[i_meth],MARK_LIST[i_meth],stroke(0)))
        end
        plot!(HCHAINS, do_line(A,B), label=false, color=COLOR_LIST[i_meth], linewidth=3)
        
        @show NAME
        @show A,B
    end
end
C,D=just_fit(HCHAINS[1:thc_len], THC_H)
@show C,D
scatter!(HCHAINS[1:thc_len], THC_H, label=false, marker=(MARKER_SIZE,COLOR_LIST[i_thc],MARK_LIST[i_thc],stroke(0)),size=SIZE)
plot!(HCHAINS, do_line(C,D), label=false, color=COLOR_LIST[i_thc], linewidth=3)
savefig("HARDNESS.png")

Phardness



using Plots
gr()
using Plots.PlotMeasures
using ColorSchemes
FONT=font(40)
SIZE=[2150,780*2]
L_MARG=[15mm 0mm]
B_MARG=[10mm 0mm]
COLOR_LIST = [:black,:red,:blue,:green,:grey,:red,:magenta,:grey,:saddlebrown,:indigo,:brown,:darkgreen]
MARK_LIST = [:circle,:rtriangle,:ltriangle,:dtriangle,:utriangle,:hexagon,:pentagon,:star4,:star7,:star5,:square,:heptagon]
MARKER_SIZE = 20

Pqub = plot()
plot!(xlabel="Number of atoms in chain \n\n",ylabel="Number of qubits", xscale=:log, yscale=:log);
plot!(xtickfont = FONT,xguidefont=FONT,ytickfont = FONT,yguidefont=FONT,size=SIZE,legendfont=FONT,
                    left_margin=L_MARG,bottom_margin=B_MARG, legend=:outerbottom, legendcolumns=3, legendfontsize=30, xticks=([10,20,30],["10","20","30"]));
@show all_methods
for (i_meth, method) in enumerate(all_methods)
    if (method in RM_LIST) == false
        meth_len = sum(isInPlot[i_meth, :])
        @show method, meth_len

        Qmeth = Qubs4plot[i_meth, 1:meth_len]

        if method == "orbital-optimized sparse"
            NAME = "OO-Pauli"
        elseif method == "orbital-optimized AC"
            NAME = "OO-AC"
        elseif method == "sparse"
            NAME = "Pauli"
        elseif method == "DF_openfermion"
            NAME = "Optimized DF"
        elseif method == "SVD(T)"
            NAME = "SVD"
        elseif method == "CP4(T)"
            NAME = "CP4"
        else
            NAME = method
        end
        
        if NAME == "CP4"
            A,B = just_fit(HCHAINS[6:meth_len], Qmeth[6:end])
        else
            A,B = just_fit(HCHAINS[1:meth_len], Qmeth)
        end
        plot!(HCHAINS, do_line(A,B), label=false, color=COLOR_LIST[i_meth], linewidth=3)
        
        @show NAME
        @show A,B
        scatter!(HCHAINS[1:meth_len], Qmeth, label=NAME, marker=(MARKER_SIZE,COLOR_LIST[i_meth],MARK_LIST[i_meth],stroke(0)))
    end
end
E,F=just_fit(HCHAINS[1:thc_len], THC_Q)
@show E,F
scatter!(HCHAINS[1:thc_len], THC_Q, label="THC", marker=(MARKER_SIZE,COLOR_LIST[i_thc],MARK_LIST[i_thc],stroke(0)),size=SIZE)
plot!(HCHAINS, do_line(E,F), label=false, color=COLOR_LIST[i_thc], linewidth=3)

savefig("QUBITS.png")

Pqub

    
#=Pqub = plot()
plot!(xlabel="Number of atoms in chain",ylabel="Number of qubits", xscale=:log, yscale=:log);
plot!(xtickfont = FONT,xguidefont=FONT,ytickfont = FONT,yguidefont=FONT,size=SIZE,legendfont=FONT,
                    left_margin=L_MARG,bottom_margin=B_MARG, legend=:topleft);
for (i_meth, method) in enumerate(all_methods)
    if (method in RM_LIST)!
        meth_len = sum(isInPlot[i_meth, :])

        Qmeth = Qubs4plot[i_meth, 1:meth_len]

        if method == "orbital-optimized sparse"
            NAME = "OO-Pauli"
        elseif method == "orbital-optimized AC"
            NAME = "OO-AC"
        elseif method == "sparse"
            NAME = "Pauli"
        elseif method == "DF_openfermion"
            NAME = "Optimized DF"
        elseif method == "SVD(T)"
            NAME = "SVD"
        elseif method == "CP4(T)"
            NAME = "CP4"
        else
            NAME = method
        end

        scatter!(HCHAINS[1:meth_len], Qmeth, label=NAME, marker=(MARKER_SIZE,COLOR_LIST[i_meth],MARK_LIST[i_meth],stroke(0)))
    end
end
savefig("QUBITS.pdf")=#

using GLM
function just_fit(OR_Pauli)
  OR_fit = lm(reshape(λQs, length(λQs), 1), OR_Pauli)
  OR_coef = GLM.coef(OR_fit)
  ERR = stderror(OR_fit)
  return OR_coef[1], ERR[1]
end

function linear_fit(OR_Pauli, num)
    OR_fit = lm(reshape(λQs, length(λQs), 1), OR_Pauli)
    OR_coef = GLM.coef(OR_fit)
    ERR = stderror(OR_fit)
    plot!(λQs, λQs .* OR_coef, label=false, line=(LWIDTH,COLOR_LIST[num]))
    return OR_coef[1], ERR[1]
end
