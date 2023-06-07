function to_OP(F :: F_FRAG)
	# transforms fermionic fragment to operator
	if F.has_coeff
		return F.coeff * fermionic_frag_representer(F.nUs, F.U, F.C, F.N, F.spin_orb, F.TECH)
	else
		return fermionic_frag_representer(F.nUs, F.U, F.C, F.N, F.spin_orb, F.TECH)
	end
end

function F_OP(F :: F_FRAG)
	return to_OP(F)
end

function cartan_1b_to_obt(C :: cartan_1b)
	obt = Diagonal(C.λ)

	return obt
end

function cartan_2b_to_tbt(C :: cartan_2b)
	tbt = zeros(Float64,C.N,C.N,C.N,C.N)

	idx = 1
	for i in 1:C.N
		for j in 1:i
			tbt[i,i,j,j] = tbt[j,j,i,i] = C.λ[idx]
			idx += 1
		end
	end

	return tbt
end

function cartan_2b_num_params(N)
	return Int(N*(N+1)/2)
end

function cartan_SD_num_params(N)
	return N + cartan_2b_num_params(N)
end

function cartan_SD_to_F_OP(C :: cartan_SD)
	obt = Diagonal(C.λ1)
	
	tbt = zeros(Float64,C.N,C.N,C.N,C.N)

	idx = 1
	for i in 1:C.N
		for j in 1:i
			tbt[i,i,j,j] = tbt[j,j,i,i] = C.λ2[idx]
			idx += 1
		end
	end	

	return F_OP(([0], obt, tbt), C.spin_orb)
end

function fermionic_frag_representer(nUs, U, C, N, spin_orb, TECH :: CSA)
	if nUs != 1
		error("Trying to build CSA fragment with $nUs unitaries defined, should be 1!")
	end
	tbt = cartan_2b_to_tbt(C)
	Urot = one_body_unitary(U[1])
	tbt = cartan_tbt_rotation(Urot, tbt, N)

	return F_OP(2,([0.0], [0.0], tbt), [false,false,true], spin_orb, N)
end

function fermionic_frag_representer(nUs, U, C, N, spin_orb, TECH :: CSA_SD)
	if nUs != 1
		error("Trying to build CSA fragment with $nUs unitaries defined, should be 1!")
	end
	F = cartan_SD_to_F_OP(C)

	return F_OP_rotation(U[1], F)
end

function fermionic_frag_representer(nUs, U, C, N, spin_orb, TECH :: DF)
	if nUs != 1
		error("Trying to build DF fragment with $nUs unitaries defined, should be 1!")
	end
	Urot = one_body_unitary(U[1])

	tbt = zeros(Float64,N,N,N,N)
	for i in 1:N
		tbt[i,i,i,i] = C.λ[i]^2
		for j in i+1:N
			tbt[i,i,j,j] = tbt[j,j,i,i] = C.λ[i] * C.λ[j]
		end
	end

	if typeof(Urot[1]) <: Complex
		tbt = cartan_tbt_complex_rotation(Urot, tbt, N)
		println("Complex rotation!")
	else
		tbt = cartan_tbt_rotation(Urot, tbt, N)
	end

	return F_OP(2,([0.0], [0.0], tbt), [false,false,true], spin_orb, N)
end

function fermionic_frag_representer(nUs, U, C, N, spin_orb, TECH :: OBF)
	if nUs != 1
		error("Trying to build OBF fragment with $nUs unitaries defined, should be 1!")
	end
	Urot = one_body_unitary(U[1])

	obt = collect(Diagonal(C.λ))

	if typeof(Urot[1]) <: Complex
		obt = cartan_obt_complex_rotation(Urot, obt, N)
		println("Complex rotation!")
	else
		obt = cartan_obt_rotation(Urot, obt, N)
	end

	return F_OP(1,([0.0], obt), [false,true], spin_orb, N)
end

function fermionic_frag_representer(nUs, U, C, N, spin_orb, TECH :: THC)
	if nUs != 2
		error("Trying to build H_THC fragment with $nUs unitaries defined, should be 2!")
	end
	U1 = one_body_unitary(U[1])
	U2 = one_body_unitary(U[2])

	tbt = zeros(Float64,N,N,N,N)
	@einsum tbt[a,b,c,d] = U1[a,1] * U1[b,1] * U2[c,1] * U2[d,1]
	@einsum tbt[a,b,c,d] += U2[a,1] * U2[b,1] * U1[c,1] * U1[d,1]

	return F_OP(2,([0.0], [0.0], tbt), [false,false,true], spin_orb, N)
end

function fermionic_frag_representer(nUs, U, C, N, spin_orb, TECH :: MTD_CP4)
	if nUs != 4
		error("Trying to build MTD_CP4 fragment with $nUs unitaries defined, should be 4!")
	end
	U1 = one_body_unitary(U[1])
	U2 = one_body_unitary(U[2])
	U3 = one_body_unitary(U[3])
	U4 = one_body_unitary(U[4])

	tbt = zeros(Float64,N,N,N,N)
	@einsum tbt[a,b,c,d] = U1[a,1] * U2[b,1] * U3[c,1] * U4[d,1]
	@einsum tbt[a,b,c,d] += U4[a,1] * U3[b,1] * U2[c,1] * U1[d,1]

	return F_OP(2,([0.0], [0.0], tbt), [false,false,true], spin_orb, N)
end

function fermionic_frag_representer(nUs, U, C, N, spin_orb, TECH :: MTD_PARAFAC)
	if nUs != 4
		error("Trying to build MTD_PARAFAC fragment with $nUs unitaries defined, should be 4!")
	end
	c1 = U[1].cns
	c2 = U[2].cns
	c3 = U[3].cns
	c4 = U[4].cns

	tbt = zeros(Float64,N,N,N,N)
	@einsum tbt[a,b,c,d] = c1[a] * c2[b] * c3[c] * c4[d]
	
	return F_OP(2,([0.0], [0.0], tbt), [false,false,true], spin_orb, N)
end

function obt_to_tbt(obt)
	## transform one-body tensor into two-body tensor
    ## requires obt to be in spin-orbitals!
    
    #println("Transforming one-body tensor into two-body object, small numerical errors might appear...")
	
    Dobt, Uobt = eigen(obt)
    #obt ≡ Uobt * Diagonal(Dobt) * (Uobt')

    n = size(obt)[1]

    tbt = zeros(typeof(Uobt[1]),n,n,n,n)
    for i in 1:n
        tbt[i,i,i,i] = Dobt[i]
    end

    rotated_tbt = zeros(typeof(Uobt[1]),n,n,n,n)

    @einsum rotated_tbt[a,b,c,d] = Uobt[a,l] * conj(Uobt[b,l]) * Uobt[c,l] * conj(Uobt[d,l]) * tbt[l,l,l,l]
    
    return rotated_tbt
end

function tbt_orb_to_so(tbt)
    n = size(tbt)[1]
    n_qubit = 2n

    tbt_so = zeros(2n,2n,2n,2n)
    for i1 in 1:n
        for i2 in 1:n
            for i3 in 1:n
                for i4 in 1:n
                    for a in -1:0
                        for b in -1:0
                            tbt_so[2i1+a,2i2+a,2i3+b,2i4+b] = tbt[i1,i2,i3,i4]
                        end
                    end
                end
            end
        end
    end

    return tbt_so
end

function obt_orb_to_so(obt)
    n = size(obt)[1]
    n_qubit = 2n

    obt_so = zeros(2n,2n)
    for i1 in 1:n
        for i2 in 1:n
            for a in -1:0
                obt_so[2i1+a,2i2+a] = obt[i1,i2]
            end
        end
    end

    return obt_so
end

function mbt_orb_to_so(mbt)
	#transforms orbital many-body fermionic tensor into spin-orbitals
	error("Not implemented!")
end

function to_SO(F :: F_OP)
	#returns operator in spin-orbitals (i.e. spin_orb = true)
	#conversion is not very efficient, shouldn't be used often!
	if F.spin_orb == true
		#println("Fermionic operator is already in spin-orbitals, nothing was done...")
		return F
	elseif F.Nbods == 1
		mbts = (F.mbts[1], obt_orb_to_so(F.mbts[2]))
	elseif F.Nbods == 2
		if F.filled[2]
			mbts = (F.mbts[1], obt_orb_to_so(F.mbts[2]), tbt_orb_to_so(F.mbts[3]))
		else
			mbts = (F.mbts[1], [0.0], tbt_orb_to_so(F.mbts[3]))
		end
	elseif F.Nbods > 2
		M_arr = Array{Float64}[]
		for i in 1:N
			if !F.filled[i]
				push!(M_arr,[0.0])
			else
				push!(M_arr, mbt_orb_to_so(mbt[i]))
			end
		end
		mbts = tuple(M_arr...)
	end

	return F_OP(copy(F.Nbods),mbts,copy(F.filled),true,2*F.N)
end

import Base.+

function +(F1 :: F_OP, F2 :: F_OP)
	Nmax = maximum([F1.Nbods, F2.Nbods])
	filled = ones(Bool, Nmax+1)
	M_arr = Array{Float64}[]

	if F1.spin_orb != F2.spin_orb
		println("Summing spin-orbital with orbital fermionic operators, converting both to spin-orbitals...")
		return +(to_SO(F1), to_SO(F2))
	end

	if F1.N != F2.N
		@show F1.N
		@show F2.N
		error("Trying to sum fermionic operators over different number of orbitals!")
	end

	filled_1 = zeros(Bool,Nmax+1)
	filled_2 = zeros(Bool,Nmax+1)
	filled_1[1:F1.Nbods+1] = F1.filled
	filled_2[1:F2.Nbods+1] = F2.filled

	for i in 1:Nmax+1
		if filled_1[i] == false
			if filled_2[i] == false
				push!(M_arr,[0.0])
				filled[i] = false
			else
				push!(M_arr,F2.mbts[i])
			end
		else
			if filled_2[i] == false
				push!(M_arr,F1.mbts[i])
			else
				push!(M_arr,F1.mbts[i] + F2.mbts[i])
			end
		end
	end

	return F_OP(Nmax, tuple(M_arr...), filled, F1.spin_orb, F1.N)
end

import Base.*

function *(F :: F_OP, a :: Real)
	return F_OP(F.Nbods, a .* F.mbts, F.filled, F.spin_orb, F.N)
end

function *(a :: Real, F :: F_OP)
	return F_OP(F.Nbods, a .* F.mbts, F.filled, F.spin_orb, F.N)
end

import Base.-

function -(F1 :: F_OP, F2 :: F_OP)
	return F1 + (-1 * F2)
end

+(F1 :: F_OP, F2 :: F_FRAG) = F1 + F_OP(F2)
+(F1 :: F_FRAG, F2 :: F_OP) = F_OP(F1) + F2
+(F1 :: F_FRAG, F2 :: F_FRAG) = F_OP(F1) + F_OP(F2)

-(F1 :: F_OP, F2 :: F_FRAG) = F1 - F_OP(F2)
-(F1 :: F_FRAG, F2 :: F_OP) = F_OP(F1) - F2
-(F1 :: F_FRAG, F2 :: F_FRAG) = F_OP(F1) - F_OP(F2)


function F_OP_collect_obt(F :: F_OP)
	#return fermionic operator with one-body tensor collected into two-body tensor
	if F.Nbods < 1
		error("No one-body tensor to collect in fermionic operator!")
	elseif F.Nbods == 1
		F_mid = to_SO(F)
		tbt = obt_to_tbt(F_mid.mbts[2])
		mbts = (F_mid.mbts[1], [0.0], tbt)
		filled = [F_mid.filled[1], false, true]
		Nret = F_mid.N
	else
		if F.spin_orb == false
			F_mid = to_SO(F)
			tbt = F_mid.mbts[3] + obt_to_tbt(F_mid.mbts[2])
			mbts = tuple(F_mid.mbts[1], [0.0], tbt, F_mid.mbts[4:end])
			filled = F_mid.filled
			filled[2] = false
			Nret = F_mid.N
		else
			tbt = F.mbts[3] + obt_to_tbt(F.mbts[2])
			mbts = tuple(F.mbts[1], [0.0], tbt, F.mbts[4:end]...)
			filled = F.filled
			filled[2] = false
			Nret = F.N
		end
	end

	return F_OP(F.Nbods, mbts, filled, true, Nret)
end

import Base.copy

copy(F :: F_OP) = 0 * F + F
copy(F :: F_FRAG) = F_FRAG(F.nUs, F.U, F.TECH, F.C, F.N, F.spin_orb, F.coeff, F.has_coeff)

function CSA_x_to_F_FRAG(x, N, spin_orb, cartan_L = Int(N*(N+1)/2); do_Givens = CSA_GIVENS)
	if do_Givens
		return F_FRAG(1,tuple(givens_real_orbital_rotation(N, x[cartan_L+1:end])),CSA(),cartan_2b(spin_orb,x[1:cartan_L],N),N, spin_orb)
	else
		return F_FRAG(1,tuple(real_orbital_rotation(N, x[cartan_L+1:end])),CSA(),cartan_2b(spin_orb,x[1:cartan_L],N),N, spin_orb)
	end
end

function CSA_SD_x_to_F_FRAG(x, N, spin_orb, cartan_L = Int(N*(N+1)/2); do_Givens = CSA_GIVENS)
	if do_Givens
		return F_FRAG(1,tuple(givens_real_orbital_rotation(N, x[cartan_L+N+1:end])),CSA_SD(),cartan_SD(spin_orb,x[1:N],x[N+1:N+cartan_L],N),N, spin_orb)
	else
		return F_FRAG(1,tuple(real_orbital_rotation(N, x[cartan_L+N+1:end])),CSA_SD(),cartan_SD(spin_orb,x[1:N],x[N+1:N+cartan_L],N),N, spin_orb)
	end
end

function MTD_CP4_x_to_F_FRAG(x, N, spin_orb=false)
	return MTD_PARAFAC_x_to_F_FRAG(x, N, spin_orb)
	#=
	Uvecs = reshape(x[1:end-1], (N,4))
	omega = x[end]

	Us = tuple([restricted_orbital_rotation(N, Uvecs[:,i]) for i in 1:4]...)
	return F_FRAG(4, Us, MTD_CP4(), cartan_m1(), N, spin_orb, omega, true)
	# =#
end

function MTD_PARAFAC_x_to_F_FRAG(x, N, spin_orb=false)
	Uvecs = reshape(x[1:end-1], (N,4))
	omega = x[end]

	Us = tuple([single_orbital_rotation(N, Uvecs[:,i]) for i in 1:4]...)
	return F_FRAG(4, Us, MTD_PARAFAC(), cartan_m1(), N, spin_orb, omega, true)
end


function THC_x_to_F_FRAGS(x, α, N)
	num_ζ = Int(α*(α+1)/2) #ζij is non-zero only for i≥j since operators are hermitized
	ζ = x[1:num_ζ]
	θs = reshape(x[num_ζ+1:end], N-1, α)
	FRAGS = F_FRAG[]

	idx = 1
	for i in 1:α
		Ui = restricted_orbital_rotation(N, θs[:,i])
		for j in 1:i
			Uj = restricted_orbital_rotation(N, θs[:,j])
			frag = F_FRAG(2, (Ui,Uj), THC(), cartan_m1(), N, false, ζ[idx], true)
			push!(FRAGS, frag)
			idx += 1
		end
	end

	return FRAGS
end

function ob_correction(F :: F_OP; return_op=false)
	#returns correction to one-body tensor coming from tbt inside fermionic operator F
	if F.spin_orb
		obt = sum([F.mbts[3][:,:,r,r] for r in 1:F.N])
	else
		obt = 2*sum([F.mbts[3][:,:,r,r] for r in 1:F.N])
	end
	
	if return_op
		return F_OP(([0], obt), F.spin_orb)
	else
		return obt
	end
end

function ob_correction(tbt :: Array{Float64, 4}, spin_orb=false)
	N = size(tbt)[1]
	if spin_orb
		return sum([tbt[:,:,r,r] for r in 1:N])
	else
		return 2*sum([tbt[:,:,r,r] for r in 1:N])
	end
end

function ob_correction(F :: F_FRAG; return_op = false)
	if return_op == false
		return ob_correction(to_OP(F))
	else
		return F_OP(([0],ob_correction(to_OP(F))))
	end
end

function initialize_FRAG(N :: Int64, TECH :: DF; REAL=true)
	nUs = 1
	if REAL
		U = tuple(real_orbital_rotation(N, zeros(Int(N*(N-1)/2))))
	else
		U = tuple(orbital_rotation(N, zeros(Int(N^2))))
	end
	C = cartan_1b(false, zeros(N), N)

	return F_FRAG(nUs, U, DF(), C, N, false, 1, false)
end

function initialize_FRAG(N :: Int64, TECH :: CSA; REAL=true)
	nUs = 1
	if REAL
		U = tuple(real_orbital_rotation(N, zeros(Int(N*(N-1)/2))))
	else
		U = tuple(orbital_rotation(N, zeros(Int(N^2))))
	end
	C = cartan_2b(false, zeros(Int(N*(N+1)/2)), N)

	return F_FRAG(nUs, U, CSA(), C, N, false, 1, false)
end

function initialize_FRAG(N :: Int64, TECH :: THC; REAL=true)
	nUs = 2
	if REAL
		U = tuple(real_orbital_rotation(N, zeros(Int(N*(N-1)/2))),
				  real_orbital_rotation(N, zeros(Int(N*(N-1)/2))))
	else
		U = tuple(orbital_rotation(N, zeros(Int(N^2))),
				  orbital_rotation(N, zeros(Int(N^2))))
	end
	C = cartan_m1()

	return F_FRAG(nUs, U, THC(), C, N, false, 1, true)
end

function to_OBF(F :: F_OP)
	#transforms one-body fermionic operator into one-body fragment
	if F.Nbods != 1
		error("Fermionic operator is not one-body, cannot transform to one-body fragment!")
	end

	D, U = eigen(F.mbts[2])
	C = cartan_1b(F.spin_orb, D, F.N)
	fU = tuple(f_matrix_rotation(F.N, U))

	return F_FRAG(1, fU, OBF(), C, F.N, F.spin_orb, 1, false)
end

function F_OP(tbt :: Array{Float64, 4}, spin_orb=false)
	#by default assume tbt is in spacial orbitals
	return F_OP(([0], [0], tbt), spin_orb)
end

function F_OP(obt :: Array{Float64, 2}, spin_orb=false)
	#by default assume tbt is in spacial orbitals
	return F_OP(([0], obt), spin_orb)
end

function to_OBF(obt :: Array{Float64, 2})
	return to_OBF(F_OP(obt))
end

function F_OP_to_eri(F :: F_OP)
	#transform fermionic operator into electronic repulsion integrals
	obt = copy(F.mbts[2])
	tbt = copy(F.mbts[3])

	obt += sum(tbt[:,k,k,:] for k in 1:F.N)

	return obt, 2*tbt
end

function eri_to_F_OP(obt, tbt, hconst :: Array = [0])
	#transform electronic repulsion integrals into fermionic operator
	N = size(obt)[1]

	mbts = (hconst, obt - sum([0.5*tbt[:,k,k,:] for k in 1:N]), 0.5*tbt)
	
	
	return F_OP(mbts)
end

function eri_to_F_OP(obt, tbt, hconst :: Number)
	return eri_to_F_OP(obt, tbt, [hconst])
end
