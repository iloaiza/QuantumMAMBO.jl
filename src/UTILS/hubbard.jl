function build_1D_hubbard(U, J, nsites)
	#returns 1D Hubbardd hamiltonian with near-neighbout coupling -J and site potential U
	obt = zeros(nsites,nsites)
	for i in 1:nsites-1
		obt[i,i+1] -= J
		obt[i+1,i] -= J
	end

	obt = obt_orb_to_so(obt)

	tbt = zeros(2*nsites,2*nsites,2*nsites,2*nsites)
	for i in 1:nsites
		iup = 2i-1
		idown = 2i

		tbt[iup,iup,idown,idown] += U/2
		tbt[idown,idown,iup,iup] += U/2
	end

	return F_OP(([0],obt,tbt), true)
end