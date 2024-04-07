# Test Module
using QuantumMAMBO:DATAFOLDER, SAVELOAD_HAM, RUN_L1, symmetry_treatment, INTERACTION, bliss_optimizer, quadratic_bliss, bliss_linprog, quadratic_bliss_optimizer,F_OP_converter,F_OP_compress,PAULI_L1, F_OP
using HDF5
using Test

TESTFOLDER = @__DIR__

TESTFOLDER = TESTFOLDER * "/"
molecules=["h2","h3","ch3","beh2","lih","h2o","nh3","ch2","c2h2","ru","femoco","ts_ru"]
compression_begin=10
#compression=[false,false,true,false,true,true,false,false,false,false,false,false]

@testset "BLISS" begin
	for (idx,mol_name) in enumerate(molecules)
		FILENAME = TESTFOLDER*mol_name*"_test.h5"
		fid = h5open(FILENAME, "cw")
		if haskey(fid, "BLISS")
			ham=fid["BLISS_HAM"]
			H_bliss_test=F_OP((read(ham,"h_const"),read(ham,"obt"),read(ham,"tbt")))
			close(fid)
		end
		
		
		#H_test,η_test = SAVELOAD_HAM(mol_name, FILENAME)
		FILENAME = TESTFOLDER*mol_name
		H,η = SAVELOAD_HAM(mol_name, FILENAME)
		if idx>=compression_begin
			H=F_OP_compress(H)
		end
		H_bliss,_= bliss_linprog(H, η, SAVELOAD = false, SAVENAME=FILENAME*"_BLISS.h5")
		#H_bliss_test,_=bliss_linprog(H_test, η_test)
		for i=1:3
		
			H_bliss.mbts[i][:]=round.(H_bliss.mbts[i],digits=10)
			H_bliss_test.mbts[i][:]=round.(H_bliss_test.mbts[i],digits=10)
		end
		check= H_bliss.mbts[:] .== H_bliss_test.mbts[:]
		check_prod=check[1] && check[2] && check[3] 
		@test check_prod == true
	end
end

molecules=["h2","lih","beh2","h3","ch3"]

@testset "GRAD_COMPARE" begin
	for (idx, mol_name) in enumerate(molecules)
		FILENAME = TESTFOLDER*mol_name
		H,η = SAVELOAD_HAM(mol_name, FILENAME)
		H_bliss,_=bliss_linprog(H, η, SAVELOAD = false, SAVENAME=FILENAME*"_BLISS.h5")
		H_grad_bliss=quadratic_bliss_optimizer(H, η)
		
		
		check=round(PAULI_L1(H_bliss),digits=2) <= round(PAULI_L1(H_grad_bliss), digits=2)
		@test check== true
	end
end
