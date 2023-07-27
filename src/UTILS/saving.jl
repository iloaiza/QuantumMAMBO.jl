#saving module, very practical :)
#use @saving "name" var1 var2 to save in DATAFOLDER ("./" unless defined elsewhere) a "name.h5" file (with movement to .old if preexisting) "var1", "var2", ... values
#use @loading "name" for automatic loading of all saved variables

global SAVING_LOADED = true

using HDF5
if !(@isdefined(DATAFOLDER))
	DATAFOLDER = @__DIR__
	DATAFOLDER = DATAFOLDER * "/../../SAVED/"
end
if @isdefined myid
	if myid() == 1
		println("Loading and saving in $DATAFOLDER")
	end
else
	println("Loading and saving in $DATAFOLDER")
end

export DATAFOLDER

function stringSymbolSeparator(params,numparams=length(params))
	paramsNames = String[]
	for i in 1:numparams
		current = "$(params[i])"
		push!(paramsNames,current)
	end

	return paramsNames
end

function oldfile(funcname::String)
	#frees filename for saving, moves all old files to old+1
	filename = DATAFOLDER*funcname*".h5"
	if isfile(filename)
		oldname = filename*".old"
		if isfile(oldname)
			oldcount = 1
			while isfile(oldname*"$oldcount")
				oldcount += 1
			end
			oldcount -= 1
			for (i,oldnum) in enumerate(oldcount:-1:1)
				run(`mv -f $oldname$oldnum $oldname$(oldnum+1)`)
				println("Moved file $oldname$oldnum to $oldname$(oldnum+1)")
			end
			run(`mv -f $oldname $(oldname)1`)
			println("Moved file $oldname to $(oldname)1")
		end
		run(`mv -f $filename $filename.old`)
		println("Moved file $filename to $oldname")
	end
end

function file_space(funcname::String)
	#erases filename to make space for save
	filename = DATAFOLDER*funcname*".h5"
	if isfile(filename)
		run(`rm $filename`)
		#println("Removed file $filename for saving new file on top...")
	end
end

macro saving(funcname::String,params...)
	quote
		oldfile($funcname)
		h5write(DATAFOLDER*$funcname*".h5","numparams",length($params))
		h5write(DATAFOLDER*$funcname*".h5","paramsNames",stringSymbolSeparator($params))
		for (n,name) in enumerate(stringSymbolSeparator($params))
			h5write(DATAFOLDER*$funcname*".h5",name,eval($(esc(params))[n]))
			println("Saved $($(params)[n])")	
		end
	end
end

macro overwriting(funcname::String,params...)
	quote
		h5write(DATAFOLDER*$funcname*".h5","numparams",length($params))
		h5write(DATAFOLDER*$funcname*".h5","paramsNames",stringSymbolSeparator($params))
		for (n,name) in enumerate(stringSymbolSeparator($params))
			h5write(DATAFOLDER*$funcname*".h5",name,eval($(esc(params))[n]))
			println("Saved $($(params)[n])")	
		end
	end
end

function saving_xK(funcname::String,x0,K0)
	oldfile(funcname)
	h5write(DATAFOLDER*funcname*".h5","numparams",2)
	h5write(DATAFOLDER*funcname*".h5","paramsNames",["x0","K0"])
	h5write(DATAFOLDER*funcname*".h5","x0",x0)
	h5write(DATAFOLDER*funcname*".h5","K0",K0)
	#println("overwrite_xK debug message: Saved x0 and K0 in $funcname")
end

function overwrite_xK(funcname::String,x0,K0)
	file_space(funcname)
	h5write(DATAFOLDER*funcname*".h5","numparams",2)
	h5write(DATAFOLDER*funcname*".h5","paramsNames",["x0","K0"])
	h5write(DATAFOLDER*funcname*".h5","x0",x0)
	h5write(DATAFOLDER*funcname*".h5","K0",K0)
	#println("overwrite_xK debug message: Saved x0 and K0 in $funcname")
end

function overwrite(funcname::String,x)
	file_space(funcname)
	h5write(DATAFOLDER*funcname*".h5","numparams",1)
	h5write(DATAFOLDER*funcname*".h5","paramsNames",["x"])
	h5write(DATAFOLDER*funcname*".h5", "x", x)
end

function save(funcname::String,x)
	oldfile(funcname)
	h5write(DATAFOLDER*funcname*".h5","numparams",1)
	h5write(DATAFOLDER*funcname*".h5","paramsNames",["x"])
	h5write(DATAFOLDER*funcname*".h5", "x", x)
end

function load(funcname::String)
	filename = DATAFOLDER*funcname*".h5"
	if isfile(filename)
		return h5read(DATAFOLDER*funcname*".h5","x")
	else
		return false
	end
end

macro loading(funcname::String,addname=false)
	h5name = DATAFOLDER*funcname*".h5"
	numparams = h5read(h5name,"numparams")
	paramsNames = h5read(h5name,"paramsNames")
	if addname
		paramsNames .= funcname .* paramsNames
	end

	for (n,name) in enumerate(paramsNames)
		nstring = """$name = h5read("$h5name","$name")"""
		eval(Meta.parse(nstring))
	end

	println("Loaded $numparams variables from function named $funcname with names ($paramsNames)")
end

function loading(funcname::String,addname=false)
	h5name = DATAFOLDER*funcname*".h5"
	paramsNames = h5read(h5name,"paramsNames")
	if addname
		paramsNames .= funcname .* paramsNames
	end

	retArr = Any[]

	for (n,name) in enumerate(paramsNames)
		nstring = """$name = h5read("$h5name","$name")"""
		eval(Meta.parse(nstring))
		push!(retArr,eval(Meta.parse(name)))
	end

	println("Loaded $(length(paramsNames)) variables from function named $funcname with names ($paramsNames)")

	return paramsNames,retArr
end

function save_frag(F :: F_FRAG, fname = DATAFOLDER*"fragment.h5", frag_name = "CSA_SD")
	if F.TECH != CSA_SD()
		println("Saving fragment that is not CSA_SD, not fully implemented...")
		return save_frag(to_CSA_SD(F), fname, frag_name)
	end
	fid = h5open(fname, "cw")

	G = create_group(fid, frag_name)
	G["TECH"] = "CSA_SD"
	u_string = split("$(typeof(F.U[1]))", ".")[end]
	G["u_type"] = u_string
	if u_string == "f_matrix_rotation"
		G["u_params"] = F.U[1].mat
	else
		G["u_params"] = F.U[1].θs
	end
	G["N"] = F.N
	G["C_lambda_1"] = F.C.λ1
	G["C_lambda_2"] = F.C.λ2
	G["spin_orb"] = F.spin_orb
	G["coeff"] = F.coeff
	G["has_coeff"] = F.has_coeff

	print("Saved fragment in file $fname under group $frag_name")

	close(fid)
end

function load_frag(fname, frag_name = "CSA_SD")
	fid = h5open(fname, "cw")

	G = fid[frag_name]
	if read(G["TECH"]) != "CSA_SD"
		error("""Trying to load fragment of type $(G["TECH"]), not implemented!""")
	end
	N = read(G["N"])
	nUs = 1
	u_params = read(G["u_params"])
	u_string = read(G["u_type"])
	u1 = eval(Meta.parse("$u_string($(N),$(u_params))"))
	U = tuple(u1)
	spin_orb = read(G["spin_orb"])
	C = cartan_SD(spin_orb, read(G["C_lambda_1"]), read(G["C_lambda_2"]), N)
	coeff = read(G["coeff"])
	has_coeff = read(G["has_coeff"])

	close(fid)

	return F_FRAG(1, U, CSA_SD(), C, N, spin_orb, coeff, has_coeff)
end
