import juliacall
from juliacall import Main as jl

#""" USED FOR PACKAGE-BASED QuantumMAMBO INSTALLATION
jl.seval('import Pkg')
jl.seval('Pkg.add("QuantumMAMBO")')
jl.seval("using QuantumMAMBO")
#"""

""" USED FOR LOCAL QuantumMAMBO INSTALLATION
jl.seval('include("src/QuantumMAMBO.jl")')
jl.seval("using .QuantumMAMBO")
"""

mambo = jl.QuantumMAMBO
from openfermion import FermionOperator

def CSA_greedy(H:FermionOperator, a_max, do_singles = False, ret_mambo = False, **kwargs):
	#ret_mambo determines whether returned array of fragments is of QuantumMAMBO F_OP objects or Openfermion array of FermionOperators
	#do_singles=True will include one-electron operators during optimization, otherwise only consider two-electron tensor CSA decomposition
	Hmambo = mambo.OF_to_F_OP(H)
	if do_singles:
		FRAGS = mambo.CSA_SD_greedy_decomposition(Hmambo, a_max, **kwargs)
	else:
		FRAGS = mambo.CSA_greedy_decomposition(Hmambo, a_max, **kwargs)	

	if ret_mambo:
		return FRAGS
	else:
		py_frags = []
		for frag in FRAGS:
			py_frags += [mambo.to_OF(frag)]
		return py_frags

def BLISS(H:FermionOperator, num_elecs, do_T = True, ret_mambo = False, verbose=True, do_save=False, **kwargs):
	#ret_mambo determines whether returned operator is QuantumMAMBO F_OP object or Openfermion FermionOperator
	#do_T=True means the full BLISS shift is performed (i.e. H_T), False means only H_S is calculated 
	#these correspond to Eqs. 6 and 8 of Ref.[2] in the README
	Hmambo = mambo.OF_to_F_OP(H)
	if do_T:
		H_new, _ = mambo.bliss_linprog(Hmambo, num_elecs)
	else:
		H_new, shifts = mambo.symmetry_treatment(Hmambo, verbose=verbose, SAVELOAD=do_save)

	if ret_mambo:
		return H_new
	else:
		return mambo.to_OF(H_new)

def do_decomp(H:FermionOperator, decomp):
	#returns an array of unitaries corresponding to the LCU decomposition, along an array of LCU coefficients
	#accepts decomp = {"Pauli", "AC", "DF"} for Pauli decomposition, anticommuting, and double-factorization, respectively
	Hmambo = mambo.OF_to_F_OP(H)
	UNITARIES = []
	COEFFS = []

	if decomp == "Pauli":
		Hq = mambo.Q_OP(Hmambo)
		num_paulis = Hq.n_paulis
		for i in range(num_paulis):
			pw = mambo.copy(Hq.paulis[i])
			COEFFS += [pw.coeff]
			pw.coeff = 1
			UNITARIES += [mambo.to_OF(pw)]
	elif decomp == "AC":
		Hq = mambo.Q_OP(Hmambo)
		COEFFS, OPS = mambo.AC_group(Hq, ret_ops = True)
		for i in range(len(OPS)):
			UNITARIES += [mambo.to_OF(OPS[i]) / COEFFS[i]]
	elif decomp == "DF":
		print("Still not implemented!")
		DF_FRAGS = mambo.DF_decomposition(Hmambo)
	else:
		print("{} decomposition method not yet defined!".format(decomp))

	return UNITARIES, COEFFS

