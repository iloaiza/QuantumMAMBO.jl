from openfermion.resource_estimates.molecule import (cas_to_pyscf, pyscf_to_cas)

def mf_eri(pyscf_mf):
	_, pyscf_mf = cas_to_pyscf(*pyscf_to_cas(pyscf_mf))
	return pyscf_mf