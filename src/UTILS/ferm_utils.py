import numpy as np
import scipy as sp
import itertools
from openfermion import FermionOperator, hermitian_conjugated, normal_ordered, QubitOperator
import openfermion as of

def get_one_body_terms(H):
    '''
    Return the one body terms in H (plus constant term)
    '''
    one_body = FermionOperator.zero()
    for fw, val in H.terms.items():
        if len(fw) <= 2:
            one_body += FermionOperator(fw, val)
    return one_body

def get_pure_one_body_terms(H):
    '''
    Return just one body terms in H
    '''
    one_body = FermionOperator.zero()
    for fw, val in H.terms.items():
        if len(fw) == 2:
            one_body += FermionOperator(fw, val)
    return one_body

def get_pure_two_body_terms(H):
    '''
    Return just two body terms in H
    '''
    two_body = FermionOperator.zero()
    for fw, val in H.terms.items():
        if len(fw) == 4:
            two_body += FermionOperator(fw, val)
    return two_body

def get_hf(n_spinorb, nelec):
    '''
    Return the hf wfs given number of electrons and orbitals
    '''
    hf = np.zeros((n_spinorb, 1))
    hf[:nelec] = 1
    return np.flip(hf)

def get_creation_op(on):
    '''
    Get the corresponding creation operators based on ON vec 
    e.g. [0, 0, 1] = |001> -> a^+_0 
    '''
    on_flip = np.flip(on)
    op = FermionOperator.identity()
    for i in range(len(on_flip)):
        if on_flip[i] == 1:
            op = FermionOperator(term=(i, 1)) * op
    return op

def get_full_onvec(on):
    '''
    Return the onvec in full 2^n space as basis vector
    '''
    on = np.flip(on)
    n = len(on)
    idx = 0
    for i in range(len(on)):
        if on[i] == 1:
           idx += 2 ** i 
    vec = np.zeros((2**n, 1))
    vec[idx, 0] = 1
    return vec

def braket(onl, onr, H):
    '''
    Obtain the value of <l|H|r>
    '''
    def fermionic_action(fw, on):
        '''
        Acting fermionic word to on vector . 
        Return new on with phase.
        Example: ((1, 1), (0, 0)) |00> = 0 
        '''
        on = np.flip(on)
        phase = 1 
        for fs in reversed(fw):
            if fs[1] == on[fs[0]]:
                return on, 0
            else:
                on[fs[0]] = fs[1]
                phase *= (-1)**(sum(on[fs[0]+1:]))
        return np.flip(on), phase
    e = 0
    for fw, val in H.terms.items():
        cur_onr, phase = fermionic_action(fw, np.copy(onr))
        if phase != 0 and all(onl == cur_onr):
            e += val * phase
    return e

def deprecated_braket(onl, onr, H):
    '''
    Obtain the value of <l|H|r>
    '''
    lop = hermitian_conjugated(get_creation_op(onl))
    rop = get_creation_op(onr)
    op = lop * H * rop
    op = normal_ordered(op)
    return op.constant

def get_on_vec(idx, orb):
    '''
    Get the on vector form based on index and orb 
    e.g. |000> then |001> then |010> 
    '''
    on = np.zeros(orb)
    for i in range(orb):
        occ = idx % 2
        idx = idx // 2
        on[i] = occ
    return np.flip(on)
    
def get_on_idx(on):
    '''
    Get the index based on on vec 
    e.g. |000> -> 0. |001> -> 1. then |010> -> 2. 
    '''
    idx = 0
    on = np.flip(on)
    for i in range(len(on)):
        if on[i] == 1:
            idx += 2**i
    return idx

def get_ferm_basis(norb, nelec):
    '''
    Obtain the basis given number of orbitals and electrons
    '''
    choice = list(itertools.combinations([i for i in range(norb)], nelec))
    basis = []

    for indices in choice:
        curbasis = np.zeros((norb, 1))
        for idx in indices:
            curbasis[idx] = 1
        basis.append(curbasis)
    return basis

def get_fermionic_matrix(H : FermionOperator, n=None, nelec=None):
    '''
    Obtain the matrix form of fermionic operators 
    '''
    if n is None:
        n = get_spin_orbitals(H)

    if nelec is None:
        basis = []
        size = 2**n
        for i in range(size):
            basis.append(get_on_vec(i, n))
        matrix = np.zeros((size, size), np.complex128)
    else:
        basis = get_ferm_basis(n, nelec)
        size = len(basis)
        matrix = np.zeros((size, size), np.complex128)

    for i, ion in enumerate(basis):
        for j, jon in enumerate(basis):
            matrix[i, j] = braket(ion, jon, H)
    return matrix

def get_ferm_op_one(obt, spin_orb):
    '''
    Return the corresponding fermionic operators based on one body tensor
    '''
    n = obt.shape[0]
    op = FermionOperator.zero()
    for i in range(n):
        for j in range(n):
            if not spin_orb:
                for a in range(2):
                    op += FermionOperator(
                        term = (
                            (2*i+a, 1), (2*j+a, 0)
                        ), coefficient=obt[i, j]
                    )
            else:
                op += FermionOperator(
                    term = (
                        (i, 1), (j, 0)
                    ), coefficient=obt[i, j]
                )
    return op 

def get_ferm_op_two(tbt, spin_orb):
    '''
    Return the corresponding fermionic operators based on tbt (two body tensor)
    This tensor can index over spin-orbtals or orbitals
    '''
    n = tbt.shape[0]
    op = FermionOperator.zero()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n): 
                    if not spin_orb:
                        for a in range(2):
                            for b in range(2):
                                op += FermionOperator(
                                    term = (
                                        (2*i+a, 1), (2*j+a, 0),
                                        (2*k+b, 1), (2*l+b, 0)
                                    ), coefficient=tbt[i, j, k, l]
                                )
                    else:
                        op += FermionOperator(
                            term=(
                                (i, 1), (j, 0),
                                (k, 1), (l, 0)
                            ), coefficient=tbt[i, j, k, l]
                        )
    return op

def get_ferm_op(tsr, spin_orb=False):
    '''
    Return the corresponding fermionic operators based on the tensor
    This tensor can index over spin-orbtals or orbitals
    '''
    if len(tsr.shape) == 4:
        return get_ferm_op_two(tsr, spin_orb)
    elif len(tsr.shape) == 2:
        return get_ferm_op_one(tsr, spin_orb)

def get_spin_orbitals(H : FermionOperator):
    '''
    Obtain the number of spin orbitals of H
    '''
    n = -1 
    for term, val in H.terms.items():
        if len(term) == 4:
            n = max([
                n, term[0][0], term[1][0],
                term[2][0], term[3][0]
            ])
        elif len(term) == 2:
            n = max([
                n, term[0][0], term[1][0]])
    n += 1 
    return n

def fci2hf(fci, n=None, tiny=1e-6):
    '''
    Convect fci solutions to pairs of HF. [(hf_i, coeff_i), ...]
    '''
    if n is None:
        n = int(np.log2(len(fci)))
    hf_pairs = []
    
    norm = 0
    for idx, val in enumerate(fci):
        if abs(val) > tiny:
            norm += np.abs(val) ** 2
            hf_pairs.append([get_on_vec(idx, n), val])
    
    scale = norm ** (1/2)
    for i in range(len(hf_pairs)):
        hf_pairs[i][1] = hf_pairs[i][1] / scale
    return hf_pairs

def hfp_braket(hfp, Hf):
    '''
    Given Hermitian Operator Hf and hf pair [(hf_i, coeff_i), ...]
    Obtain <Hf> 
    '''
    e = 0
    nhf = len(hfp)
    for i in range(nhf):
        for j in range(i, nhf):
            cur_val = np.conj(hfp[i][1]) * hfp[j][1] * braket(hfp[i][0], hfp[j][0], Hf)
            if abs(np.imag(cur_val)) > 1e-8:
                print("i: {}. j: {}".format(i, j))
                print("ival: {}. jval: {}".format(hfp[i][1], hfp[j][1]))
            if i == j:
                e += cur_val
            else:
                e += 2 * np.real(cur_val)
    return e

def get_two_body_tensor(H : FermionOperator, n = None):
    '''
    Obtain the 4-rank tensor that represents two body interaction in H. 
    In physics ordering a^ a^ a a 
    '''
    # number of spin orbitals 
    if n is None:
        n = get_spin_orbitals(H)

    tbt = np.zeros((n, n, n, n))
    for term, val in H.terms.items():
        if len(term) == 4:
            tbt[
                term[0][0], term[1][0],
                term[2][0], term[3][0]
            ] = val
    return tbt

def get_obt(H : FermionOperator, n = None, spin_orb=False, tiny=1e-12):
    '''
    Obtain the 2-rank tensor that represents one body interaction in H. 
    In addition, simplify tensor assuming symmetry between alpha/beta coefficients
    '''
    # getting N^2 phy_tbt and then (N/2)^2 chem_tbt 
    if n is None:
        n = get_spin_orbitals(H)
    
    obt = np.zeros((n,n))
    for term, val in H.terms.items():
        if len(term) == 2:
            if term[0][1] == 1 and term[1][1] == 0:
                obt[term[0][0], term[1][0]] = val.real
            elif term[1][1] == 1 and term[0][1] == 0:
                obt[term[1][0], term[0][0]] = -val.real
            else:
                print("Warning, one-body operator has double creation/annihilation operators!")
                quit()

    if spin_orb:
        return obt

    # Spin-orbital to orbital 
    n_orb = obt.shape[0]
    n_orb = n_orb // 2

    obt_red_uu = np.zeros((n_orb, n_orb))
    obt_red_dd = np.zeros((n_orb, n_orb))
    obt_red_ud = np.zeros((n_orb, n_orb))
    obt_red_du = np.zeros((n_orb, n_orb))
    for i in range(n_orb):
        for j in range(n_orb):
            obt_red_uu[i,j] = obt[2*i, 2*j]
            obt_red_dd[i,j] = obt[2*i+1, 2*j+1]
            obt_red_ud = obt[2*i, 2*j+1]
            obt_red_du = obt[2*i+1, 2*j]

    if np.sum(np.abs(obt_red_du)) + np.sum(np.abs(obt_red_ud)) != 0:
        print("Warning, operator to one-body transformation ran with spin_orb=false, but spin-orbit couplings are not 0!")
    if np.sum(np.abs(obt_red_uu - obt_red_dd)) > tiny:
        print("Warning, operator to one-body transformation ran with spin_orb=false, but isn't symmetric to spin-flips")
        print("obt_uu - obt_dd = {}".format(obt_red_uu - obt_red_dd))

    obt = (obt_red_uu + obt_red_dd) / 2

    return obt

def get_chemist_tbt(H : FermionOperator, n = None, spin_orb=False):
    '''
    Obtain the 4-rank tensor that represents two body interaction in H. 
    In chemist ordering a^ a a^ a. 
    In addition, simplify tensor assuming symmetry between alpha/beta coefficients
    '''
    # getting N^4 phy_tbt and then (N/2)^4 chem_tbt 
    phy_tbt = get_two_body_tensor(H, n)
    chem_tbt = np.transpose(phy_tbt, [0, 3, 1, 2])

    if spin_orb:
        return chem_tbt

    # Spin-orbital to orbital 
    n_orb = phy_tbt.shape[0]
    n_orb = n_orb // 2
    alpha_indices = list(range(0, n_orb * 2, 2))
    beta_indices = list(range(1, n_orb * 2, 2))
    chem_tbt = chem_tbt[
        np.ix_(alpha_indices, alpha_indices,
                    beta_indices, beta_indices)]

    return chem_tbt

def get_chemist_obt_correction(H : FermionOperator, n = None, spin_orb=False):
    '''
    Obtain the 2-rank tensor that represents one body interaction correction to H from changing between physicist and chemist ordering 
    In chemist ordering a^ a a^ a. 
    In addition, simplify tensor assuming symmetry between alpha/beta coefficients
    '''
    # getting N^4 phy_tbt and then (N/2)^4 chem_tbt 
    phy_tbt = get_two_body_tensor(H, n)
    n_orb = phy_tbt.shape[0]
    chem_obt = np.zeros((n_orb,n_orb))
    for i in range(n_orb):
        for j in range(n_orb):
            for k in range(n_orb):
                chem_obt[i,j] -= phy_tbt[i,k,j,k]

    if spin_orb:
        return chem_obt
    else:
        println("Can't get one body correction for spin_orb = False")
        quit()

    return chem_obt

def separate_diagonal_tbt(chem_tbt):
    '''
    Separate the terms representing n_p n_q terms 
    '''
    n = chem_tbt.shape[0]
    diag_tbt = np.zeros((n, n, n, n))
    ndia_tbt = chem_tbt.copy()

    for i in range(n):
        for j in range(n):
            diag_tbt[i, i, j, j] = ndia_tbt[i, i, j, j]
            ndia_tbt[i, i, j, j] = 0
    return diag_tbt, ndia_tbt

def separate_number_op(op:FermionOperator):
    '''
    Removing number operator from op 
    '''
    diag = FermionOperator.zero()
    ndia = FermionOperator.zero()
    for term, val in op.terms.items():
        cr = []
        an = [] 
        for fw in term:
            if fw[1] == 1:
                cr.append(fw[0])
            else:
                an.append(fw[0])
        for idx in cr:
            if idx in an:
                an.remove(idx)
        curterm = FermionOperator(term=term, coefficient=val)
        if len(an) == 0:
            diag += curterm
        else:
            ndia += curterm
    return diag, ndia

def get_openfermion_hf(n_qubits, n_electrons):
    """Compute the ground hartree fock state in openfermion's format |psi><psi| 

    Args:
        n_qubits: Number of qubits (spin_orbitals).
        n_electrons: Number of electrons in hartree fock. 

    Returns:
        wfs (sparse_matrix): Density that represents the Hartree Fock state. 
    """
    # Construct ON vector
    occupation_vec = np.zeros(n_qubits)
    occupation_vec[-n_electrons:] = 1

    # Identify corresponding index in exponential basis
    idx = 0
    for i in range(len(occupation_vec)):
        if occupation_vec[i] == 1:
            idx += 2 ** i
    idx_tuple = (idx,)

    # Construct sparse matrix
    dim = 2**n_qubits
    hf_vec = np.zeros(dim)
    hf_vec[idx] = 1
    return hf_vec
    #return sp.sparse.csr_matrix(((1,), (idx_tuple, idx_tuple)), shape=(dim, dim))

def Pauli_filter(H : QubitOperator, tol=5e-4):
    H_filt = QubitOperator.zero()
    for pw, val in H.terms.items():
            if np.abs(val) > tol:
                H_filt += QubitOperator(term=pw, coefficient=val)
    return H_filt

def get_tbt(H : FermionOperator, n = None, spin_orb=False):
    '''
    Obtain the 4-rank tensor that represents two body interaction in H.
    In chemist ordering a^ a a^ a.
    In addition, simplify tensor assuming symmetry between alpha/beta coefficients
    '''
    if n is None:
        n = get_spin_orbitals(H)

    phy_tbt = np.zeros((n, n, n, n))
    for term, val in H.terms.items():
        if len(term) == 4:
            phy_tbt[
                term[0][0], term[1][0],
                term[2][0], term[3][0]
            ] = np.real_if_close(val)

    chem_tbt = np.transpose(phy_tbt, [0, 3, 1, 2])
    chem_tbt_sym = (chem_tbt - np.transpose(chem_tbt, [0,3,2,1]) + np.transpose(chem_tbt, [2,3,0,1]) - np.transpose(chem_tbt, [2,1,0,3]) ) / 4.0

    # Spin-orbital to orbital
    n_orb = phy_tbt.shape[0]
    n_orb = n_orb // 2
    alpha_indices = list(range(0, n_orb * 2, 2))
    beta_indices = list(range(1, n_orb * 2, 2))

    chem_tbt_orb = (chem_tbt_sym[np.ix_(alpha_indices, alpha_indices, beta_indices, beta_indices)]
                    - np.transpose(chem_tbt_sym[np.ix_(alpha_indices, beta_indices, beta_indices, alpha_indices)], [0,3,2,1]))
    if spin_orb:
        chem_tbt = np.zeros_like(chem_tbt_sym)
        n = chem_tbt_orb.shape[0]
        for i, j, k, l in product(range(n), repeat=4):
            for a, b in product(range(2), repeat=2):
                chem_tbt[(2*i+a, 1), (2*j+a, 0), (2*k+b, 1), (2*l+b, 0)] = chem_tbt_orb[i,j,k,l]
        return chem_tbt
    else:
        return chem_tbt_orb


def to_tensors(H : FermionOperator, n=None, spin_orb=False):
    no_h_ferm = normal_ordered(H)
    tbt = get_tbt(no_h_ferm, spin_orb = spin_orb)
    h1b = no_h_ferm - get_ferm_op(tbt, spin_orb)
    h1b = normal_ordered(h1b)
    obt = get_obt(h1b, spin_orb=spin_orb)

    return H.constant, obt, tbt
