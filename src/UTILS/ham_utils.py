import math
import pickle
import h5py
from openfermion import FermionOperator, QubitOperator, MolecularData, hermitian_conjugated, get_molecular_data
from openfermionpyscf import run_pyscf
from openfermion.transforms import get_fermion_operator
import pyscf
from pyscf import gto, scf, lo
from pyscf.lo.boys import Boys
import numpy as np


def get_spin(orb):
    '''
    Getting the corresponding S2 operator given number of orbitals
    Assume alpha/beta ordering
    '''
    def get_sp(orb):
        sp = FermionOperator.zero()
        for i in range(orb):
            sp += FermionOperator(term=(
                (2*i, 1),
                (2*i+1, 0)
            ))
        return sp

    def get_sz(orb):
        sz = FermionOperator.zero()
        for i in range(orb):
            sz += FermionOperator(term=(
                (2*i, 1),
                (2*i, 0)
            ), coefficient=1/2)
            sz += FermionOperator(term=(
                (2*i+1, 1),
                (2*i+1, 0)
            ), coefficient=-1/2)
        return sz
        
    sp = get_sp(orb)
    sm = hermitian_conjugated(sp)
    sz = get_sz(orb)

    return sp * sm + sz * sz - sz

def get_mol(mol_name, basis='sto3g', geometry=1):
    '''
    Run pyscf on the specified molecule 
    '''
    g = chooseType(mol_name, geometry)
    mol = MolecularData(g, basis, 1, 0)
    mol = run_pyscf(mol)
    return mol 

def get_system(mol_name, ferm = True, basis='sto3g', geometry=1, n_elec = False):
    '''
    Obtain system from specified parameters
    '''
    g = chooseType(mol_name, geometry)
    mol = MolecularData(g, basis, 1, 0)
    mol = run_pyscf(mol)
    ham = mol.get_molecular_hamiltonian()
    if n_elec == False:
        if ferm:
            return get_fermion_operator(ham)
        else:
            return ham
    else:
        if ferm:
            return get_fermion_operator(ham), mol.n_electrons
        else:
            return ham, mol.n_electrons
        

def system_from_xyz(xyz, ferm = True, basis='sto3g', n_elec = False, spin=0):
    '''
    Obtain system from xyz string
    '''
    g = xyz_to_type(xyz)
    mol = MolecularData(g, basis, 1, 0)
    mol = run_pyscf(mol)
    ham = mol.get_molecular_hamiltonian()
    if n_elec == False:
        if ferm:
            return get_fermion_operator(ham)
        else:
            return ham
    else:
        if ferm:
            return get_fermion_operator(ham), mol.n_electrons
        else:
            return ham, mol.n_electrons


def localized_ham_from_xyz(xyz, basis='sto3g', n_elec='true', spin=0, charge=0,rhf=False):
    mol = gto.M()
    mol.atom = xyz
    mol.basis = basis
    mol.spin = spin
    mol.charge = charge
    mol.build()
    if mol.spin:
        if rhf:
            mf=scf.ROHF(mol)
        else:
            mf = scf.UHF(mol)
    else:
        mf=scf.RHF(mol)
    
    mf.run()
    hf_mos = mf.mo_coeff #defines canonical MOs (CMOs), corresponds to C matrix: CMOs = C * AOs
    
    h1e = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    h2e = mol.intor("int2e", aosym = 's1')
    nuclear_repulsion = mol.energy_nuc()
    
    N = np.shape(h1e)[0]
    
    if mol.spin and not rhf:
        h_hf=np.zeros((2,N,N))
        h_hf[0] = hf_mos[0].T @ h1e @ hf_mos[0]
        h_hf[1] = hf_mos[1].T @ h1e @ hf_mos[1]
        
        
        #h_hf = hf_mos.T @ h1e @ hf_mos
        
        g_hf = np.copy(h2e)
       
        
        
        '''for i in range(4):
           g_hf = np.tensordot(g_hf, hf_mos, axes=1).transpose(3, 0, 1, 2)'''
        
        g_hf_sv=np.zeros((4,N,N,N,N))
        for sigma in range(2):
            for tau in range(2):
                g_hf_sv[2*sigma+tau]=np.einsum('ijkl, ia, jb, kc, ld -> abcd' , g_hf, hf_mos[sigma,:,:], hf_mos[sigma,:,:], hf_mos[tau,:,:], hf_mos[tau,:,:])
           
        #Foster-Boys Localization
        '''h_fb = fb_mos.T @ h1e @ fb_mos
        g_fb = np.copy(h2e)
        for i in range(4):
           g_fb = np.tensordot(g_fb, fb_mos, axes=1).transpose(3, 0, 1, 2)'''
           
        
        np.set_printoptions(threshold=np.inf)
        
        return nuclear_repulsion, h_hf, g_hf_sv, mol.nelectron
    else:
        h_hf = hf_mos.T @ h1e @ hf_mos
        g_hf = np.copy(h2e)
        g_hf=np.einsum('ijkl, ia, jb, kc, ld -> abcd' , g_hf, hf_mos, hf_mos, hf_mos, hf_mos)
        return nuclear_repulsion, h_hf, g_hf, mol.nelectron


def xyz_to_type(xyz):
    molData = []

    for s_line in xyz:
        s_list = s_line.split(" ")
        atom = [s_list[0]]
        coords = []
        for i in range(1, len(s_list)):
            word = s_list[i]
            if len(word) > 0 and word != "\n":
                print(word)
                coords += [float(word)]
        atom += [coords]
        molData += [atom]

    return molData


def chooseType(typeHam, geometries):
    '''
    Genreate the molecular data of specified type of Hamiltonian
    '''
    if typeHam == 'h2':
        molData = [
            ['H', [0, 0, 0]],
            ['H', [0, 0, geometries]]
        ]
    elif typeHam == 'h3':
        molData = [
            ['H', [0, 0, 0]],
            ['H', [0, 0, geometries]],
            ['H', [0, 0, 2*geometries]]
        ]
    elif typeHam == 'h4_chain':
        molData = [
            ['H', [0, 0, 0]],
            ['H', [0, 0, geometries]],
            ['H', [0, 0, 2*geometries]],
            ['H', [0, 0, 3*geometries]]
        ]
    elif typeHam == 'n2':
        molData = [
            ['N', [0, 0, 0]],
            ['N', [0, 0, geometries]]
        ]
    elif typeHam == 'lih':
        molData = [
            ['Li', [0, 0, 0]],
            ['H', [0, 0, geometries]]
        ]
    # Giving symmetrically stretch H2O. ∠HOH = 107.6°
    elif typeHam == 'h2o':
        angle = 107.6 / 2
        angle = math.radians(angle)
        xDistance = geometries * math.sin(angle)
        yDistance = geometries * math.cos(angle)
        molData = [
            ['O', [0, 0, 0]],
            ['H', [-xDistance, yDistance, 0]],
            ['H', [xDistance, yDistance, 0]]
        ]
    elif typeHam == 'n2':
        molData = [
            ['N', [0, 0, 0]],
            ['N', [0, 0, geometries]]
        ]
    elif typeHam == 'beh2':
        molData = [
            ['Be', [0, 0, 0]],
            ['H', [0, 0, -geometries]],
            ['H', [0, 0, geometries]]
        ]
    elif typeHam == 'h4':
        angle1 = math.radians(geometries/2)
        angle2 = math.radians(90-geometries/2)
        R = 1.737236
        hor_val = 2*R*math.sin(angle1)
        ver_val = 2*R*math.sin(angle2)
        molData = [
            ['H', [0, 0, 0]],
            ['H', [hor_val, 0, 0]],
            ['H', [0, ver_val, 0]],
            ['H', [hor_val, ver_val, 0]]
        ]
    elif typeHam == 'nh3':
    # Is there a more direct way of making three vectors with specific mutual angle?
        bondAngle = 107
        bondAngle = math.radians(bondAngle)
        cos = math.cos(bondAngle)
        sin = math.sin(bondAngle)

        # The idea is second and third vecctor dot product is cos(angle) * geometry^2. 
        thirdyRatio = (cos - cos**2) / sin
        thirdxRatio = (1 - cos**2 - thirdyRatio**2) ** (1/2)
        molData = [
            ['H', [0, 0, geometries]],
            ['H', [0, sin * geometries, cos * geometries]], 
            ['H', [thirdxRatio * geometries, thirdyRatio * geometries, cos * geometries]], 
            ['N', [0, 0, 0]], 
        ]
    elif typeHam=='c2h2':
        molData=[
        ['H', [0,0,0]],
        ['C', [0,0,1.06]],
        ['C', [0,0,1.203+1.06]],
        ['H', [0,0, 2*1.06+1.203]]
    ]
    elif typeHam=="cr":
        molData=[
        ['Cr',[0,0,0]]
    ]
    elif typeHam=="v":
        molData=[
        ['V',[0,0,0]]
    ]
    elif typeHam=="Ti":
        molData=[
        ['Ti',[0,0,0]]
    ]
    elif typeHam=="ch3":
        molData=[
        ['C',[0,0,0]],
        ['H',[1,0,0]],
        ['H',[math.cos(2*math.pi/3),math.sin(2*math.pi/3),0]],
        ['H',[math.cos(4*math.pi/3),math.sin(4*math.pi/3),0]]
    ]
    elif typeHam=="ch2":
        molData=[
        ['C',[0,0,0]],
        ['H',[1,0,0]],
        ['H',[math.cos(2*math.pi/3),math.sin(2*math.pi/3),0]]
    ]
    elif typeHam=='TiN':
        moldata=['Ti 0.0 0.0 0.0', 'N 0.0 0.0 3.12256']
        molData=xyz_to_type(moldata)
        
    elif typeHam=='TiH':
        moldata=['Ti 0.0 0.0 0.0', 'H 0.0 0.0 1.77542']
        molData=xyz_to_type(moldata)
         
      
    
        
    else:
        raise(ValueError(typeHam, 'Unknown type of hamiltonian given'))

    return molData

def get_heisenberg_hamiltonian(n, Jx=1, Jy=1, Jz=1, h=1):
    '''
    Generate a qubit heisenberg hamiltonian with n modes
    '''

    H = QubitOperator.zero()
    X, Y, Z = 'X', 'Y', 'Z'
    for i in range(n-1):
        H += QubitOperator(term=((i, X), (i+1, X)), coefficient=Jx) +\
            QubitOperator(term=((i, Y), (i+1, Y)), coefficient=Jy) +\
            QubitOperator(term=((i, Z), (i+1, Z)), coefficient=Jz) +\
            QubitOperator(term=(i, Z), coefficient=h)
    return H
