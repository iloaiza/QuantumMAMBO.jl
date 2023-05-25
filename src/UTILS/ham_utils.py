import math
import pickle
import h5py
from openfermion import FermionOperator, QubitOperator, MolecularData, hermitian_conjugated, get_molecular_data
from openfermionpyscf import run_pyscf
from openfermion.transforms import get_fermion_operator

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

def get_system(mol_name, ferm = False, basis='sto3g', geometry=1, n_elec = False):
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
