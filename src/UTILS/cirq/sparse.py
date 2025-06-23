import cirq
import cirq_ft as cft
import numpy as np
from math import ceil, log2
import attr
from typing import List, Tuple
from numpy.typing import NDArray
from cirq._compat import cached_property
from cirq_ft import infra, linalg
from cirq_ft.algos import (
    and_gate,
    arithmetic_gates,
    prepare_uniform_superposition,
    qrom,
    select_and_prepare,
    swap_network,
    SelectOracle
)

from utils import *

class controlled_Qija(SelectOracle):
    #returns controlled version for SELECT circuit implementing Qija, select registers 
    #for spacial orbitals i_reg and j_reg, spin register spin_reg. Controlled over ctl_register
    #SELECT|ija>|psi> = |ija> Qija |psi>
    #with Qjka = i γ{ja,0} γ{ka,1}
    def __init__(self, num_spacial_orbs, i_reg=None, j_reg=None,
        spin_reg=None, ctl_reg = None, target_reg=None):
        self.N = num_spacial_orbs
        self._i_reg = i_reg
        self._j_reg = j_reg
        self._spin_reg = spin_reg
        self._ctl_reg = ctl_reg
        self._target_reg = target_reg

    @cached_property
    def i_register(self) -> cft.SelectionRegister:
        if type(self._i_reg) == type(None):
            return spacial_orbital_sel_register(self.N, "orb_i")
        else:
            assert type(self._i_reg) == cft.SelectionRegister
            assert self._i_reg.iteration_length == self.N, f"i SelectionRegister should have iteration length matching number of spacial orbitals"
            return self._i_reg

    @cached_property
    def j_register(self) -> cft.SelectionRegister:
        if type(self._j_reg) == type(None):
            return spacial_orbital_sel_register(self.N, "orb_j")
        else:
            assert type(self._j_reg) == cft.SelectionRegister
            assert self._j_reg.iteration_length == self.N, f"j SelectionRegister should have iteration length matching number of spacial orbitals"
            return self._j_reg

    @cached_property
    def spin_register(self) -> cft.SelectionRegister:
        if type(self._j_reg) == type(None):
            return spin_sel_register("spin")
        else:
            assert type(self._spin_reg) == cft.SelectionRegister
            assert self._spin_reg.iteration_length == 2, f"spin SelectionRegister should have iteration length 2"
            return self._spin_reg

    @cached_property
    def control_register(self) -> cft.Register:
        if type(self._ctl_reg) == type(None):
            return cft.Register("control", 1)
        else:
            assert type(self._ctl_reg) == cft.Register
            return self._ctl_reg

    @cached_property
    def target_register(self) -> cft.Register:
        if type(self._target_reg) == type(None):
            return cft.Register("target", 2*self.N)
        else:
            assert type(self._target_reg) == cft.Register
            return self._target_reg

    @property 
    def selection_registers(self):
        return cft.SelectionRegisters([self.i_register, self.j_register, self.spin_register])

    @property
    def control_registers(self):
        return cft.Registers([self.control_register])

    @property
    def target_registers(self):
        return cft.Registers([self.target_register])

    
    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid], 
    ) -> cirq.OP_TREE:
        
        ctl_qubs = get_qubits(self.control_registers)

        maj_1 = cft.SelectedMajoranaFermionGate(cft.SelectionRegisters([self.i_register, self.spin_register]),
            control_regs=self.control_registers, target_gate=cirq.Y)
        yield maj_1.on_registers(**maj_1.registers.get_named_qubits())
                
        yield cirq.S.on(*ctl_qubs)
        
        maj_2 = cft.SelectedMajoranaFermionGate(cft.SelectionRegisters([self.j_register, self.spin_register]),
            self.control_registers, target_gate=cirq.X)
        yield maj_2.on_registers(**maj_2.registers.get_named_qubits())
            
    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@"] * self.control_registers.total_bits()
        wire_symbols += ["In"] * self.selection_registers.total_bits()
        i_name = self.i_register.name
        j_name = self.j_register.name
        spin_name = self.spin_register.name
        Qija_name = "Q_{" + i_name + "," + j_name + "," + spin_name + "}"
        for i, target in enumerate(self.target_registers):
            wire_symbols += [Qija_name] * target.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

class Select_Sparse(cft.SelectOracle):
    """
    Controlled selection oracle for Pauli (Sparse) encoding, applies controlled:
    |i>|j>|k>|l>|a>|b>|0>|psi> -> |i>|j>|k>|l>|a>|b>|0> Q_{ija} |psi>
    |i>|j>|k>|l>|a>|b>|1>|psi> -> |i>|j>|k>|l>|a>|b>|0> Q_{ija}*Q_{klb} |psi>
    where last register is is V register
    """ 
    
    def __init__(self, v_reg:cft.SelectionRegister, i_reg:cft.SelectionRegister, j_reg:cft.SelectionRegister,
            k_reg:cft.SelectionRegister, l_reg:cft.SelectionRegister, a_reg:cft.SelectionRegister,
            b_reg:cft.SelectionRegister, ctl_reg = None, tgt_reg = None):
        #N: total number of spacial orbitals
        N = i_reg.iteration_length
        if type(tgt_reg) == type(None):
            self.target_register = cft.Registers.build(target=2*N)[0]
        else:
            assert tgt_reg.total_bits() == 2*N, f"Target register should have 2N qubits for N number of spacial orbitals"
            self.target_register = tgt_reg

        if type(ctl_reg) == type(None):
            self.control_register = cft.Registers.build(control=1)[0]
        else:
            self.control_register = ctl_reg

        self.i_register = i_reg
        self.j_register = j_reg
        self.k_register = k_reg
        self.l_register = l_reg
        self.v_register = v_reg
        self.a_register = a_reg
        self.b_register = b_reg
        self.N = N

    @property
    def selection_registers(self) -> cft.SelectionRegisters:
        return merge_registers(self.i_register, self.j_register, self.k_register, self.l_register,
            self.v_register, self.a_register, self.b_register)

    @property
    def control_registers(self) -> cft.Registers:
        return merge_registers(self.control_register)

    @property
    def target_registers(self) -> cft.Registers:
        return merge_registers(self.target_register)
    
    def decompose_from_registers(self,
        *,
        context: cirq.DecompositionContext = cirq.DecompositionContext(cirq.ops.SimpleQubitManager()),
        **quregs: NDArray[cirq.Qid]):
        
        Qija = controlled_Qija(self.N, i_reg=self.i_register, j_reg=self.j_register,
            spin_reg=self.a_register, ctl_reg = self.control_register, target_reg=self.target_register)
        
        yield Qija.on(*get_qubits(Qija.registers))
        yield cirq.S.on_each(quregs["control"])

        Qklb = controlled_Qija(self.N, i_reg=self.k_register, j_reg=self.l_register,
            spin_reg=self.b_register, ctl_reg = self.control_register, target_reg=self.target_register)
        
        yield Qklb.on(*get_qubits(Qklb.registers))



@cirq.value_equality
@attr.frozen
class Prepare_Sparse_SOTA(cft.algos.select_and_prepare.PrepareOracle):
    i_register : cft.SelectionRegister #spacial orbital registers
    j_register : cft.SelectionRegister #spacial orbital registers
    k_register : cft.SelectionRegister #spacial orbital registers
    l_register : cft.SelectionRegister #spacial orbital registers
    v_register : cft.SelectionRegister #register distinguishing between one- and two-electron terms
    s1_register : cft.SelectionRegister #spin registers
    s2_register : cft.SelectionRegister #spin registers
    sparse_coeffs : NDArray[float] #sparsified coefficients vector, includes one- and two-electron components
    idx_arr : list #list mapping each element of sparse_coeffs to i,j,k,l indices and V mapping whether 1- or 2-electron
    alt: NDArray[int]
    keep: NDArray[int]
    mu: int
    N : int #total number of spacial orbitals, hij has dimension NxN
    
    @classmethod
    def build(
        cls, hij : NDArray[float], gijkl: NDArray[float], *, probability_epsilon: float = 1.0e-5, truncation_threshold:float = 1.0e-5
    ) -> 'Prepare_Sparse_SOTA':

        N = np.shape(hij)[0]
        i_reg = cft.SelectionRegister("orb_i", ceil(log2(N)), N)
        j_reg = cft.SelectionRegister("orb_j", ceil(log2(N)), N)
        k_reg = cft.SelectionRegister("orb_k", ceil(log2(N)), N)
        l_reg = cft.SelectionRegister("orb_l", ceil(log2(N)), N)
        v_reg = cft.SelectionRegister("V", 1, 2)
        s1_reg = spin_sel_register("spin_1")
        s2_reg = spin_sel_register("spin_2")
        g_trunc = np.where(np.abs(gijkl) < truncation_threshold, 0, gijkl)
        h_tilde = np.copy(hij)
        coeffs = []
        idx_list = []

        #naive construction, can be made more efficient with better for loops
        zeta_1 = np.zeros((N,N))
        zeta_2 = np.zeros((N,N,N,N))
        for i in range(N):
            for j in range(N):
                if i < j:
                    zeta_1[i,j] = np.sqrt(2)
                elif i == j:
                    zeta_1[i,j] = 1

                h_tilde[i,j] *= zeta_1[i,j]
                
                for k in range(N):
                    for l in range(N):
                        if i < k or (i == k and j < l):
                            zeta_2[i,j,k,l] = np.sqrt(2)
                        elif i == k and j == l:
                            zeta_2[i,j,k,l] = 1
                    
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for l in range(N):
                        g_trunc[i,j,k,l] *= zeta_1[i,j] * zeta_1[k,l] * zeta_2[i,j,k,l]
                        if g_trunc[i,j,k,l] != 0:
                            coeffs += [g_trunc[i,j,k,l]]
                            idx_list += [(i,j,k,l,1)]

        for i in range(N):
            for j in range(N):
                if h_tilde[i,j] != 0:
                    coeffs += [h_tilde[i,j]]
                    idx_list += [(i,j,0,0,0)]

        print("\n\n Number of non-zero coeffs for Pauli sparse loading = {}\n\n".format(len(coeffs)))

        num_alpha = len(coeffs)
        idx_arr = np.zeros((num_alpha, 5))
        for alpha in range(num_alpha):
            idx_arr[alpha,:] = idx_list[alpha]

        alt, keep, mu = linalg.preprocess_lcu_coefficients_for_reversible_sampling(
            lcu_coefficients=np.abs(coeffs), epsilon=probability_epsilon
        )
        
        return Prepare_Sparse_SOTA(
            i_register = i_reg,
            j_register = j_reg,
            k_register = k_reg,
            l_register = l_reg,
            v_register = v_reg,
            s1_register = s1_reg,
            s2_register = s2_reg,
            sparse_coeffs = np.array(coeffs),
            idx_arr = idx_arr,
            alt=np.array(alt),
            keep=np.array(keep),
            mu=mu,
            N=N)

    @cached_property
    def selection_registers(self) -> cft.SelectionRegisters:
        return merge_registers(self.v_register, self.i_register, self.j_register, self.k_register, self.l_register,
            self.s1_register, self.s2_register)


    def _value_equality_values_(self):
        return (
            self.i_register,
            self.j_register,
            self.k_register,
            self.l_register,
            self.v_register,
            self.s1_register,
            self.s2_register,
            tuple(self.sparse_coeffs.ravel()),
            tuple(self.idx_arr.ravel()),
            tuple(self.alt.ravel()),
            tuple(self.keep.ravel()),
            self.mu
        )

    @cached_property
    def sigma_mu_bitsize(self) -> int:
        return self.mu

    @cached_property
    def alpha_len(self) -> int:
        return len(self.sparse_coeffs)

    @cached_property
    def alpha_bitsize(self) -> int:
        return ceil(log2(self.alpha_len))

    @cached_property
    def alternates_bitsize(self) -> int:
        return 4*ceil(log2(self.N)) + 1
        
    @cached_property
    def keep_bitsize(self) -> int:
        return self.mu

    @cached_property
    def junk_registers(self) -> cft.Registers:
        logical_regs = cft.Registers.build(
            theta=1,
            sigma_mu=self.sigma_mu_bitsize,
            alt=self.alternates_bitsize,
            keep=self.keep_bitsize,
            less_than_equal=1,
            swap_ij_kl = 1,
            swap_ij = 1,
            swap_kl = 1)
        num_alpha = len(self.sparse_coeffs)
        sel_regs = cft.SelectionRegister("alpha", ceil(log2(num_alpha)), num_alpha)

        return merge_registers(logical_regs, sel_regs)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:

        N = self.i_register.iteration_length
        N_bitsize = ceil(log2(N))
        alpha, less_than_equal = quregs["alpha"], quregs["less_than_equal"]
        sigma_mu, alt, keep = quregs['sigma_mu'], quregs['alt'], quregs['keep']

        num_alpha = self.alpha_len
        ind_i = np.zeros(num_alpha)
        ind_j = np.zeros(num_alpha)
        ind_k = np.zeros(num_alpha)
        ind_l = np.zeros(num_alpha)
        ind_v = np.zeros(num_alpha)
        for i_alpha in range(num_alpha):
            i,j,k,l,v = self.idx_arr[i_alpha, :]
            ind_i[i_alpha] = i
            ind_j[i_alpha] = j
            ind_k[i_alpha] = k
            ind_l[i_alpha] = l
            ind_v[i_alpha] = v
        i_qubs = get_qubits(self.i_register)
        j_qubs = get_qubits(self.j_register)
        k_qubs = get_qubits(self.k_register)
        l_qubs = get_qubits(self.l_register)
        v_qubs = get_qubits(self.v_register)
        theta_data = np.array(np.where(self.sparse_coeffs <0, 1, 0), dtype=int)
        theta_qubs = quregs["theta"]
        
        print("Warning, uniform superposition gate does not give back correct number of gates in t_complexity! Cirq-ft bug...")
        yield prepare_uniform_superposition.PrepareUniformSuperposition(num_alpha).on(*alpha)
        yield cirq.H.on_each(*sigma_mu)
        
        qrom_gate = qrom.QROM(
            [self.alt, self.keep, ind_i, ind_j, ind_k, ind_l, ind_v, theta_data],
            (self.alpha_bitsize,),
            (self.alternates_bitsize, self.keep_bitsize, N_bitsize, N_bitsize, N_bitsize, N_bitsize, 1, 1),
        )

        yield qrom_gate.on_registers(selection=alpha, target0=alt, target1=keep, target2=i_qubs, 
            target3=j_qubs, target4=k_qubs, target5=l_qubs, target6=v_qubs, target7=theta_qubs)

        yield arithmetic_gates.LessThanEqualGate(self.mu, self.mu).on(
            *keep, *sigma_mu, *less_than_equal
        )

        ijklv_qubs = [*i_qubs, *j_qubs, *k_qubs, *l_qubs, *v_qubs]
        yield swap_network.MultiTargetCSwap.make_on(
            control=less_than_equal, target_x=alt, target_y=ijklv_qubs
        )

        spin_qubs = get_qubits(self.s1_register, self.s2_register)
        yield cirq.H.on_each(*spin_qubs)
        yield cirq.Z.on_each(*theta_qubs)

        #generate symmetries of the state. Start by initializing swap registers
        swap_ij_qubs = quregs["swap_ij"]
        swap_kl_qubs = quregs["swap_kl"]
        swap_ijkl_qubs = quregs["swap_ij_kl"]
        yield cirq.H.on_each(*swap_ij_qubs)

        swaps_init = cirq.H.on_each(*swap_ijkl_qubs, *swap_ij_qubs)
        for h_gate in swaps_init:
            yield h_gate.controlled_by(*v_qubs)

        #swap ijkl controlled by swap registers
        ij_qubs = [*i_qubs, *j_qubs]
        kl_qubs = [*k_qubs, *l_qubs]
        yield swap_network.MultiTargetCSwap.make_on(control=swap_ijkl_qubs, target_x=ij_qubs, target_y=kl_qubs)
        yield swap_network.MultiTargetCSwap.make_on(control=swap_ij_qubs, target_x=i_qubs, target_y=j_qubs)
        yield swap_network.MultiTargetCSwap.make_on(control=swap_kl_qubs, target_x=k_qubs, target_y=l_qubs)




