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

def controlled_majorana_vec(orbs_reg, spins_reg , is_x = True, ctl_reg = cft.Registers.build(control=1)):
    #creates controlled majorana vec application over spacial orbitals register orbs_reg and spin register spin_reg
    sel_regs = cft.SelectionRegisters([orbs_reg, spins_reg])

    if is_x:
        which_gate = cirq.X
    else:
        which_gate = cirq.Y

    gate = cft.SelectedMajoranaFermionGate(sel_regs,ctl_reg,target_gate=which_gate)
    circuit = to_circuit(gate)

    return circuit, gate

class controlled_Qija(SelectOracle):
    #returns controlled version for SELECT circuit implementing Qija, select registers 
    #for spacial orbitals i_reg and j_reg, spin register spin_reg. Controlled over ctl_register
    #SELECT|ija>|psi> = |ija> Qija |psi>
    #with Qjka = i γ{ja,0} γ{ka,1}

    i_register : cft.SelectionRegister
    j_register : cft.SelectionRegister
    spin_register : cft.SelectionRegister
    ctl_register : cft.Registers
    target_register : cft.Registers

    def __init__(self, num_spacial_orbs, i_name="orb_i", j_name="orb_j",
        spin_name="spin_1", ctl_reg = cft.Registers.build(control=1), target_name="ψ"):
        self.i_register = spacial_orbital_sel_register(num_spacial_orbs, i_name)
        self.j_register = spacial_orbital_sel_register(num_spacial_orbs, j_name)
        self.spin_register = spin_sel_register(spin_name)
        self.ctl_register = ctl_reg
        self.target_register = cft.Registers([cft.Register(name=target_name, shape=num_spacial_orbs+1)])

    
    @property 
    def selection_registers(self):
        return cft.SelectionRegisters([self.i_register, self.j_register, self.spin_register])

    @property
    def control_registers(self):
        return self.ctl_register

    @property
    def target_registers(self):
        return self.target_register
    
    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid], 
    ) -> cirq.OP_TREE:
        
        ctl_qubs = get_qubits(self.ctl_register)

        ''' Controlled application of global i phase
        i_phase = cirq.global_phase_operation(coefficient=1j)
        yield i_phase.controlled_by(*ctl_qubs)
        '''

        maj_1 = cft.SelectedMajoranaFermionGate(cft.SelectionRegisters([self.i_register, self.spin_register]),
            control_regs=self.ctl_register, target_gate=cirq.Y)
        yield maj_1.on_registers(**maj_1.registers.get_named_qubits())
        
        
        yield cirq.S.on(*ctl_qubs)
        
        maj_2 = cft.SelectedMajoranaFermionGate(cft.SelectionRegisters([self.j_register, self.spin_register]),
            self.ctl_register, target_gate=cirq.X)
        yield maj_2.on_registers(**maj_2.registers.get_named_qubits())
            
    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@"] * self.ctl_register.total_bits()
        wire_symbols += ["In"] * self.selection_registers.total_bits()
        i_name = self.i_register.name
        j_name = self.j_register.name
        spin_name = self.spin_register.name
        Qija_name = "Q" + i_name + j_name + spin_name
        for i, target in enumerate(self.target_registers):
            wire_symbols += [Qija_name] * target.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)


@cirq.value_equality()
@attr.frozen
class TargetedPrepare(cft.StatePreparationAliasSampling):
    #same as StatePreparationAliasSampling but added ability to array of selection registers for T-counting
    selection_registers: cft.SelectionRegisters
    alt: NDArray[np.int_]
    keep: NDArray[np.int_]
    mu: int
        
    @classmethod
    def from_lcu_probs(
        cls, lcu_probabilities: List[float], sel_regs = None, *, probability_epsilon: float = 1.0e-5
    ) -> 'TargetedPrepare':
        """Factory to construct the state preparation gate for a given set of LCU coefficients.

        Args:
            lcu_probabilities: The LCU coefficients.
            probability_epsilon: The desired accuracy to represent each probability
                (which sets mu size and keep/alt integers).
                See `cirq_ft.linalg.lcu_util.preprocess_lcu_coefficients_for_reversible_sampling`
                for more information.
        """
        alt, keep, mu = linalg.preprocess_lcu_coefficients_for_reversible_sampling(
            lcu_coefficients=lcu_probabilities, epsilon=probability_epsilon
        )
        if type(sel_regs) == type(None):
            N = len(lcu_probabilities)
            sel_regs = cft.SelectionRegisters([cft.SelectionRegister("selection", ceil(log2(N)), N)])
        return TargetedPrepare(
            selection_registers=sel_regs,
            alt=np.array(alt),
            keep=np.array(keep),
            mu=mu)

    @property
    def registers(self) -> cft.Registers:
        return merge_registers(self.selection_registers, self.junk_registers)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:
        selection = get_qubits(self.selection_registers)
        less_than_equal = quregs['less_than_equal']
        sigma_mu, alt, keep = quregs['sigma_mu'], quregs['alt'], quregs['keep']
        N = 1
        for reg in self.selection_registers:
            N *= reg.iteration_length
        print("Warning, uniform superposition gate does not give back correct number of gates in t_complexity! Cirq-ft bug...")
        yield prepare_uniform_superposition.PrepareUniformSuperposition(N).on(*selection)
        yield cirq.H.on_each(*sigma_mu)
        qrom_gate = qrom.QROM(
            [self.alt, self.keep],
            (self.selection_bitsize,),
            (self.alternates_bitsize, self.keep_bitsize),
        )
        yield qrom_gate.on_registers(selection=selection, target0=alt, target1=keep)
        yield arithmetic_gates.LessThanEqualGate(self.mu, self.mu).on(
            *keep, *sigma_mu, *less_than_equal
        )
        yield swap_network.MultiTargetCSwap.make_on(
            control=less_than_equal, target_x=alt, target_y=selection
        )

"""
@cirq.value_equality()
@attr.frozen
class SparseSelect(cft.SelectOracle):
    #selection oracle
    control_registers : cft.Registers
    selection_registers : cft.SelectionRegisters
    target_registers : cft.Registers
    selection_names : list

    @classmethod
    def build(cls, V_reg, i_reg, j_reg, k_reg, l_reg, s1_reg, s2_reg, ctl_reg, tgt_reg = "ψ"):
        control_registers = ctl_reg
        selection_registers = merge_registers(V_reg, i_reg, j_reg, k_reg, l_reg, s1_reg, s2_reg)
        selection_names = get_names(selection_registers)
        if type(tgt_reg) == str:
            num_spacial_orbs = i_reg.iteration_length
            target_registers = cft.Registers([cft.Register(name=tgt_reg, shape=num_spacial_orbs+1)])
        elif type(tgt_reg) == cft.Registers or type(tgt_reg) == cft.Register:
            target_registers = merge_registers(tgt_reg)
        return SparseSelect(
            control_registers = control_registers,
            selection_registers = selection_registers,
            target_registers = target_registers,
            selection_names = selection_names)
    
    def _value_equality_values_(self):
        return (
            self.control_registers,
            self.selection_registers,
            self.target_registers,
            self.selection_names
        )
    
    def decompose_from_registers(self,
        *,
        context: cirq.DecompositionContext = cirq.DecompositionContext(cirq.ops.SimpleQubitManager()),
        **quregs: NDArray[cirq.Qid]):
        v_name, i_name, j_name, k_name, l_name, s1_name, s2_name = self.selection_names
        ctl_name = self.control_registers[0].name
        tgt_name = self.target_registers[0].name
        
        v_qubs = quregs[v_name]
        i_qubs = quregs[i_name]
        j_qubs = quregs[j_name]
        k_qubs = quregs[k_name]
        l_qubs = quregs[l_name]
        s1_qubs = quregs[s1_name]
        s2_qubs = quregs[s2_name]
        ctl_qubs = quregs[ctl_name]
        targ_qubs = quregs[targ_name]
"""


def sparse_select(V_reg, i_reg, j_reg, k_reg, l_reg, s1_reg, s2_reg, ctl_reg, tgt_name = "ψ"):
    #builds select operator where V (one qubit) register indicates V=0 -> one electron, V=1 -> two-electron
    num_spacial_orbs = i_reg.iteration_length
    Qija = controlled_Qija(num_spacial_orbs, i_reg.name, j_reg.name, s1_reg.name, ctl_reg, tgt_name)
    
    and_ctl_reg = cft.Registers([ctl_reg[0], V_reg[0]])
    and_targ_reg = cft.Registers.build(and_targ=1)
    
    and_op = cirq.CCNOT.on(*get_qubits(and_ctl_reg), *get_qubits(and_targ_reg))

    Qklb = controlled_Qija(num_spacial_orbs, i_reg.name, j_reg.name, s1_reg.name, and_targ_reg, tgt_name)

    OPS = [to_operation(Qija), and_op, to_operation(Qklb)]
    CIRC = to_circuit(Qija) + cirq.Circuit(and_op) + to_circuit(Qklb)

    return CIRC, OPS


def sparse_factorized_select(fact_reg, i_reg, j_reg, k_reg, l_reg, s1_reg, s2_reg, ctl_reg, tgt_name="ψ"):
    #build select for sparse factorization applied over l register, for which l=0 applies one-electron
    #operator Qij(s1) and l>0 applies two-body corresponding to Qij(s1)*Qkl(s2)

    num_spacial_orbs = i_reg.iteration_length
    Qija = controlled_Qija(num_spacial_orbs, i_reg.name, j_reg.name, s1_reg.name, ctl_reg, tgt_name)
    
    fact_len = fact_reg.total_bits()
    and_ancilla_register = cft.Registers.build(and_ancilla = fact_len - 1)
    and_ctl_regs = merge_registers(ctl_reg, fact_reg)
    and_targ_reg = cft.Registers.build(and_targ = 1)
    and_gate = cft.And((0,) * (fact_len + 1))
    and_op = and_gate.on_registers(
        control=get_qubits(and_ctl_regs), ancilla=get_qubits(and_ancilla_register), target=get_qubits(and_targ_reg)
    )

    not_and_circ = cirq.X(*and_targ_reg.get_named_qubits()["and_targ"])
    
    Qklb = controlled_Qija(num_spacial_orbs, k_reg.name, l_reg.name, s2_reg.name, and_targ_reg, tgt_name)

    return to_circuit(Qija) + cirq.Circuit(and_op) + not_and_circ + to_circuit(Qklb)

def delta(a,b):
    if a == b:
        return 1
    else:
        return 0

def naive_sparse_prepare(hij, gijkl, epsilon = 1.0e-5):
    print("WARNING, NOT ACCOUNTING FOR SIGNS, EVERYTHING IS POSITIVE...")
    print("NOT ACCOUNTING FOR ORDER OF COEFFICIENTS CORRECTLY")
    n_orbs = np.shape(hij)[0]

    h_norm = np.sum(np.abs(hij))
    g_norm = np.sum(np.abs(gijkl))
    
    vijaklb_coeffs = np.zeros((2, n_orbs, n_orbs, 2, n_orbs, n_orbs, 2))
    for V in range(2):
        for i in range(n_orbs):
            for j in range(n_orbs):
                for k in range(n_orbs):
                    for l in range(n_orbs):
                        for a in range(2):
                            for b in range(2):
                                if V == 0:
                                    vijaklb_coeffs[V,i,j,a,k,l,b] = delta(k,0) * delta(l,0) * delta(b,0) * np.sqrt(2) * np.abs(hij[i,j])
                                else:
                                    vijaklb_coeffs[V,i,j,a,k,l,b] = 2 * np.abs(gijkl[i,j,k,l])

    pos_coeffs = vijaklb_coeffs.flatten()
    print("Coefficients 1-norm = {}, 2-norm = {}".format(np.sum(vijaklb_coeffs), np.sum(vijaklb_coeffs**2)))

    i_reg = spacial_orbital_sel_register(n_orbs, orb_name="orb_i")
    j_reg = spacial_orbital_sel_register(n_orbs, orb_name="orb_j")
    k_reg = spacial_orbital_sel_register(n_orbs, orb_name="orb_k")
    l_reg = spacial_orbital_sel_register(n_orbs, orb_name="orb_l")
    s1_reg = spin_sel_register(spin_name="spin_1")
    s2_reg = spin_sel_register(spin_name="spin_2")
    v_reg = cft.SelectionRegisters.build(V=1)

    SEL_REGS = merge_registers(v_reg, i_reg, j_reg, s1_reg, k_reg, l_reg, s2_reg)

    PREP = TargetedPrepare.from_lcu_probs(pos_coeffs.tolist(), sel_regs = SEL_REGS, probability_epsilon=epsilon)

    return PREP

@cirq.value_equality()
@attr.frozen
class FactorizedGijlSubprepare(cft.algos.select_and_prepare.PrepareOracle):
    i_register : cft.SelectionRegister
    j_register : cft.SelectionRegister
    l_register : cft.SelectionRegister
    alts_l: NDArray[np.int_] #first coordinate for l, second for (i,j)
    keeps_l: NDArray[np.int_] #first coordinate for l, second for (i,j)
    mu: int
    L: int


    @classmethod
    def from_gijl(
        cls, gijl: NDArray[np.float], i_reg=None, j_reg=None, l_reg=None, *, probability_epsilon: float = 1.0e-5
    ) -> 'FactorizedGijlSubprepare':
        """Factory to construct the state preparation gate for a given gijl tensor,
        carries coefficients of single-factorized two-electron tensor
        (i.e. gpqrs = sum_l gpql grsl )

        Args:
            gijl: [LxNxN] tensor 
            probability_epsilon: The desired accuracy to represent each probability
                (which sets mu size and keep/alt integers).
                See `cirq_ft.linalg.lcu_util.preprocess_lcu_coefficients_for_reversible_sampling`
                for more information.
                sel_regs: 
        """
        alt_1, keep_1, mu = linalg.preprocess_lcu_coefficients_for_reversible_sampling(
            lcu_coefficients=gijl[0,:,:].flatten(), epsilon=probability_epsilon
        )
        
        L = np.shape(gijl)[0]
        N2 = len(alt_1)
        alts_l = np.zeros((L, N2))
        keeps_l = np.zeros((L, N2))
        alts_l[0,:] = alt_1
        keeps_l[0,:] = keep_1

        for l in range(1, L):
            alts_l[l,:], keeps_l[l,:], mu = linalg.preprocess_lcu_coefficients_for_reversible_sampling(
            lcu_coefficients=gijl[l,:,:].flatten(), epsilon=probability_epsilon)

        N = np.shape(gijl)[1]
        if type(i_reg) == type(None):
            i_reg = cft.SelectionRegister("orb_i", ceil(log2(N)), N)
        if type(j_reg) == type(None):
            j_reg = cft.SelectionRegister("orb_j", ceil(log2(N)), N)
        if type(l_reg) == type(None):
            l_reg = cft.SelectionRegister("SF_l", ceil(log2(L)), L)
        return FactorizedGijlSubprepare(
            i_register=i_reg,
            j_register=j_reg,
            l_register=l_reg,
            alts_l=alts_l,
            keeps_l=keeps_l,
            mu=mu,
            L=L)

    @cached_property
    def selection_registers(self) -> infra.SelectionRegisters:
        return merge_registers(self.i_register, self.j_register, self.l_register)

    @cached_property
    def sigma_mu_bitsize(self) -> int:
        return self.mu

    @cached_property
    def ij_bitsize(self) -> int:
        return self.i_register.total_bits() + self.j_register.total_bits()

    @cached_property
    def alternates_bitsize(self) -> int:
        return self.i_register.total_bits() + self.j_register.total_bits()

    @cached_property
    def keep_bitsize(self) -> int:
        return self.mu

    @cached_property
    def L_bitsize(self) -> int:
        return ceil(log2(self.L))

    @cached_property
    def junk_registers(self) -> infra.Registers:
        return infra.Registers.build(
            sigma_mu=self.sigma_mu_bitsize,
            alt=self.alternates_bitsize,
            keep=self.keep_bitsize,
            less_than_equal=1,
        )

    def _value_equality_values_(self):
        return (
            self.i_register,
            self.j_register,
            self.l_register,
            tuple(self.alts_l.ravel()),
            tuple(self.keeps_l.ravel()),
            self.mu,
            self.L
        )

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:
        orb_sel_registers = merge_registers(self.i_register, self.j_register)
        less_than_equal = quregs['less_than_equal']
        sigma_mu, alt, keep = quregs['sigma_mu'], quregs['alt'], quregs['keep']
        

        N = 1
        for reg in orb_sel_registers:
            N *= reg.iteration_length

        orb_selection = get_qubits(orb_sel_registers)
        print("Warning, uniform superposition gate does not give back correct number of gates in t_complexity! Cirq-ft bug...")
        yield prepare_uniform_superposition.PrepareUniformSuperposition(N).on(*orb_selection)
        yield cirq.H.on_each(*sigma_mu)
        
        qrom_gate = qrom.QROM(
            [self.alts_l, self.keeps_l],
            (self.L_bitsize, self.ij_bitsize,),
            (self.alternates_bitsize, self.keep_bitsize),
        )
        
        l_selection = get_qubits(self.l_register)
        yield qrom_gate.on_registers(selection0=l_selection, selection1=orb_selection, target0=alt, target1=keep)
        yield arithmetic_gates.LessThanEqualGate(self.mu, self.mu).on(
            *keep, *sigma_mu, *less_than_equal
        )
        yield swap_network.MultiTargetCSwap.make_on(
            control=less_than_equal, target_x=alt, target_y=orb_selection
        )

@cirq.value_equality
@attr.frozen
class SingleFactorizedPrepare(cft.algos.select_and_prepare.PrepareOracle):
    i_register : cft.SelectionRegister #spacial orbital registers
    j_register : cft.SelectionRegister #spacial orbital registers
    k_register : cft.SelectionRegister #spacial orbital registers
    l_register : cft.SelectionRegister #spacial orbital registers
    SF_register : cft.SelectionRegister #register holding single-factorization index
    s1_register : cft.SelectionRegister #spin registers
    s2_register : cft.SelectionRegister #spin registers
    hij : NDArray[np.float_] #one-electron tensor
    gijl : NDArray[np.float_] #two-electron single-factorized tensor (gpqrs = sum_l gpql*grsl)
    probability_epsilon : float #probability for QROM precision

    @classmethod
    def build(
        cls, hij : NDArray[np.float], gijl: NDArray[np.float], *, probability_epsilon: float = 1.0e-5
    ) -> 'SingleFactorizedPrepare':

        N = np.shape(hij)[0]
        L = np.shape(gijl)[0] + 1
        i_reg = cft.SelectionRegister("orb_i", ceil(log2(N)), N)
        j_reg = cft.SelectionRegister("orb_j", ceil(log2(N)), N)
        k_reg = cft.SelectionRegister("orb_k", ceil(log2(N)), N)
        l_reg = cft.SelectionRegister("orb_l", ceil(log2(N)), N)
        SF_reg = cft.SelectionRegister("SF_l", ceil(log2(L)), L)
        s1_reg = spin_sel_register("spin_1")
        s2_reg = spin_sel_register("spin_2")

        return SingleFactorizedPrepare(
            i_register = i_reg,
            j_register = j_reg,
            k_register = k_reg,
            l_register = l_reg,
            SF_register = SF_reg,
            s1_register = s1_reg,
            s2_register = s2_reg,
            hij = hij,
            gijl = gijl,
            probability_epsilon = probability_epsilon)

    @cached_property
    def selection_registers(self) -> cft.SelectionRegisters:
        return merge_registers(self.i_register, self.j_register, self.k_register, self.l_register, self.SF_register,
            self.s1_register, self.s2_register)


    def _value_equality_values_(self):
        return (
            self.i_register,
            self.j_register,
            self.k_register,
            self.l_register,
            self.SF_register,
            self.s1_register,
            self.s2_register,
            tuple(self.hij.ravel()),
            tuple(self.gijl.ravel()),
            self.probability_epsilon,
        )

    @cached_property
    def junk_registers(self) -> cft.Registers:
        return cft.Registers.build(
            theta_1=1,
            theta_2=1,)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:

        L = np.shape(self.gijl)[0]
        wl = np.zeros(L+1)
        wl[0] = np.sqrt(np.sum(np.abs(self.hij)))
        for l in range(L):
            wl[l+1] = np.sum(np.abs(self.gijl[l,:,:]))
        SF_superposition_gate = TargetedPrepare.from_lcu_probs(wl, sel_regs = merge_registers(self.SF_register),
            probability_epsilon = self.probability_epsilon)
        superposition_qubs = get_qubits(SF_superposition_gate.registers)
        yield SF_superposition_gate.on(*superposition_qubs)

        N = np.shape(self.hij)[0]
        g_12el = np.zeros((L+1, N, N))
        g_12el[0,:,:] = self.hij
        g_12el[1:,:,:] = self.gijl
        prep1 = FactorizedGijlSubprepare.from_gijl(np.abs(g_12el), i_reg=self.i_register, j_reg=self.j_register,
            l_reg=self.SF_register, probability_epsilon=self.probability_epsilon)
        prep1_qubs = get_qubits(prep1.registers)
        yield prep1.on(*prep1_qubs)
        
        thetas_1 = np.array(np.where(g_12el <0, 1, 0), dtype=int)
        qrom_theta1 = qrom.QROM.build(thetas_1)
        i_qubs = get_qubits(self.i_register)
        j_qubs = get_qubits(self.j_register)
        SF_qubs = get_qubits(self.SF_register)
        theta_1_qubs = quregs["theta_1"]
        yield qrom_theta1.on_registers(selection0=SF_qubs, selection1=i_qubs, selection2=j_qubs, target0=theta_1_qubs)

        g_12el[0,:,:] = 0
        g_12el[0,0,0] = 1
        prep2 = FactorizedGijlSubprepare.from_gijl(np.abs(g_12el), i_reg=self.k_register, j_reg=self.l_register,
            l_reg=self.SF_register, probability_epsilon=self.probability_epsilon)
        prep2_qubs = get_qubits(prep2.registers)
        yield prep2.on(*prep2_qubs)

        thetas_2 = np.array(np.where(g_12el <0, 1, 0), dtype=int)
        qrom_theta2 = qrom.QROM.build(thetas_2)
        k_qubs = get_qubits(self.k_register)
        l_qubs = get_qubits(self.l_register)
        theta_2_qubs = quregs["theta_2"]
        yield qrom_theta2.on_registers(selection0=SF_qubs, selection1=k_qubs, selection2=l_qubs, target0=theta_2_qubs)

        spin_qubs = get_qubits(self.s1_register, self.s2_register)
        yield cirq.H.on_each(*spin_qubs)
        yield cirq.Z.on_each(*theta_1_qubs, *theta_2_qubs)


@cirq.value_equality
@attr.frozen
class SOTASparsePrepare(cft.algos.select_and_prepare.PrepareOracle):
    i_register : cft.SelectionRegister #spacial orbital registers
    j_register : cft.SelectionRegister #spacial orbital registers
    k_register : cft.SelectionRegister #spacial orbital registers
    l_register : cft.SelectionRegister #spacial orbital registers
    v_register : cft.SelectionRegister #register distinguishing between one- and two-electron terms
    s1_register : cft.SelectionRegister #spin registers
    s2_register : cft.SelectionRegister #spin registers
    sparse_coeffs : NDArray[np.float_] #sparsified coefficients vector, includes one- and two-electron components
    idx_arr : list #list mapping each element of sparse_coeffs to i,j,k,l indices and V mapping whether 1- or 2-electron
    alt: NDArray[np.int_]
    keep: NDArray[np.int_]
    mu: int
    
    @classmethod
    def build(
        cls, hij : NDArray[np.float], gijkl: NDArray[np.float], *, probability_epsilon: float = 1.0e-5, truncation_threshold:float = 1.0e-5
    ) -> 'SOTASparsePrepare':

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
                    
                    g_trunc[i,j,k,l] *= zeta_1[i,j] * zeta_1[k,l] * zeta_2[i,j,k,l]
                    if g_trunc[i,j,k,l] != 0:
                        coeffs += [g_trunc[i,j,k,l]]
                        idx_list += [(i,j,k,l,1)]

        for i in range(N):
            for j in range(N):
                if h_tilde[i,j] != 0:
                    coeffs += [h_tilde[i,j]]
                    idx_list += [(i,j,0,0,0)]

        num_alpha = len(coeffs)
        idx_arr = np.zeros((num_alpha, 5))
        for alpha in range(num_alpha):
            idx_arr[alpha,:] = idx_list[alpha]

        alt, keep, mu = linalg.preprocess_lcu_coefficients_for_reversible_sampling(
            lcu_coefficients=np.abs(coeffs), epsilon=probability_epsilon
        )
        
        return SOTASparsePrepare(
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
            mu=mu)

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
        N = self.i_register.total_bits()
        return 4*N+1

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




