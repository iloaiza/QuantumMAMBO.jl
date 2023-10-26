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

class FactorizedGijlSubprepare(cft.algos.select_and_prepare.PrepareOracle):
    i_register : cft.SelectionRegister
    j_register : cft.SelectionRegister
    l_register : cft.SelectionRegister
    alts_l: NDArray[int] #first coordinate for l, second for (i,j)
    keeps_l: NDArray[int] #first coordinate for l, second for (i,j)
    mu: int
    L: int

    def __init__(self, gijl : NDArray, i_reg = None, j_reg = None, l_reg = None, probability_epsilon : float = 1.0e-5):
        """Construct the state preparation gate for a given gijl tensor,
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
            lcu_coefficients=np.abs(gijl[0,:,:]).flatten(), epsilon=probability_epsilon
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

        assert i_reg.iteration_length == N
        self.i_register = i_reg

        assert j_reg.iteration_length == N
        self.j_register = j_reg

        assert l_reg.iteration_length == L
        self.l_register = l_reg

        self.alts_l = alts_l
        self.keeps_l = keeps_l
        self.mu = mu
        self.L = L
        self.N = N

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

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:
        orb_sel_registers = merge_registers(self.i_register, self.j_register)
        less_than_equal = quregs['less_than_equal']
        sigma_mu, alt, keep = quregs['sigma_mu'], quregs['alt'], quregs['keep']
        
        orb_selection = get_qubits(orb_sel_registers)
        print("Warning, uniform superposition gate does not give back correct number of gates in t_complexity! Cirq-ft bug...")
        yield prepare_uniform_superposition.PrepareUniformSuperposition(N).on(*get_qubits(self.i_register))
        yield prepare_uniform_superposition.PrepareUniformSuperposition(N).on(*get_qubits(self.j_register))
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
    hij : NDArray[float] #one-electron tensor
    gijl : NDArray[float] #two-electron single-factorized tensor (gpqrs = sum_l gpql*grsl)
    probability_epsilon : float #probability for QROM precision

    @classmethod
    def build(
        cls, hij : NDArray[float], gijl: NDArray[float], *, probability_epsilon: float = 1.0e-5
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