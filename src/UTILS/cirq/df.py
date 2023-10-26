import cirq
import cirq_ft as cft
import numpy as np
from math import ceil, log2, pi
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

from givens import *
from utils import *

class DF_Subprepare(cft.GateWithRegisters):
    """
    Subprepare routine for DF. Corresponds to preparation of mu_j^(l) coefficients multiplexed by l SelectionRegister
    """

    def __init__(
        self, l_reg, mus_mat: NDArray[float], j_reg=None, probability_epsilon: float = 1.0e-5, dagger:bool = False
    ):
        """
        Args:
            - l_reg: SelectionRegister l over which operation is multiplexed, length L (includes one-body term)
            - mu_mat: [LxN] matrix holding mu_mat[l,j] = mu_j^(l) coefficients
            - j_reg: SelectionRegister with N iteration length over which coherent state will be created
            - probability_epsilon: The desired accuracy to represent each probability
                (which sets mu size and keep/alt integers).
                See `cirq_ft.linalg.lcu_util.preprocess_lcu_coefficients_for_reversible_sampling`
                for more information.
            - dagger: whether regular circuit (False) or inverse (True) is implemented
        """
        alt_1, keep_1, mu = linalg.preprocess_lcu_coefficients_for_reversible_sampling(
            lcu_coefficients=np.abs(mus_mat[0,:]), epsilon=probability_epsilon
        )
        self.dagger = dagger
        L = l_reg.iteration_length
        self.L = L
        Lmu, N = np.shape(mus_mat)
        self.N = N
        assert Lmu == L

        N2 = len(alt_1)
        alts_l = np.zeros((L, N2))
        keeps_l = np.zeros((L, N2))
        alts_l[0,:] = alt_1
        keeps_l[0,:] = keep_1

        for l in range(1, L):
            alts_l[l,:], keeps_l[l,:], mu = linalg.preprocess_lcu_coefficients_for_reversible_sampling(
            lcu_coefficients=np.abs(mus_mat[l,:]), epsilon=probability_epsilon)

        if type(j_reg) == type(None):
            self.j_register = cft.SelectionRegister("j", ceil(log2(N)), N)
        else:
            assert j_reg.iteration_length == N
            self.j_register = j_reg

        signs = np.zeros_like(mus_mat)
        for l in range(self.L):
            for n in range(self.N):
                if mus_mat[l,n] < 0:
                    signs[l,n] = 1

        self.l_register = l_reg
        self.alts_l = alts_l
        self.keeps_l = keeps_l
        self.signs = signs
        self.mu = mu

    @cached_property
    def sigma_mu_bitsize(self) -> int:
        return self.mu

    @cached_property
    def alternates_bitsize(self) -> int:
        return self.j_register.total_bits()

    @cached_property
    def keep_bitsize(self) -> int:
        return self.mu

    @cached_property
    def junk_registers(self) -> infra.Registers:
        return infra.Registers.build(
            sigma_mu=self.sigma_mu_bitsize,
            alt=self.alternates_bitsize,
            keep=self.keep_bitsize,
            less_than_equal=1,
            sign = 1,
        )

    @property
    def registers(self) -> cft.Registers:
        return merge_registers(self.l_register, self.j_register, self.junk_registers)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:
        l_qubs = get_qubits(self.l_register)
        less_than_equal = quregs['less_than_equal']
        sigma_mu, alt, keep = quregs['sigma_mu'], quregs['alt'], quregs['keep']
        sign_qubs = quregs['sign']
        j_qubs = get_qubits(self.j_register)
        
        qrom_gate = qrom.QROM(
            [self.alts_l, self.keeps_l],
            (len(l_qubs), len(j_qubs),),
            (self.alternates_bitsize, self.keep_bitsize),
        )
        qrom_signs = qrom.QROM.build(self.signs)

        print("Warning, uniform superposition gate does not give back correct number of gates in t_complexity! Cirq-ft bug...")
        if self.dagger == False: #all operations are self-inverse, so inverse just runs through them in the opposite direction
            yield prepare_uniform_superposition.PrepareUniformSuperposition(self.j_register.iteration_length).on(*j_qubs)
            yield cirq.H.on_each(*sigma_mu)
            yield qrom_gate.on_registers(selection0=l_qubs, selection1=j_qubs, target0=alt, target1=keep)
            yield arithmetic_gates.LessThanEqualGate(self.mu, self.mu).on(*keep, *sigma_mu, *less_than_equal)
            yield swap_network.MultiTargetCSwap.make_on(control=less_than_equal, target_x=alt, target_y=j_qubs)
            yield qrom_signs.on_registers(selection0=l_qubs, selection1=j_qubs, target0=sign_qubs)
        else:
            yield qrom_signs.on_registers(selection0=l_qubs, selection1=j_qubs, target0=sign_qubs)
            yield swap_network.MultiTargetCSwap.make_on(control=less_than_equal, target_x=alt, target_y=j_qubs)
            yield arithmetic_gates.LessThanEqualGate(self.mu, self.mu).on(*keep, *sigma_mu, *less_than_equal)
            yield qrom_gate.on_registers(selection0=l_qubs, selection1=j_qubs, target0=alt, target1=keep)
            yield cirq.H.on_each(*sigma_mu)
            yield prepare_uniform_superposition.PrepareUniformSuperposition(self.j_register.iteration_length).on(*j_qubs)



class DF_Select(cft.SelectOracle):
    """
    Yields L1(l) selection oracle for Double Factorization approach
    Implements the normalized version of the operation: (i.e. divided by 2*sum_j |mu_j^(l)|)
        L1(l)|l>||psi> -> |l> sum_{j,sigma} mu_j^(l) Ul * (i*gamma_{j,sigma,0}*gamma_{j,sigma,1}) * Ul'
    Where Ul is implemented via Givens rotations
    Considers N spacial orbitals

    inputs:
    Mandatory:
        - l_reg: SelectionRegister storing l index, has dimension L+1 (first entry is for 1-body term)
        - l_not_0_reg: one-qubit register which keeps track of l != 0 states
        - mus_mat: L*N-dimensional array holding mu_j^(l) entries
        - thetas_givens: [LxNx(N-1)]-dimensional tensor, holding N-1 Givens angles per mu_j^(l)
    Built on the fly if not specified:
        - j_reg: SelectionRegister that holds j index
        - spin_reg: SelectionRegister that holds spin index sigma
        - psi_up_reg: Register that holds spin up wavefunction
        - psi_down_reg: Register that holds spin down wavefunction
        - ctl_reg: one-qubit register by which this operation is controlled
        - probability_epsilon: tolerance for LCU coefficient encoding
    """
    def __init__(self, l_reg, l_not_0_reg, mus_mat, thetas_givens, j_reg = None, spin_reg = None, psi_up_reg = None,
            psi_down_reg = None, ctl_reg = None, probability_epsilon:float = 1.0e-5):
        self.L, self.N = np.shape(mus_mat)
        self.mus_mat = mus_mat
        
        assert l_reg.iteration_length == self.L

        assert np.shape(thetas_givens) == (self.L,self.N,self.N-1)

        if type(j_reg) == type(None):
            self.j_register = spacial_orbital_sel_register(self.N, orb_name="j")
        else:
            assert j_reg.iteration_length == self.N
            self.j_register = j_reg

        if type(spin_reg) == type(None):
            self.spin_register = spin_sel_register(spin_name="σ")
        else:
            assert spin_reg.iteration_length == 2

        if type(psi_up_reg) == type(None):
            self.psi_up_reg = cft.Register("psi_up", self.N)
        else:
            assert psi_up_reg.total_bits() == self.N
            self.psi_up_reg = psi_up_reg

        if type(psi_down_reg) == type(None):
            self.psi_down_reg = cft.Register("psi_down", self.N)
        else:
            assert psi_down_reg.total_bits() == self.N
            self.psi_down_reg = psi_down_reg

        if type(ctl_reg) == type(None):
            self.control_register = cft.Register("control", 1)
        else:
            assert ctl_reg.total_bits == 1
            self.control_register = ctl_reg

        self.l_register = l_reg

        assert l_not_0_reg.total_bits() == 1
        self.l_not_0_reg = l_not_0_reg

        self.probability_epsilon = probability_epsilon

        self.thetas_givens = thetas_givens


    @property
    def control_registers(self) -> infra.Registers:
        return merge_registers(self.control_register)

    @property
    def selection_registers(self) -> infra.SelectionRegisters:
        return merge_registers(self.l_register)

    @property
    def target_registers(self) -> infra.Registers:
        return merge_registers(self.psi_up_reg, self.psi_down_reg)


    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid], 
    ) -> cirq.OP_TREE:

        ctl_qub = get_qubits(self.control_register)
        l_qubs = get_qubits(self.l_register)
        l_not_0_qubs = get_qubits(self.l_not_0_reg)
        psi_up_qubs = get_qubits(self.psi_up_reg)
        psi_down_qubs = get_qubits(self.psi_down_reg)
        spin_qubs = get_qubits(self.spin_register)
        j_qubs = get_qubits(self.j_register)

        #build PREP*SEL*PREP' first part of circuit
        PREP = DF_Subprepare(self.l_register, self.mus_mat, j_reg=self.j_register, probability_epsilon = self.probability_epsilon)
        PREP_qubs = get_qubits(PREP.registers)
        prep_op = PREP.on(*PREP_qubs)
        yield prep_op
        yield cirq.H.on(*spin_qubs)

        lj_regs = merge_registers(self.l_register, self.j_register)
        spin_swap = swap_network.MultiTargetCSwap.make_on(control=spin_qubs, target_x=psi_up_qubs, target_y=psi_down_qubs)
        yield spin_swap

        Ugivens = MultiplexedEFTGivens(self.thetas_givens, dual = True, selection_registers = lj_regs, target_register = self.psi_up_reg)
        givens_qubs = get_qubits(Ugivens.registers)
        givens_op = Ugivens.on(*givens_qubs)

        yield givens_op

        psi1_qub = psi_up_qubs[0]
        Z1 = cirq.ZPowGate(exponent=1, global_shift=1)
        Z1op = Z1.on(psi1_qub)

        yield Z1op.controlled_by(*ctl_qub)

        Ugivens_inv = MultiplexedEFTGivens(self.thetas_givens, dual = True, selection_registers = lj_regs, target_register = self.psi_up_reg, dagger=True)
        givens_inv_op = Ugivens_inv.on(*givens_qubs)
        yield givens_inv_op
        yield spin_swap

        PREP_INV = DF_Subprepare(self.l_register, self.mus_mat, j_reg=self.j_register, probability_epsilon = self.probability_epsilon, dagger=True)
        prep_op_inv = PREP_INV.on(*PREP_qubs)
        yield prep_op_inv
        yield cirq.H.on(*spin_qubs)

        #inversion for T2 part. Builds temporary register to hold result of and gate, applies controlled Z on this register
        fact_len = len(j_qubs) + 2
        and_ancilla_register = cft.Registers.build(and_ancilla = fact_len - 2)
        and_ctl_regs = merge_registers(self.l_not_0_reg, self.j_register, self.spin_register)
        and_targ_reg = cft.Registers.build(and_targ = 1)
        and_gate = cft.And((0,) + (1,) * (fact_len - 1))
        and_op = and_gate.on_registers(control=get_qubits(and_ctl_regs), ancilla=get_qubits(and_ancilla_register), target=get_qubits(and_targ_reg))
        yield and_op
        yield cirq.Z.on(*ctl_qub).controlled_by(*get_qubits(and_targ_reg))
        yield and_op

        #second part of walk operator
        yield prep_op
        yield cirq.H.on(*spin_qubs)
        yield spin_swap
        yield givens_op
        yield Z1op.controlled_by(*ctl_qub, *l_not_0_qubs)

        yield givens_inv_op
        yield spin_swap

        yield prep_op_inv
        yield cirq.H.on(*spin_qubs)

        yield and_op
        yield cirq.Z.on(*ctl_qub).controlled_by(*get_qubits(and_targ_reg))
        yield and_op

        yield prep_op

class DF_Prepare(cft.PrepareOracle):
    """
    Double Factorization preparation oracle. Prepares superposition over L = R+1 coefficents, where R is the number 
    of fragments for factorizing the two-electron tensor

    Inputs:
        - l_reg: SelectionRegister storing l index, has dimension L = R+1 (first entry is for 1-body term)
        - l_not_0_reg: one-qubit register which keeps track of l != 0 states
        - mus_mat: L*N-dimensional array holding mu_j^(l) entries
        - probability_epsilon: accuracy for preparing superposition state
        - dagger: boolean variable which indicates whether the inverse (True) or regular circuit (False) are implemented
    """
    def __init__(self, mus_mat, l_reg = None, l_not_0_reg = None, probability_epsilon:float = 1e-5, dagger:bool = False):
        self.L, self.N = np.shape(mus_mat)
        self.dagger = dagger

        if type(l_reg) == type(None):
            self.l_register = spacial_orbital_sel_register(self.L, "l")
        else:
            assert l_reg.iteration_length == self.L
            self.l_register = l_reg

        if type(l_not_0_reg) == type(None):
            self.l_not_0_register = cft.Register("l_not_0", 1)
        else:
            assert l_not_0_reg.total_bits() == 1
            self.l_not_0_register = l_not_0_reg

        lambdas_arr = np.zeros(self.L)
        lambdas_arr[0] = np.sum(np.abs(mus_mat[0,:]))

        for l in range(1,self.L):
            lambda_l = np.sum(np.abs(mus_mat[l,:]))
            lambdas_arr[l] = lambda_l ** 2

        self.lambdas_arr = lambdas_arr

        alt, keep, mu = linalg.preprocess_lcu_coefficients_for_reversible_sampling(
            lcu_coefficients=lambdas_arr, epsilon=probability_epsilon
        )

        self.alt = np.array(alt)
        self.keep = np.array(keep)
        self.mu = mu

    @property
    def selection_registers(self) -> infra.SelectionRegisters:
        return merge_registers(self.l_register, self.l_not_0_register)

    @cached_property
    def sigma_mu_bitsize(self) -> int:
        return self.mu

    @cached_property
    def alternates_bitsize(self) -> int:
        return self.l_register.total_bits()

    @cached_property
    def keep_bitsize(self) -> int:
        return self.mu

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
        l_qubs = get_qubits(self.l_register)
        l_not_0_qubs = get_qubits(self.l_not_0_register)
        less_than_equal = quregs['less_than_equal']
        sigma_mu, alt, keep = quregs['sigma_mu'], quregs['alt'], quregs['keep']
        
        l_not_0_data = np.ones(self.L, dtype = int)
        l_not_0_data[0] = 0
        qrom_gate = qrom.QROM(
            [self.alt, self.keep, l_not_0_data],
            (self.alternates_bitsize,),
            (self.alternates_bitsize, self.keep_bitsize, 1),
        )

        print("Warning, uniform superposition gate does not give back correct number of gates in t_complexity! Cirq-ft bug...")
        if self.dagger == False: #all opertions are self-inverse, so dagger runs in opposite sense
            yield prepare_uniform_superposition.PrepareUniformSuperposition(self.L).on(*l_qubs)
            yield cirq.H.on_each(*sigma_mu)
            yield qrom_gate.on_registers(selection=l_qubs, target0=alt, target1=keep, target2=l_not_0_qubs)
            yield arithmetic_gates.LessThanEqualGate(self.mu, self.mu).on(*keep, *sigma_mu, *less_than_equal)
            yield swap_network.MultiTargetCSwap.make_on(control=less_than_equal, target_x=alt, target_y=l_qubs)
        else:
            yield swap_network.MultiTargetCSwap.make_on(control=less_than_equal, target_x=alt, target_y=l_qubs)
            yield arithmetic_gates.LessThanEqualGate(self.mu, self.mu).on(*keep, *sigma_mu, *less_than_equal)
            yield qrom_gate.on_registers(selection=l_qubs, target0=alt, target1=keep, target2=l_not_0_qubs)
            yield cirq.H.on_each(*sigma_mu)
            yield prepare_uniform_superposition.PrepareUniformSuperposition(self.L).on(*l_qubs)

