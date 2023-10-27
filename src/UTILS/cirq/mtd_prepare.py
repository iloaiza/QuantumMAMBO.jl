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
import abc
from utils import *

debug_tiny = 1e-8 #make sure 2-norm of reconstructed g tensor is within this tolerance if debug == True

class one_el_Prepare(cft.PrepareOracle):
    """
    Prepares v register flagging one-electron term along h_tilde on i,j registers

    inputs:
        - h_tilde: NxN tensor (see MTD_Prepare for more info)
        - lambda_2: one-norm of two-electron part
        - i_reg: register flagging |i>
        - j_reg: register flagging |j>
        - v_reg: register flagging one-electron term (is 1 for two-electron)
        - probability_epsilon: tolerance for implementing one-electron term
    """

    def __init__(self, h_tilde : NDArray[float], lambda_2 : float, i_reg = None, j_reg = None,
        v_reg = None, probability_epsilon = 1.0e-5):
        assert len(np.shape(h_tilde)) == 2
        N,Nbis = np.shape(h_tilde)
        assert N == Nbis
        lambda_1 = np.sum(np.abs(h_tilde))
        assert h_tilde == np.transpose(h_tilde)

        self.N = N
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        if type(i_reg) == type(None):
            self.i_register = spacial_orbital_sel_register(N, "i")
        else:
            assert type(i_reg) == cft.SelectionRegister
            assert i_reg.iteration_length == N
            self.i_register = i_reg

        if type(j_reg) == type(None):
            self.j_register = spacial_orbital_sel_register(N, "j")
        else:
            assert type(j_reg) == cft.SelectionRegister
            assert j_reg.iteration_length == N
            self.j_register = j_reg

        if type(v_reg) == type(None):
            self.v_register = cft.SelectionRegister("v", 1, 2)
        else:
            assert type(v_reg) == cft.SelectionRegister
            assert v_reg.iteration_length == 2
            self.v_register = v_reg

        N_cal = int(N*(N+1)/2)
        self.N_cal = N_cal
        h_info = np.zeros(self.N_cal)

        Ncal_dict = {}
        idx = 0
        for i in range(N):
            for j in range(i):
                Ncal_dict[idx] = (i,j)
                if i == j:
                    h_info[idx] = h_tilde[i,i]
                else:
                    h_info[idx] = np.sqrt(2) * h_tilde[i,j]
                idx += 1

        self.h_signs = np.array(np.where(h_info<0, 1, 0), dtype=int)

        alt, keep, self.mu = cft.linalg.preprocess_lcu_coefficients_for_reversible_sampling(
            lcu_coefficients=np.abs(h_info), epsilon=probability_epsilon)

        self.keep = np.array(keep)

        assert len(alt) == N_cal

        alt_i = np.zeros(N_cal)
        alt_j = np.zeros(N_cal)

        for idx in range(N_cal):
            i,j = Ncal_dict[idx]
            alt_i[idx] = i
            alt_j[idx] = j

        self.alt_i = alt_i
        self.alt_j = alt_j


    @property
    def selection_registers(self) -> cft.SelectionRegisters:
        return merge_registers(self.i_register, self.j_register, self.v_register)

    @cached_property
    def sigma_mu_bitsize(self) -> int:
        return self.mu

    @cached_property
    def alternates_bitsize(self) -> int:
        return ceil(log2(self.N))

    @cached_property
    def keep_bitsize(self) -> int:
        return self.mu
        
    @cached_property
    def contiguous_bitsize(self) -> int:
        return ceil(log2(self.N_cal))

    @cached_property
    def junk_registers(self) -> cft.Registers:
        return cft.Registers.build(
            sigma_mu=self.sigma_mu_bitsize,
            alt_i=self.alternates_bitsize,
            alt_j=self.alternates_bitsize,
            keep=self.keep_bitsize,
            less_than_equal=1,
            theta=1, #for implementing sign
            herm_sym=1, #for implementing h_ij hermitian symmetry
            contiguous=self.contiguous_bitsize,
        )

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:
        sigma_qubits = quregs["sigma_mu"]
        alt_i_qubits = quregs["alt_i"]
        alt_j_qubits = quregs["alt_j"]
        keep_qubits = quregs["keep"]
        less_than_equal_qubits = quregs["less_than_equal"]
        theta_qubits = quregs["theta"]
        herm_sym_qubits = quregs["herm_sym"]
        i_qubits = quregs["i"]
        j_qubits = quregs["j"]
        v_qubits = quregs["v"]
        contiguous_qubits = quregs["contiguous"]

        #prepare V as normalized lambda_1|0> + lambda_2|1> state
        lambda_tot = np.sqrt(self.lambda_1 + self.lambda_2)
        c1 = np.sqrt(self.lambda_1) / lambda_tot
        v_rads = np.arccos(c1)
        yield cirq.Ry(rads=v_rads).on(*v_qubits)

        #flip V states for controlled operation on 1 instead of 0, have to undo flip in the end
        yield cirq.X.on(*v_qubits)

        #implement prepare(h_tilde) conditioned on V = 0:
        #start with uniform superposition of |i> and |j> states:
        yield prepare_uniform_superposition.PrepareUniformSuperposition(self.N).on(*i_qubits).controlled_by(*v_qubits)
        yield prepare_uniform_superposition.PrepareUniformSuperposition(self.N).on(*j_qubits).controlled_by(*v_qubits)

        #prepare sigma register on superposition
        yield cirq.H.on_each(*sigma_qubits)

        #prepare contiguous register
        con_gate = ContiguousRegisterGate(bitsize=self.alternates_bitsize, target_bitsize=self.contiguous_bitsize)
        yield con_gate.on(*i_qubits, *j_qubits, *contiguous_qubits).controlled_by(*v_qubits)

        #load QROM data
        qrom_gate = qrom.QROM(
            [self.alt_i, self.alt_j, self.keep, self.h_signs],
            (self.contiguous_bitsize,),
            (self.alternates_bitsize, self.alternates_bitsize, self.keep_bitsize, 1),
            num_controls = 1,
        )
        yield qrom_gate.on_registers(selection=contiguous_qubits, target0=alt_i_qubits, target1=alt_j_qubits,
            target2=keep_qubits, target3 = theta_qubits, control = v_qubits)

        #do inequality test
        yield arithmetic_gates.LessThanEqualGate(self.mu, self.mu).on(
            *keep_qubits, *sigma_qubits, *less_than_equal_qubits)

        #prepare sign
        yield cirq.Z.on(*theta_qubits)

        #swap registers i(j) and alt_i(j):
        yield swap_network.MultiTargetCSwap.make_on(control=v_qubits, target_x=alt_i_qubits, target_y=i_qubits)
        yield swap_network.MultiTargetCSwap.make_on(control=v_qubits, target_x=alt_j_qubits, target_y=j_qubits)

        #do symmetry swap
        yield cirq.H.on(*herm_sym_qubits)
        yield swap_network.MultiTargetCSwap.make_on(control=herm_sym_qubits, target_x=alt_j_qubits, target_y=j_qubits)

        #undo V flip
        yield cirq.X.on(*v_qubits)



class MTD_Prepare(cft.PrepareOracle):
    """
    MTD state preparation base class

    Given the electronic Hamiltonian:
    H = sum_{i,j = 1}^{N} h[i,j]*Fij + sum_{i,j,k,l = 1}^{N} g[i,j,k,l]*Fij*Fkl = H1 + H2
    where Fij = sum_{s ∈ {↑,↓}} a^†_{is} a_{js} are excitation operators acting on spacial orbitals i ∈ {1,...,N}

        MTD decomposition:
    H2 = sum_{w_tsr, s1, s2} Ω(w_tsr) * p(1,w_tsr,s1) * p(2,w_tsr,s1) * p(3,w_tsr,s2) * p(4,w_tsr,s2)
        For s1, s2 spins and p(i,w_tsr,s) unitary polynomials of Majoranas, defined as:
        
        p(m,w_tsr,s) = U(m,w_tsr) * γ_{1s,n_m} * U(m,w_tsr)^†, where:
            - U(m,w_tsr) is an orbital rotation (only need N entries since it only acts on orbital 1)
                Vector U[m,w_tsr,:] with length N will play a role (since everythin else is absorbed in w_tsr tensor)
                -> will use notation U[m,w_tsr,i] since this can be represented as an N-dimensional vector
            - n_m = 0 (if m = 1 or m = 3), nm = 1 (if m = 2 or m = 4)

    We'll use M as an index to indicate the different number of rotations. In general, M <= 4.
    General (M=4) MTD decomposition correspond to equality:
        - g[i,j,k,l] = sum_{w_tsr} Ω(w_tsr)*U(1,w_tsr)[i]*U(2,w_tsr)[j]*U(3,w_tsr)[k]*U(4,w_tsr)[l]

    inputs:
        - probability_epsilon: accuracy for loading coefficients
        - h_tilde: one-electron tensor that incorporates correction from two-electron tensor
            h_tilde[i,j] = h[i,j] + 2*sum_{k} g_[i,j,k,k] with g the 2-electron tensor in chemist notation
        - g: two-electron tensor, used for sanity checks and calculating 1-norm
        - w_shape: shape of w tensor
        - Omega_info: minimum information for building Ω tensor with dimensions [w_tsr_shape]
        - U_info: information for building U tensor with dimensions [N,4,w_shape]
            For different decompositions, these corresponds to:
        - flavour: string containing flavour of MTD, possible choices are in [DF, THC, arbitrary, CP4, MPS]:
            - DF: M = 2, w_shape = (R_DF, N, N) -> [r, p, q]
                H2 = sum_{r, p, q, s1, s2}  ϵ^{r}_p * ϵ^{r}_q * U(r) * n_{p,s1} * n_{q,s2} * U(r)^†
                    where n_{p,s}  is number operator on spin-orbital p,s
                    can be re-written as:
                H2 = sum_{r, p, q, s1, s2}  ϵ^{r}_p * ϵ^{r}_q * U(r,p) * n_{1,s1} * U(r,p)^† * U(r,q) * n_{1,s2} * U(r,q)^†
                Ω[r,p,q] = ϵ^{r}_p * ϵ^{r}_q
                    -> input Omega_info[r,p] = ϵ^{r}_p
                U[(1 or 2),r,p,q,i] = delta(q,0) * U(r,p)[i]
                U[(3 or 4),r,p,q,i] = delta(p,0) * U(r,q)[i]
                    -> input U_info[i,r,p] = U(r,p)[i] with shape (N, R_DF, N)
            - THC: M = 2, w_shape = (R_THC, R_THC)
                Ω[r1,r2] = ζ_{r1,r2} (symmetric to r1 <-> r2 exchange)
                    -> input Omega_info[z] = κ_{r1(z),r2(z)}*ζ_{r1(z),r2(z)} for z = r1 + r2*(r2-1)/2
                        and κ_{r1,r2} = (r1>r2 -> √2; r1 = r2 -> 1; r1 < r2 -> 0)
                        Omega length is R_THC*(R_THC+1)/2
                U[(1 or 2),r1,r2,i] = U(r1)[i] * delta(r2, 0)
                U[(3 or 4),r1,r2,i] = U(r2)[i] * delta(r1, 0)
                    -> input U_info[i,r] with shape (N, R_THC)
            - arbitrary MTD: M = 4, w_shape is arbitrary
                -> input Omega_info[w_tsr] = Ω[w_tsr]
                -> input U_info[i, m,w_tsr] with shape (N, 4, w_shape)
            - CP4: M=4, w_shape = W
                Ω[w] = Ω_w
                    -> input Omega_info(w) = Ω_w, shape (W,)
                U[i,m,w] = U(m,w)[i], for m = 1,2,3,4
                    -> input is U_info[i,m,w] tensor of shape (N, 4, W)
            - MPS: M = 4, w_shape = (A1, A2, A3)
                Ω[a1,a2,a3] = S(1)[a1]*S(2)[a2]*S(3)[a3]
                    -> input is tuple with three entries:
                        Omega_info = (S1, S2, S3), respective lengths (A1, A2, A3)
                        -> Ω[a1,a2,a3] = Omega_info[0][a1]*Omega_info[1][a2]*Omega_info[2][a3]
                U[1,a1,a2,a3,i] = U(1,a1)[i] * delta(a2,0) * delta(a3,0)
                U[2,a1,a2,a3,i] = U(2,a1,a2)[i] * delta(a3,0)
                U[3,a1,a2,a3,i] = U(3,a2,a3)[i] * delta(a1,0)
                U[4,a1,a2,a3,i] = U(4,a3)[i] * delta(a1,0) * delta(a2,0)
                    -> requires 4 tensors, each with shape
                        (N, A1)
                        (N, A1, A2)
                        (N, A2, A3)
                        (N, A3)
                    -> input tuple with four entries, showing shapes in []
                        U_info = (U1[N,A1], U2[N,A1,A2], U3[N,A2,A3], U4[N,A3])
        
    """

    def __init__(self, probability_epsilon : float, h_tilde : NDArray[float], g : NDArray[float],
        w_shape : tuple, Omega_info, U_info, flavour : str, debug=True):
        self.probability_epsilon = probability_epsilon
        flavour_list = ["MPS", "CP4", "THC", "DF", "arbitrary"]
        assert (flavour == flavour_list).any(), f"MTD flavour not implemented!"
        self.flavour == flavour

        self.Omega_info = Omega_info
        self.U_info = U_info
        self.h_tilde = h_tilde

        self.lambda_1 = np.sum(np.abs(h_tilde))
        self.lambda_2 = np.sum(np.abs(g))

        if flavour == "DF":
            assert len(w_shape) == 3
            R,N,N2 = w_shape
            self.R = R
            self.N = N
            assert N == N2
            assert np.shape(Omega_info) == (R,N)
            assert np.shape(U_info) == (N,R,N)

            if debug == True:
                Omega = np.zeros(R,N,N)
                U = np.zeros(4,R,N,N,N)

                for r in range(R):
                    for p in range(N):
                        epsilon_rp = Omega_info[r,p]
                        for q in range(N):
                            Omega[r,p,q] = epsilon_rp * Omega_info[r,q]
                        
                        for i in range(N):
                            for m in range(2):
                                #inplicit delta(q,0)
                                U[i,m,r,p,0] = U_info[i,r,p]
                            for m in range(2,4):
                                #inplicit delta(q,0)
                                U[i,m,r,0,p] = U_info[i,r,p]

        elif flavour == "THC":
            assert len(w_shape) == 1
            Rcal = w_shape[0]
            assert np.shape(Omega_info) == (Rcal,)

            R = int((np.sqrt(1+8*Rcal) + 1)/ 2)
            self.R = R

            N, R3 = np.shape(U_info)
            self.N = N
            assert R == R3

            if debug == True:
                Omega = np.zeros(R,R)
                U = np.zeros(N,4,R,R)

                idx = 0
                for r1 in range(R):
                    for r2 in range(r1):
                        if r1 == r2:
                            Omega[r1,r1] = Omega_info[idx]
                            for i in range(N):
                                for m in range(4):
                                    U[i,m,r1,r1] = U_info[i,r1]
                        else:
                            Omega[r1,r2] = Omega_info[idx] / np.sqrt(2)
                            Omega[r2,r1] = Omega[r1,r2]
                        idx += 1

                        for i in range(N):
                            for m in range(2):
                                #inplicit delta(r2,0)
                                U[i,m,r,0] = U_info[i,r]
                            for m in range(2,4):
                                #inplicit delta(r1,0)
                                U[i,m,0,r] = U_info[i,r]
        
        elif flavour == "MPS":
            assert len(w_shape) == 3
            A1,A2,A3 = w_shape
            self.A1 = A1
            self.A2 = A2
            self.A3 = A3

            assert len(Omega_info) == 3
            assert len(Omega_info[0]) == A1
            assert len(Omega_info[1]) == A2
            assert len(Omega_info[2]) == A3

            assert len(np.shape(U[0])) == 2
            assert len(np.shape(U[1])) == 3
            assert len(np.shape(U[2])) == 3
            assert len(np.shape(U[3])) == 2

            N,A1_bis = np.shape(U[0])
            assert A1_bis == A1

            Nbis,A1_bis,A2_bis = np.shape(U[1])
            assert A1_bis == A1
            assert A2_bis == A2
            assert Nbis == N

            Nbis,A2_bis,A3_bis = np.shape(U[2])
            assert A3_bis == A3
            assert A2_bis == A2
            assert Nbis == N

            Nbis,A3_bis = np.shape(U[3])
            assert A3_bis == A3
            assert A2_bis == A2
            assert Nbis == N

            self.N = N

            if debug == True:
                Omega = np.zeros(A1,A2,A3)
                U = np.zeros(4,A1,A2,A3,N)

                for a1 in range(A1):
                    for a2 in range(A2):
                        for a3 in range(A3):
                            Omega[a1,a2,a3] = Omega_info[0][a1] * Omega_info[1][a2] * Omega_info[2][a3]

                for i in range(N):
                    for a1 in range(A1):
                        U[i,0,a1,0,0] = U_info[0][i,a1]
                        for a2 in range(A2):
                            U[i,1,a1,a2,0] = U_info[1][i,a1,a2]

                    for a3 in range(A3):
                        U[i,3,0,0,a3] = U_info[3][i,a3]
                        for a2 in range(A2):
                            U[i,2,0,a2,a3] = U_info[2][i,a2,a3]

        elif flavour == "CP4":
            assert len(w_shape) == 1
            W = len(w_shape[0])
            self.W = W
            assert len(Omega_info) == W


            N, bis_4, W_bis = np.shape(U_info)
            assert bis_4 == 4
            assert W == W_bis
            self.N == N

            if debug == True:
                Omega = Omega_info
                U = U_info
        elif flavour == "arbitrary":
            assert np.shape(Omega_info) == w_shape
            num_ws = len(w_shape)

            u_shape = np.shape(U_info)
            N = u_shape[-1]
            self.N = N
            assert u_shape == (N, 4, *w_shape)

            if debug == True:
                Omega = Omega_info
                U = U_info

        assert np.shape(h_tilde) == (N,N)
        assert np.shape(g) == (N,N,N,N)

        if debug == True:
            g_recons = np.zeros(N,N,N,N)

            for idx in np.ndindex(w_shape):
                for i in range(N):
                    idx_0 = (i,0,) + idx
                    for j in range(N):
                        idx_1 = (j,1,) + idx
                        for k in range(N):
                            idx_2 = (k,2,) + idx
                            for l in range(N):
                                idx_3 = (l,3,) + idx
                                g_recons[i,j,k,l] += Omega[idx] * U[idx_0] * U[idx_1] * U[idx_2] * U[idx_3]
            g_diff = np.sum(np.abs((g - g_recons) ** 2))
            assert g_diff < debug_tiny, f"Reconstructed g is not within tolerance of g: 2-norm difference = {g_diff}"

    @cached_property
    def selection_registers(self) -> cft.SelectionRegisters:
        i_register = spacial_orbital_sel_register(self.N, "i")
        j_register = spacial_orbital_sel_register(self.N, "j")
        k_register = spacial_orbital_sel_register(self.N, "k")
        l_register = spacial_orbital_sel_register(self.N, "l")
        v_register = cft.SelectionRegister("v", 1, 2)

        return merge_registers(i_register, j_register, k_register, l_register, v_register)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:

        i_qubs = quregs["i"]
        j_qubs = quregs["j"]
        k_qubs = quregs["k"]
        l_qubs = quregs["l"]
        v_qubs = quregs["v"]

        one_el_gate = one_el_Prepare(self.h_tilde, self.lambda_2, probability_epsilon=self.probability_epsilon)
        yield one_el_gate.on_registers(i=i_qubs, j=j_qubs, v=v_qubs)
        #yield one_el_gate.on_registers(i=i_qubs, j=j_qubs, v=v_qubs, **one_el_gate.junk_registers.get_named_qubits())

        if self.flavour == "THC":
            two_el_gate = MTD_THC(self.probability_epsilon, self.R, self.N, self.Omega_info, self.U_info)
        elif self.flavour == "DF":
            two_el_gate = MTD_DF(self.probability_epsilon, self.R, self.N, self.Omega_info, self.U_info)
        elif self.flavour == "MPS":
            two_el_gate = MTD_MPS(self.probability_epsilon, self.A1, self.A2, self.A3, self.N, self.Omega_info, self.U_info)
        elif self.flavour == "CP4":
            two_el_gate = MTD_CP4(self.probability_epsilon, self.W, self.N, self.Omega_info, self.U_info)

        yield two_el_gate.on_registers(i = i_qubs, j = j_qubs, k = k_qubs, l = l_qubs, v = v_qubs)
        #yield one_el_gate.on_registers(**two_el_gate.registers.get_named_qubits())

class Prepare_U(cft.PrepareOracle):
    """
    Implements prepare circuit of U with dimensions U[N, multiplex_shape]

    inputs:
        - N: number of spacial orbitals
        - multiplex_shape: each entry corresponds to shape of register over which multiplexed operation happens
        - U_tsr: tensor corresponding to U[N, multiplex_shape] entries
        - probability_epsilon: accuracy with which coefficients are implemented
    """
    def __init__(self, N : int, multiplex_shape : tuple, U_tsr : NDArray[float], probability_epsilon = 1.0e-5):
        self.N = N
        self.multiplex_shape = multiplex_shape
        self.num_sels = len(multiplex_shape)

        first_U = np.zeros(N)
        for i in range(N):
            first_U[i] = np.abs(U_tsr[(i,) + self.num_sels * (0,)])

        alt_0, keep_0, self.mu = cft.linalg.preprocess_lcu_coefficients_for_reversible_sampling(
            lcu_coefficients=first_U, epsilon=probability_epsilon)
        alt_len = len(alt_0)
        keep_len = len(keep_0)

        alts = np.zeros((alt_len, *multiplex_shape), dtype=int)
        keeps = np.zeros((keep_len, *multiplex_shape), dtype=int)

        for idx in np.ndindex(multiplex_shape):
            my_U = np.zeros(N)
            for i in range(N):
                my_U[i] = np.abs(U_tsr[(i,) + idx])

            alt, keep, _ = cft.linalg.preprocess_lcu_coefficients_for_reversible_sampling(
                lcu_coefficients=my_U, epsilon=probability_epsilon)
            for i in range(alt_len):
                alts[(i,) + idx] = alt[i]
            for i in range(keep_len):
                keeps[(i,) + idx] = keep[i]

        self.alts = alts
        self.keeps = keeps
        self.U_signs = np.array(np.where(U_tsr<0, 1, 0), dtype=int)

    @cached_property
    def sigma_mu_bitsize(self) -> int:
        return self.mu

    @cached_property
    def alternates_bitsize(self) -> int:
        return ceil(log2(self.N))

    @cached_property
    def keep_bitsize(self) -> int:
        return self.mu

    @cached_property
    def junk_registers(self) -> cft.Registers:
        return cft.Registers.build(
            sigma_mu=self.sigma_mu_bitsize,
            alt=self.alternates_bitsize,
            keep=self.keep_bitsize,
            less_than_equal=1,
            theta=1)

    @property
    def control_registers(self) -> cft.Registers:
        return cft.Registers.build(control = 1)

    @cached_property
    def selection_registers(self) -> cft.SelectionRegisters:
        regs = [spacial_orbital_sel_register(self.N, "i")]
        for i in len(self.multiplex_shape):
            regs += [cft.SelectionRegister("selection"+str(i), ceil(log2(self.multiplex_shape[i])), self.multiplex_shape[i])]
        return merge_registers(regs)

    @cached_property
    def selection_bitsizes(self) -> tuple:
        bitsizes = []
        for reg in self.selection_registers:
            bitsizes += [reg.total_bits()]

        return tuple(bitsizes)

    @cached_property
    def registers(self) -> cft.Registers:
        return merge_registers(self.selection_registers, self.control_registers, self.junk_registers)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:

        sigma_mu, alt, keep = quregs["sigma_mu"], quregs["alt"], quregs["keep"]
        less_than_equal, theta = quregs["less_than_equal"], quregs["theta"]
        control = quregs["control"]

        selections = []
        for i in len(self.multiplex_shape):
            selections += [quregs["selection"+str(i)]]
        i_qubs = quregs["i"]

        yield prepare_uniform_superposition.PrepareUniformSuperposition(self.N).on(*i_qubs).controlled_by(*control)
        yield cirq.H.on_each(sigma_mu)

        qrom_gate = qrom.QROM(
            [self.alts, self.keeps, self.U_signs],
            self.selection_bitsizes,
            (self.alternates_bitsize, self.keep_bitsize, 1),
            num_controls = 1)

        qrom_sels_dict = {}
        qrom_sels_dict["selection0"] = i_qubs
        for i in range(len(self.multiplex_shape)):
            qrom_sels_dict["selection"+str(i+1)] = selections[i]

        yield qrom_gate.on_registers(**qrom_sels_dict, target0 = alt, target1 = keep, target2 = theta, control=control)

        yield arithmetic_gates.LessThanEqualGate(self.mu, self.mu).on(*keep, *sigma, *less_than_equal)
        yield swap_network.MultiTargetCSwap.make_on(control=control, target_x=alt, target_y=i_qubs)


class MTD_DF(cft.PrepareOracle):
    probability_epsilon : float
    R : int
    N : int
    Omega_info : NDArray[float]
    U_info : NDArray[float]

    @cached_property
    def selection_registers(self) -> cft.SelectionRegisters:
        i_register = spacial_orbital_sel_register(self.N, "i")
        j_register = spacial_orbital_sel_register(self.N, "j")
        k_register = spacial_orbital_sel_register(self.N, "k")
        l_register = spacial_orbital_sel_register(self.N, "l")
        v_register = cft.SelectionRegister("v", 1, 2)

        return merge_registers(i_register, j_register, k_register, l_register, v_register)

    @cached_property
    def R_bitsize(self) -> int:
        return ceil(log2(self.R))

    @cached_property
    def N_bitsize(self) -> int:
        return ceil(log2(self.N))

    @cached_property
    def get_omega_prep(self) -> int:
        alt_0, keep_0, mu = cft.linalg.preprocess_lcu_coefficients_for_reversible_sampling(
            lcu_coefficients=np.abs(Omega_info[0,:]), epsilon=probability_epsilon)

        len_alt = len(alt_0)
        len_keep = len(keep_0)
        alts = np.zeros(self.R, len_alt)
        keeps = np.zeros(self.R, len_keep)

        for r in range(1,self.R):
            alts[R,:], keeps[R,:], _ = cft.linalg.preprocess_lcu_coefficients_for_reversible_sampling(
                lcu_coefficients=np.abs(Omega_info[r,:]), epsilon=probability_epsilon)
        
        rp_signs = np.array(np.where(Omega_info<0, 1, 0), dtype=int)

        self.omega_alts = alts
        self.omega_keeps = keeps
        self.omega_rp_signs = rp_signs
        self.omega_mu = mu

        return self.omega_mu

    @cached_property
    def omega_sigma_mu_bitsize(self) -> int:
        return self.get_omega_prep

    @cached_property
    def omega_alternates_bitsize(self) -> int:
        return ceil(log2(self.N))

    @cached_property
    def omega_keep_bitsize(self) -> int:
        return self.get_omega_prep

    @cached_property
    def omega_junk_registers(self) -> cft.Registers:
        return cft.Registers.build(
            omega_sigma_mu_p=self.omega_sigma_mu_bitsize,
            omega_alt_p=self.omega_alternates_bitsize,
            omega_keep_p=self.omega_keep_bitsize,
            omega_less_than_equal_p=1,
            omega_sigma_mu_q=self.omega_sigma_mu_bitsize,
            omega_alt_q=self.omega_alternates_bitsize,
            omega_keep_q=self.omega_keep_bitsize,
            omega_less_than_equal_q=1,
            theta_p=1,
            theta_q=1)

    @cached_property
    def junk_registers(self) -> cft.Registers:
        w_regs = cft.Registers.build(r=self.R_bitsize, p=self.N_bitsize, q=self.N_bitsize)

        return merge_registers(w_regs, self.omega_junk_registers)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:

        i_qubs = quregs["i"]
        j_qubs = quregs["j"]
        k_qubs = quregs["k"]
        l_qubs = quregs["l"]
        v_qubs = quregs["v"]
        r_qubs = quregs["r"]
        p_qubs = quregs["p"]
        q_qubs = quregs["q"]
        theta_p_qubs = quregs["theta_p"]
        theta_q_qubs = quregs["theta_q"]
        omega_sigma_mu_qubs_p = quregs["omega_sigma_mu_qubits_p"]
        omega_alt_qubs_p = quregs["omega_alt_p"]
        omega_keep_qubs_p = quregs["omega_keep_p"]
        omega_less_than_equal_qubs_p = quregs["omega_less_than_equal_p"]
        omega_sigma_mu_qubs_q = quregs["omega_sigma_mu_qubits_q"]
        omega_alt_qubs_q = quregs["omega_alt_q"]
        omega_keep_qubs_q = quregs["omega_keep_q"]
        omega_less_than_equal_qubs_q = quregs["omega_less_than_equal_q"]

        omega_mu = self.omega_sigma_mu_bitsize
        #build prepare Omega circuit
        yield prepare_uniform_superposition.PrepareUniformSuperposition(self.R).on(*r_qubits).controlled_by(*v_qubits)
        
        ##controlled superposition epsilon(r,p) on both p and q registers
        yield cirq.H.on_each(*omega_sigma_mu_qubs_p)
        yield cirq.H.on_each(*omega_sigma_mu_qubs_q)

        qrom_gate = qrom.QROM(
            [self.alts, self.keeps, self.rp_signs],
            (self.R_bitsize, self.N_bitsize),
            (self.omega_alternates_bitsize, self.omega_keep_bitsize, 1),
        )
        yield qrom_gate.on_registers(selection0=r_qubs, selection1=p_qubs, target0=omega_alt_qubs_p, target1=omega_keep_qubs_p,
            target2=theta_p_qubs)
        yield qrom_gate.on_registers(selection0=r_qubs, selection1=q_qubs, target0=omega_alt_qubs_q, target1=omega_keep_qubs_q,
            target2=theta_q_qubs)

        yield arithmetic_gates.LessThanEqualGate(self.omega_mu, self.omega_mu).on(
            *omega_keep_qubs_p, *omega_sigma_qubs_p, *omega_less_than_equal_qubs_p)
        yield arithmetic_gates.LessThanEqualGate(self.omega_mu, self.omega_mu).on(
            *omega_keep_qubs_q, *omega_sigma_qubs_q, *omega_less_than_equal_qubs_q)

        yield cirq.Z.on_each(*theta_p_qubs, *theta_q_qubs)

        yield swap_network.MultiTargetCSwap.make_on(control=omega_less_than_equal_qubs_p, target_x=p_qubs, target_y=omega_alt_qubs_p)
        yield swap_network.MultiTargetCSwap.make_on(control=omega_less_than_equal_qubs_q, target_x=q_qubs, target_y=omega_alt_qubs_q)

        #build prepare Us circuit
        U_rp = Prepare_U(self.N, (self.R,self.N), self.U_info, probability_epsilon = self.probability_epsilon)
        yield U_rp.on_registers(i=i_qubs, selection0=r_qubs, selection1=p_qubs, control=v_qubs)
        yield U_rp.on_registers(i=j_qubs, selection0=r_qubs, selection1=p_qubs, control=v_qubs)
        yield U_rp.on_registers(i=k_qubs, selection0=r_qubs, selection1=q_qubs, control=v_qubs)
        yield U_rp.on_registers(i=l_qubs, selection0=r_qubs, selection1=q_qubs, control=v_qubs)

class MTD_THC(cft.PrepareOracle):
    probability_epsilon : float
    R : int
    N : int
    Omega_info : NDArray[float]
    U_info : NDArray[float]

    @cached_property
    def selection_registers(self) -> cft.SelectionRegisters:
        i_register = spacial_orbital_sel_register(self.N, "i")
        j_register = spacial_orbital_sel_register(self.N, "j")
        k_register = spacial_orbital_sel_register(self.N, "k")
        l_register = spacial_orbital_sel_register(self.N, "l")
        v_register = cft.SelectionRegister("v", 1, 2)

        return merge_registers(i_register, j_register, k_register, l_register, v_register)

    @cached_property
    def R_bitsize(self) -> int:
        return ceil(log2(self.R))

    @cached_property
    def N_bitsize(self) -> int:
        return ceil(log2(self.N))

    @cached_property
    def Rcal(self) -> int:
        R = self.R
        return int(R*(R+1)/2)

    @cached_property
    def Rcal_bitsize(self) -> int:
        return ceil(log2(self.Rcal))

    @cached_property
    def get_omega_prep(self) -> int:
        alt, self.keep, self.mu = cft.linalg.preprocess_lcu_coefficients_for_reversible_sampling(
            lcu_coefficients=np.abs(Omega_info), epsilon=probability_epsilon)
        
        rcal_dict = {}
        idx = 0
        for r1 in range(self.R):
            for r2 in range(r1):
                rcal_dict[idx] = (r1,r2)
                idx += 1
        alt_r1 = np.zeros_like(alt)
        alt_r2 = np.zeros_like(alt)

        for rcal in range(self.Rcal):
            r1,r2 = rcal_dict[rcal]
            alt_r1[rcal] = r1
            alt_r2[rcal] = r2

        self.alt_r1 = alt_r1
        self.alt_r2 = alt_r2

        self.rcal_signs = np.array(np.where(Omega_info<0, 1, 0), dtype=int)

        return self.mu

    @cached_property
    def omega_sigma_mu_bitsize(self) -> int:
        return self.get_omega_prep

    @cached_property
    def omega_alternates_bitsize(self) -> int:
        return ceil(log2(self.R))

    @cached_property
    def omega_keep_bitsize(self) -> int:
        return self.get_omega_prep

    @cached_property
    def omega_junk_registers(self) -> cft.Registers:
        return cft.Registers.build(
            omega_sigma_mu=self.omega_sigma_mu_bitsize,
            omega_alt_r1=self.omega_alternates_bitsize,
            omega_alt_r2=self.omega_alternates_bitsize,
            omega_contiguous=self.Rcal_bitsize,
            omega_less_than_equal_p=1,
            theta_contiguous=1,
            omega_keep=self.omega_keep_bitsize,
            omega_symmetry=1)

    @cached_property
    def junk_registers(self) -> cft.Registers:
        w_regs = cft.Registers.build(r1=self.R_bitsize, r2=self.N_bitsize)

        return merge_registers(w_regs, self.omega_junk_registers)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:

        i_qubs = quregs["i"]
        j_qubs = quregs["j"]
        k_qubs = quregs["k"]
        l_qubs = quregs["l"]
        v_qubs = quregs["v"]
        r1_qubs = quregs["r1"]
        r2_qubs = quregs["r2"]
        contiguous = quregs["omega_contiguous"]
        theta = quregs["theta_contiguous"]
        omega_sigma_mu = quregs["omega_sigma_mu"]
        omega_alt_r1 = quregs["omega_alt_r1"]
        omega_keep = quregs["omega_keep"]
        omega_less_than_equal = quregs["omega_less_than_equal"]
        omega_alt_r2 = quregs["omega_alt_r2"]
        omega_sym=quregs["omega_symmetry"]

        omega_mu = self.omega_sigma_mu_bitsize
        #build prepare Omega circuit
        yield prepare_uniform_superposition.PrepareUniformSuperposition(self.R).on(*r1_qubs).controlled_by(*v_qubs)
        yield prepare_uniform_superposition.PrepareUniformSuperposition(self.R).on(*r2_qubs).controlled_by(*v_qubs)
        yield cirq.H.on_each(*omega_sigma_mu)

        con_gate = ContiguousRegisterGate(bitsize=self.omega_alternates_bitsize, target_bitsize=self.Rcal_bitsize)
        yield con_gate.on(*r1_qubits, *r2_qubits, *contiguous).controlled_by(*v_qubs)

        qrom_gate = qrom.QROM(
            [self.alt_r1, self.alt_r2, self.keep, self.rcal_signs],
            (self.Rcal_bitsize,),
            (self.omega_alternates_bitsize, self.omega_alternates_bitsize, self.omega_keep_bitsize, 1),
            num_controls = 1
        )
        yield qrom_gate.on_registers(selection=contiguous, target0=omega_alt_r1, target1=omega_alt_r2,
            target2=omega_keep, target3=theta, control=v_qubs)

        yield cirq.Z.on(*theta)
        yield arithmetic_gates.LessThanEqualGate(self.mu, self.mu).on(*omega_keep, *omega_sigma_mu,
            *omega_less_than_equal)

        yield swap_network.MultiTargetCSwap.make_on(control=v_qubs, target_x=omega_alt_r1, target_y=r1_qubs)
        yield swap_network.MultiTargetCSwap.make_on(control=v_qubs, target_x=omega_alt_r2, target_y=r2_qubs)

        yield cirq.H.on(*omega_sym)
        yield swap_network.MultiTargetCSwap.make_on(control=omega_sym, target_x=r1_qubs, target_y=r2_qubs)


        #build prepare Us circuit
        U_r = Prepare_U(self.N, (self.R,), self.U_info, probability_epsilon = self.probability_epsilon)
        yield U_r.on_registers(i=i_qubs, selection0=r1_qubs, control=v_qubs)
        yield U_r.on_registers(i=j_qubs, selection0=r1_qubs, control=v_qubs)
        yield U_r.on_registers(i=k_qubs, selection0=r2_qubs, control=v_qubs)
        yield U_r.on_registers(i=l_qubs, selection0=r2_qubs, control=v_qubs)


class MTD_CP4(cft.PrepareOracle):
    probability_epsilon : float
    W : int
    N : int
    Omega_info : NDArray[float]
    U_info : NDArray[float]

    @cached_property
    def selection_registers(self) -> cft.SelectionRegisters:
        i_register = spacial_orbital_sel_register(self.N, "i")
        j_register = spacial_orbital_sel_register(self.N, "j")
        k_register = spacial_orbital_sel_register(self.N, "k")
        l_register = spacial_orbital_sel_register(self.N, "l")
        v_register = cft.SelectionRegister("v", 1, 2)

        return merge_registers(i_register, j_register, k_register, l_register, v_register)

    @cached_property
    def W_bitsize(self) -> int:
        return ceil(log2(self.W))

    @cached_property
    def N_bitsize(self) -> int:
        return ceil(log2(self.N))

    @cached_property
    def get_omega_prep(self) -> int:
        self.alt, self.keep, self.mu = cft.linalg.preprocess_lcu_coefficients_for_reversible_sampling(
            lcu_coefficients=np.abs(Omega_info), epsilon=probability_epsilon)
        
        self.w_signs = np.array(np.where(Omega_info<0, 1, 0), dtype=int)

        return self.mu

    @cached_property
    def omega_sigma_mu_bitsize(self) -> int:
        return self.get_omega_prep

    @cached_property
    def omega_alternates_bitsize(self) -> int:
        return self.W_bitsize

    @cached_property
    def omega_keep_bitsize(self) -> int:
        return self.get_omega_prep

    @cached_property
    def omega_junk_registers(self) -> cft.Registers:
        return cft.Registers.build(
            omega_sigma_mu=self.omega_sigma_mu_bitsize,
            omega_alt=self.omega_alternates_bitsize,
            omega_keep=self.omega_keep_bitsize,
            omega_less_than_equal=1,
            theta_p=1)
    @cached_property
    def junk_registers(self) -> cft.Registers:
        return cft.Registers.build(w=self.W_bitsize)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:

        i_qubs = quregs["i"]
        j_qubs = quregs["j"]
        k_qubs = quregs["k"]
        l_qubs = quregs["l"]
        v_qubs = quregs["v"]
        w_qubs = quregs["w"]
        theta = quregs["theta"]
        sigma_mu, alt, keep = quregs["omega_sigma_mu"], quregs["omega_alt"], quregs["omega_keep"]
        less_than_equal = quregs["omega_less_than_equal"]

        #build prepare Omega circuit
        yield prepare_uniform_superposition.PrepareUniformSuperposition(self.W).on(*w_qubs).controlled_by(*v_qubs)
        yield cirq.H.on_each(*sigma_mu)

        qrom_gate = qrom.QROM(
            [self.alts, self.keeps, self.w_signs],
            (self.W_bitsize,),
            (self.omega_alternates_bitsize, self.omega_keep_bitsize, 1),
            num_controls = 1
        )
        yield qrom_gate.on_registers(selection=w_qubs, target0=alt, target1=keep,target2=theta, control=v_qubs)
        yield arithmetic_gates.LessThanEqualGate(self.mu, self.mu).on(*keep, *sigma, *less_than_equal)
        yield cirq.Z.on_each(*theta)
        yield swap_network.MultiTargetCSwap.make_on(control=less_than_equal, target_x=w_qubs, target_y=alt)

        #build prepare Us circuit
        U_1w = Prepare_U(self.N, (self.W,), self.U_info[0], probability_epsilon = self.probability_epsilon)
        U_2w = Prepare_U(self.N, (self.W,), self.U_info[1], probability_epsilon = self.probability_epsilon)
        U_3w = Prepare_U(self.N, (self.W,), self.U_info[2], probability_epsilon = self.probability_epsilon)
        U_4w = Prepare_U(self.N, (self.W,), self.U_info[3], probability_epsilon = self.probability_epsilon)
        yield U_1w.on_registers(i=i_qubs, selection0=w_qubs, control=v_qubs)
        yield U_2w.on_registers(i=j_qubs, selection0=w_qubs, control=v_qubs)
        yield U_3w.on_registers(i=k_qubs, selection0=w_qubs, control=v_qubs)
        yield U_4w.on_registers(i=l_qubs, selection0=w_qubs, control=v_qubs)


class MTD_MPS(cft.PrepareOracle):
    probability_epsilon : float
    A1 : int
    A2 : int
    A3 : int
    N : int
    Omega_info : NDArray[float]
    U_info : NDArray[float]

    @cached_property
    def selection_registers(self) -> cft.SelectionRegisters:
        i_register = spacial_orbital_sel_register(self.N, "i")
        j_register = spacial_orbital_sel_register(self.N, "j")
        k_register = spacial_orbital_sel_register(self.N, "k")
        l_register = spacial_orbital_sel_register(self.N, "l")
        v_register = cft.SelectionRegister("v", 1, 2)

        return merge_registers(i_register, j_register, k_register, l_register, v_register)

    @cached_property
    def A1_bitsize(self) -> int:
        return ceil(log2(self.A1))

    @cached_property
    def A2_bitsize(self) -> int:
        return ceil(log2(self.A2))

    @cached_property
    def A3_bitsize(self) -> int:
        return ceil(log2(self.A3))

    @cached_property
    def N_bitsize(self) -> int:
        return ceil(log2(self.N))

    @cached_property
    def get_omega_preps(self) -> int:
        self.alt_1, self.keep_1, self.mu_1 = cft.linalg.preprocess_lcu_coefficients_for_reversible_sampling(
            lcu_coefficients=np.abs(Omega_info[0]), epsilon=probability_epsilon)
        self.a1_signs = np.array(np.where(Omega_info[0]<0, 1, 0), dtype=int)

        self.alt_2, self.keep_2, self.mu_2 = cft.linalg.preprocess_lcu_coefficients_for_reversible_sampling(
            lcu_coefficients=np.abs(Omega_info[1]), epsilon=probability_epsilon)
        self.a2_signs = np.array(np.where(Omega_info[1]<0, 1, 0), dtype=int)

        self.alt_3, self.keep_3, self.mu_3 = cft.linalg.preprocess_lcu_coefficients_for_reversible_sampling(
            lcu_coefficients=np.abs(Omega_info[2]), epsilon=probability_epsilon)
        self.a3_signs = np.array(np.where(Omega_info[2]<0, 1, 0), dtype=int)

        return (self.mu_1, self.mu_2, self.mu_3)

    @cached_property
    def sigma_mu_bitsizes(self) -> int:
        return self.get_omega_prep

    @cached_property
    def alternates_bitsizes(self) -> int:
        return (self.A1_bitsize, self.A2_bitsize, self.A3_bitsize)

    @cached_property
    def keep_bitsizes(self) -> int:
        return self.get_omega_prep

    @cached_property
    def omega_junk_registers(self) -> cft.Registers:
        (mu1, mu2, mu3) = self.get_omega_prep
        return cft.Registers.build(
            sigma_a1=mu1,
            sigma_a2=mu2,
            sigma_a3=mu3,
            alt_a1=self.A1_bitsize,
            alt_a2=self.A2_bitsize,
            alt_a3=self.A3_bitsize,
            keep_a1=mu1,
            keep_a2=mu2,
            keep_a3=mu3,
            lte_a1=1,
            lte_a2=1,
            lte_a3=1,
            theta_a1=1,
            theta_a2=1,
            theta_a3=1)

    @cached_property
    def junk_registers(self) -> cft.Registers:
        return merge_registers(cft.Registers.build(a1=self.A1_bitsize, a2=self.A1_bitsize, a3=self.A1_bitsize),
            self.omega_junk_registers)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:

        i_qubs = quregs["i"]
        j_qubs = quregs["j"]
        k_qubs = quregs["k"]
        l_qubs = quregs["l"]
        v_qubs = quregs["v"]
        a1_qubs = quregs["a1"]
        a2_qubs = quregs["a2"]
        a3_qubs = quregs["a3"]
        theta_a1 = quregs["theta_a1"]
        theta_a2 = quregs["theta_a2"]
        theta_a3 = quregs["theta_a3"]
        sigma_a1, alt_a1, keep_a1 = quregs["sigma_a1"], quregs["alt_a1"], quregs["keep_a1"]
        lte_a1 = quregs["lte_a1"]
        sigma_a2, alt_a2, keep_a2 = quregs["sigma_a2"], quregs["alt_a2"], quregs["keep_a2"]
        lte_a2 = quregs["lte_a2"]
        sigma_a3, alt_a3, keep_a3 = quregs["sigma_a3"], quregs["alt_a3"], quregs["keep_a3"]
        lte_a3 = quregs["lte_a3"]

        #build prepare Omega circuit
        yield prepare_uniform_superposition.PrepareUniformSuperposition(self.A1).on(*a1_qubs).controlled_by(*v_qubs)
        yield prepare_uniform_superposition.PrepareUniformSuperposition(self.A2).on(*a2_qubs).controlled_by(*v_qubs)
        yield prepare_uniform_superposition.PrepareUniformSuperposition(self.A3).on(*a3_qubs).controlled_by(*v_qubs)

        yield cirq.H.on_each(*sigma_a1)
        yield cirq.H.on_each(*sigma_a2)
        yield cirq.H.on_each(*sigma_a3)

        qrom_gate_a1 = qrom.QROM(
            [self.alt_a1, self.keep_a1, self.a1_signs],
            (self.A1_bitsize,),
            (self.A1.bitsize, self.mu_1, 1),
            num_controls = 1
        )
        yield qrom_gate.on_registers(selection=a1_qubs, target0=alt_a1, target1=keep_a1, target2=theta_a1, control=v_qubs)
        
        qrom_gate_a2 = qrom.QROM(
            [self.alt_a2, self.keep_a2, self.a2_signs],
            (self.A2_bitsize,),
            (self.A2.bitsize, self.mu_2, 1),
            num_controls = 1
        )
        yield qrom_gate.on_registers(selection=a2_qubs, target0=alt_a2, target1=keep_a2, target2=theta_a2, control=v_qubs)
        
        qrom_gate_a3 = qrom.QROM(
            [self.alt_a3, self.keep_a3, self.a3_signs],
            (self.A3_bitsize,),
            (self.A3.bitsize, self.mu_3, 1),
            num_controls = 1
        )
        yield qrom_gate.on_registers(selection=a3_qubs, target0=alt_a3, target1=keep_a3, target2=theta_a3, control=v_qubs)
              

        yield arithmetic_gates.LessThanEqualGate(self.mu_1, self.mu_1).on(*keep_a1, *sigma_a1, *lte_a1)
        yield arithmetic_gates.LessThanEqualGate(self.mu_2, self.mu_2).on(*keep_a2, *sigma_a2, *lte_a2)
        yield arithmetic_gates.LessThanEqualGate(self.mu_3, self.mu_3).on(*keep_a3, *sigma_a3, *lte_a3)

        yield cirq.Z.on_each(*theta_a1)
        yield cirq.Z.on_each(*theta_a2)
        yield cirq.Z.on_each(*theta_a3)

        yield swap_network.MultiTargetCSwap.make_on(control=lte_a1, target_x=a1_qubs, target_y=alt_a1)
        yield swap_network.MultiTargetCSwap.make_on(control=lte_a2, target_x=a2_qubs, target_y=alt_a2)
        yield swap_network.MultiTargetCSwap.make_on(control=lte_a3, target_x=a3_qubs, target_y=alt_a3)

        #build prepare Us circuit
        U_1_a1 = Prepare_U(self.N, (self.A1,), self.U_info[0], probability_epsilon = self.probability_epsilon)
        U_2_a1a2 = Prepare_U(self.N, (self.A1, self.A2), self.U_info[1], probability_epsilon = self.probability_epsilon)
        U_3_a2a3 = Prepare_U(self.N, (self.A2, self.A3), self.U_info[2], probability_epsilon = self.probability_epsilon)
        U_4_a3 = Prepare_U(self.N, (self.A3,), self.U_info[3], probability_epsilon = self.probability_epsilon)
        yield U_1_a1.on_registers(i=i_qubs, selection0=a1_qubs, control=v_qubs)
        yield U_2_a1a2.on_registers(i=j_qubs, selection0=a1_qubs, selection1=a2_qubs ,control=v_qubs)
        yield U_3_a2a3.on_registers(i=k_qubs, selection0=a2_qubs, selection1=a3_qubs, control=v_qubs)
        yield U_4_a3.on_registers(i=l_qubs, selection0=a3_qubs, control=v_qubs)

