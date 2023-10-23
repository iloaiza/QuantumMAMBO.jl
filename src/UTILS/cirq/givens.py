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

from utils import *

def bin_vec_to_int(bin_vec):
    #transforms binary vector (e.g. [0,1,1,1,0,0,1]) to integer with big-endian notation
    return int("".join(str(x) for x in bin_vec), 2)

def theta_to_vec(theta, beta):
    #approximated theta to beta binary bits for QROM of theta for implementing multiplexed gates
    theta_norm = (theta / (2*pi)) % 1 #theta between 0 and 1
    theta_vec = np.zeros(beta, dtype=int)
    theta_rem = theta_norm
    for i in range(beta):
        theta_try = theta_rem - 2**(-(i+1))
        if theta_try > 0:
            theta_vec[i] = 1
            theta_rem = theta_try
    
    return theta_vec

def iterative_angle_load(thetas, beta):
    #returns binary vector angles for iterative QROM, where M-th step combines unloading (M-1) and loading M in a single data load
    #last entry will give QROM for final unloading, returning clean ancilla basis
    num_thetas = len(thetas)
    vecs = np.zeros((num_thetas + 1, beta), dtype=int)

    for i in range(num_thetas):
        vecs[i,:] = theta_to_vec(thetas[i], beta)

    for i in reversed(range(1,num_thetas+1)):
        vecs[i,:] = (vecs[i,:] - vecs[i-1,:]) % 2

    return vecs

def theta_to_int(theta, beta):
    #transforms theta angle into integer for QROM using beta bits of accuray
    return bin_vec_to_int(theta_to_vec(theta, beta))

def int_to_vec(i, beta):
    #transforms integer i into big-endian binary vector with beta bits
    assert i < 2**beta, f"integer {i} should be smaller than 2**{beta} for transforming into big-endian vector"

    vec = np.zeros(beta, dtype=int)

    s = format(i, 'b').zfill(beta)
    for (idx,bin_num) in enumerate(s):
        vec[idx] = int(bin_num)

    return vec

@attr.frozen
class MultiplexedRzTheta(cft.GateWithRegisters):
    #apply controlled rotation Rz(theta) with approximation of angle theta using beta bits
    theta_register : cft.Register
    target_register : cft.Register
    coeff : int #coefficient multiplying all thetas

    @cached_property
    def beta(self) -> int:
        return len(get_qubits(self.theta_register))

    @property
    def registers(self) -> cft.Registers:
        return merge_registers(self.theta_register, self.target_register)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:
        theta_bits = get_qubits(self.theta_register)
        target_bits = get_qubits(self.target_register)
        for i in range(self.beta):
            rz_theta_i = cirq.Rz(rads = -self.coeff * pi/(2**(i-2)))
            rz_op = rz_theta_i.on(*target_bits)
            yield rz_op.controlled_by(theta_bits[i])


class MultiplexedPauliExponential(cft.GateWithRegisters):
    #implements exponential exp(-iP*theta) multiplexed by "theta" quantum register
    #P is given as integer vector with 0=id, 1=x, 2=y, 3=z (e.g. x1*y2*z4 = [1,2,0,3])
    #uses stair algorithm to decompose Pauli into CNOT stair with Rz(theta) multiplexed rotation
    #coeff is a constant multiplying the angle theta
    
    def __init__(self, pauli_vec, theta_register, target_register, coeff = 1):
        assert pauli_vec[0] != 0, f"First entry of pauli vector needs to be different to identity, not implemented!"
        assert len(pauli_vec) == len(get_qubits(target_register)), f"Pauli vector needs to have same length as number of target qubits"
        self.pauli_vec = pauli_vec
        self.theta_register = theta_register
        self.target_register = target_register
        self.coeff = coeff
    

    @property
    def registers(self):
        return merge_registers(self.theta_register, self.target_register)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:

        target_qubs = get_qubits(self.target_register)
        theta_qubs = get_qubits(self.theta_register)

        last_qubit_register = qubit_to_register(target_qubs[-1])

        num_targs = len(target_qubs)

        #add left layer of diagonalization to z
        rz_plus_gate = cirq.Rz(rads = pi/2) #for bringing y Pauli term to Z form
        for i in range(num_targs):
            if self.pauli_vec[i] == 1:
                yield cirq.H.on(target_qubs[i])
            elif self.pauli_vec[i] == 2:
                yield cirq.H.on(target_qubs[i])
                yield rz_plus_gate.on(target_qubs[i])

        #add CNOT cascade
        for i in reversed(range(1, num_targs)):
            if self.pauli_vec[i] == 0:
                yield cirq.SWAP.on(target_qubs[i-1], target_qubs[i])
            else:
                yield cirq.CNOT.on(target_qubs[i-1], target_qubs[i])

        Rz_rotation = MultiplexedRzTheta(theta_register = self.theta_register, target_register = last_qubit_register, coeff = self.coeff)
        yield Rz_rotation.on(*get_qubits(Rz_rotation.registers))

        #do second part of cascade
        for i in range(1, num_targs):
            if self.pauli_vec[i] == 0:
                yield cirq.SWAP.on(target_qubs[i-1], target_qubs[i])
            else:
                yield cirq.CNOT.on(target_qubs[i-1], target_qubs[i])    

        #add right layer of diagonalization to z
        rz_minus_gate = cirq.Rz(rads = -pi/2) #for bringing y Pauli term to Z form
        for i in range(num_targs):
            if self.pauli_vec[i] == 1:
                yield cirq.H.on(target_qubs[i])
            elif self.pauli_vec[i] == 2:
                yield cirq.H.on(target_qubs[i])
                yield rz_minus_gate.on(target_qubs[i])


class MultiplexedEFTGivens(cft.GateWithRegisters):
    """
    Early fault-tolerant implenentation of Givens rotation on target register
    (early fault-tolerant part: uses beta-bit register for rotation angles, will run QROM N times for N the number of bits in target register)
    beta: number of qubits used to represent Givens rotation angle, obtained from accuracy epsilon

    Inputs:
        - thetas_arr: Givens angles array. For each dimension besides the last one (e.g. dims(thetas_arr[i,j,n]) = 2) expects selection register with same length
            last dimension has N-1 angles for Givens rotations of N-bits target register
        - dual: True for implementing U(u,0)*U(u,1), otherwise expects 0 or 1 for each different Majorana rotation class
        - selection_registers: iterable of selection registers, should have same total iteration_length as thetas_arr has elements
        - target_register: register where Givens rotations will be applied, has N qubits
        - theta_register: clean qubit register where rotation angles will be stored, uses length beta (see epsilon below) and will do a total of N-1 QROMs
        - epsilon: tolerance for rotation accuracy, approximate each Givens rotation using beta=ceil(0.5 + log2(N*pi/epsilon)) angles
        - dagger: whether inverse rotation should be applied (True) or just regular rotation (False)
    """
    def __init__(self, thetas_arr, dual = True, selection_registers = None,
            target_register = None, theta_register = None, epsilon = 1.0e-4, dagger : bool = False):

        self.dagger = dagger
        if dagger:
            self.thetas_arr = -thetas_arr
        else:
            self.thetas_arr = thetas_arr

        if dual == True:
            self.dual = True
        else:
            self.dual = False
            if dual != 0:
                assert dual == 1, f"if dual is not true it should be 0 or 1"
                self.m = dual

        thetas_shape = np.shape(thetas_arr)
        N = thetas_shape[-1] + 1
        self.N = N

        if type(target_register) == type(None):
            self.target_register = cft.Register("target", self.N)
        else:
            assert self.N == target_register.total_bits()
            self.target_register = target_register

        self.beta = ceil(0.5 + log2(N*pi/epsilon))
        if type(theta_register) == type(None):
            self.theta_register = cft.Register("Givens_theta", self.beta)
        else:
            assert theta_register.total_bits() == self.beta
            self.theta_register = theta_register


        num_sels = len(thetas_shape) - 1
        if type(selection_registers) == type(None):
            sel_strings = [""] * num_sels
            for i in range(num_sels):
                sel_strings[i] = "selection_" + str(i+1) + "_"
            self.selection_registers = merge_registers(set(cft.SelectionRegister(sel_strings[i], thetas_shape[i], ceil(log2(thetas_shape[i]))) for i in range(num_sels)))
        else:
            assert num_sels == len(selection_registers)
            for i in range(num_sels):
                assert selection_registers[i].iteration_length == thetas_shape[i]
            self.selection_registers = selection_registers

    @property
    def registers(self) -> cft.Registers:
            return merge_registers(self.selection_registers, self.target_register, self.theta_register)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid], 
    ) -> cirq.OP_TREE:

        #Start by generating integer values for QROMs
        thetas_shape = np.shape(self.thetas_arr)
        multiplex_shape = thetas_shape[:-1]
        
        thetas_int_vals = np.zeros(thetas_shape)
        for (idx, theta) in np.ndenumerate(self.thetas_arr):
            thetas_int_vals[idx] = theta_to_int(theta, self.beta)

        iterative_thetas = np.zeros((*multiplex_shape, self.N))
        for idx in np.ndindex(multiplex_shape):
            theta_bin_vecs = iterative_angle_load(self.thetas_arr[idx + (...,)], self.beta)
            for n in range(self.N):
                iterative_thetas[idx + (n,)] = bin_vec_to_int(theta_bin_vecs[n, :])

        #generated N-1 layers of rotations
        theta_qubs = get_qubits(self.theta_register)
        target_qubs = get_qubits(self.target_register)

        multiplex_qubs_arr = []
        for i in range(len(multiplex_shape)):
            multiplex_qubs_arr += [get_qubits(self.selection_registers[i])]

        ## prepare selection dictionary for quick passing to QROM
        sel_dict = {}
        for i in range(len(multiplex_shape)):
            sel_string_i = "selection" + str(i)
            sel_dict[sel_string_i] = multiplex_qubs_arr[i]

        if self.dagger == False:
            for n in range(self.N-1):
                qrom_i = qrom.QROM.build(iterative_thetas[..., n])
                yield qrom_i.on_registers(target0 = theta_qubs, **sel_dict)

                n_nplus_register = qubits_to_register(target_qubs[n:n+2])
                if self.dual == True or ((self.dual == False) and self.m == 0):
                    Vui = MultiplexedPauliExponential(pauli_vec = [2,1], theta_register = self.theta_register, target_register = n_nplus_register, coeff = -1)
                    Vui_regs = Vui.registers
                    yield Vui.on(*get_qubits(Vui_regs))

                if self.dual == True or ((self.dual == False) and self.m == 1):
                    Vui = MultiplexedPauliExponential(pauli_vec = [1,2], theta_register = self.theta_register, target_register = n_nplus_register, coeff = 1)
                    Vui_regs = Vui.registers
                    yield Vui.on(*get_qubits(Vui_regs))

            #unload QROM final step
            qrom_i = qrom.QROM.build(iterative_thetas[..., -1])
            yield qrom_i.on_registers(target0 = theta_qubs, **sel_dict)
        else:
            #do all operations in opposite direction
            qrom_i = qrom.QROM.build(iterative_thetas[..., -1])
            yield qrom_i.on_registers(target0 = theta_qubs, **sel_dict)
        
            for n in reversed(range(self.N-1)):
                qrom_i = qrom.QROM.build(iterative_thetas[..., n])
                yield qrom_i.on_registers(target0 = theta_qubs, **sel_dict)

                n_nplus_register = qubits_to_register(target_qubs[n:n+2])
                if self.dual == True or ((self.dual == False) and self.m == 0):
                    Vui = MultiplexedPauliExponential(pauli_vec = [2,1], theta_register = self.theta_register, target_register = n_nplus_register, coeff = -1)
                    Vui_regs = Vui.registers
                    yield Vui.on(*get_qubits(Vui_regs))

                if self.dual == True or ((self.dual == False) and self.m == 1):
                    Vui = MultiplexedPauliExponential(pauli_vec = [1,2], theta_register = self.theta_register, target_register = n_nplus_register, coeff = 1)
                    Vui_regs = Vui.registers
                    yield Vui.on(*get_qubits(Vui_regs))

















