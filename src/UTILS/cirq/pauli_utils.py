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
from collections.abc import Iterable

class PauliExponential(cft.GateWithRegisters):
    #implements exponential exp(-iP*theta)
    #P is given as integer vector with 0=id, 1=x, 2=y, 3=z (e.g. x1*y2*z4 = [1,2,0,3])
    #uses stair algorithm to decompose Pauli into CNOT stair with Rz(theta) controlled rotation
    #coeff is a constant multiplying the angle theta
    def __init__(self, pauli_vec, ctl_register, target_registers, theta):
        pauli_vec_red = np.copy(pauli_vec)
        target_reg_red = target_registers
        #eliminate identities from start:
        if pauli_vec[0] == 0:
            init_pauli_int = 0
            while init_pauli_int == 0:
                pauli_vec_red = pauli_vec_red[1:]
                target_qubs = get_qubits(target_reg_red)[1:]
                target_reg_red = qubits_to_register(target_qubs)
                init_pauli_int = pauli_vec_red[0]

        assert pauli_vec_red[0] != 0, f"First entry of pauli vector needs to be different to identity, not implemented!"
        assert len(pauli_vec_red) == len(get_qubits(target_reg_red)), f"Pauli vector needs to have same length as number of target qubits"
        self.pauli_vec = pauli_vec_red
        self.control_register = ctl_register
        self.target_registers = target_reg_red
        self.theta = theta

    
    @property
    def registers(self):
        return merge_registers(self.control_register, self.target_registers)

    def on_qubits(target_qubs, ctl_qubs, pauli_vec, theta):
        last_qubit = target_qubs[-1]
        num_targs = len(target_qubs)

        starting_idx = 0
        while pauli_vec[starting_idx] == 0:
            starting_idx += 1

        assert starting_idx < num_targs

        #add left layer of diagonalization to z
        rz_plus_gate = cirq.Rz(rads = pi/2) #for bringing y Pauli term to Z form
        for i in range(starting_idx, num_targs):
            if pauli_vec[i] == 1:
                yield cirq.H.on(target_qubs[i])
            elif pauli_vec[i] == 2:
                yield cirq.H.on(target_qubs[i])
                yield rz_plus_gate.on(target_qubs[i])

        #add CNOT cascade
        for i in reversed(range(starting_idx+1, num_targs)):
            if pauli_vec[i] == 0:
                yield cirq.SWAP.on(target_qubs[i-1], target_qubs[i])
            else:
                yield cirq.CNOT.on(target_qubs[i-1], target_qubs[i])

        Rz_rotation = cirq.Rz(rads = 2*theta)

        if isinstance(ctl_qubs, Iterable):
            yield Rz_rotation.on(last_qubit).controlled_by(*ctl_qubs)
        else:
            yield Rz_rotation.on(last_qubit)

        #do second part of cascade
        for i in range(starting_idx + 1, num_targs):
            if pauli_vec[i] == 0:
                yield cirq.SWAP.on(target_qubs[i-1], target_qubs[i])
            else:
                yield cirq.CNOT.on(target_qubs[i-1], target_qubs[i])    

        #add right layer of diagonalization to z
        rz_minus_gate = cirq.Rz(rads = -pi/2) #for bringing y Pauli term to Z form
        for i in range(num_targs):
            if pauli_vec[i] == 1:
                yield cirq.H.on(target_qubs[i])
            elif pauli_vec[i] == 2:
                yield cirq.H.on(target_qubs[i])
                yield rz_minus_gate.on(target_qubs[i])

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:

        target_qubs = get_qubits(self.target_registers)
        ctl_qubs = get_qubits(self.control_register)

        yield PauliExponential.on_qubits(target_qubs, theta_qubs, self.pauli_vec, self.theta)
        