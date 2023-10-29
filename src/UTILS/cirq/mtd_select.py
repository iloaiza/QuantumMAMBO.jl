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

class Select_Qija(cft.SelectOracle):
    """
    controlled application of Qija
    """
    def __init__(self, N : int):
        self.N = N

    @property
    def control_registers(self) -> cft.Registers:
        return cft.Registers.build(control=1)

    @property
    def selection_registers(self) -> cft.SelectionRegisters:
        i_reg = spacial_orbital_sel_register(self.N, "i")
        j_reg = spacial_orbital_sel_register(self.N, "j")
        spin_reg = spin_sel_register("spin")
        return merge_registers(i_reg, j_reg, spin_reg)

    @property
    def target_registers(self) -> cft.Registers:
        return cft.Registers.build(psi = 2*self.N)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:

        i_qubs = quregs["i"]
        j_qubs = quregs["j"]
        psi_qubs = quregs["psi"]
        control = quregs["control"]
        spin = quregs["spin"]

        yield cft.SelectedMajoranaFermionGate.make_on(target_gate=cirq.Y, selection=i_qubs, control=control, target=psi_qubs)
        yield cirq.S.on(*control)
        yield cft.SelectedMajoranaFermionGate.make_on(target_gate=cirq.X, selection=j_qubs, control=control, target=psi_qubs)

class MTD_Select(cft.SelectOracle):
    """
    MTD select circuit
    """
    def __init__(self, N : int):
        self.N = N

    @property
    def control_registers(self) -> cft.Registers:
        return cft.Registers.build(control=1)

    @property
    def selection_registers(self) -> cft.SelectionRegisters:
        i_reg = spacial_orbital_sel_register(self.N, "i")
        j_reg = spacial_orbital_sel_register(self.N, "j")
        k_reg = spacial_orbital_sel_register(self.N, "k")
        l_reg = spacial_orbital_sel_register(self.N, "l")
        v_reg = cft.SelectionRegister("v", 1, 2)
        return merge_registers(i_reg, j_reg, k_reg, l_reg, v_reg)

    @property
    def target_registers(self) -> cft.Registers:
        return cft.Registers.build(psi = 2*self.N)

    @property
    def extra_registers(self) -> cft.Registers:
        s1_reg = spin_sel_register("s1")
        s2_reg = spin_sel_register("s2")
        v_and_ctl = cft.Register("v_and_ctl", 1)

        return merge_registers(s1_reg, s2_reg, v_and_ctl)

    @property
    def registers(self) -> cft.Registers:
        return merge_registers(self.control_registers, self.selection_registers, self.target_registers, self.extra_registers)

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
        psi_qubs = quregs["psi"]
        control = quregs["control"]
        s1 = quregs["s1"]
        s2 = quregs["s2"]
        v_and_ctl = quregs["v_and_ctl"]

        yield cirq.H.on(*s1)
        yield cirq.H.on(*s2)
        Qijs1 = Select_Qija(self.N)

        yield Qijs1.on_registers(i=i_qubs, j=j_qubs, psi=psi_qubs,
            control=control, spin=s1)

        yield cft.And((1,1)).on_registers(control=[*control, *v_qubs], ancilla=[], target=v_and_ctl)

        yield Qijs1.on_registers(i=k_qubs, j=l_qubs, psi=psi_qubs,
            control=v_and_ctl, spin=s2)


        


