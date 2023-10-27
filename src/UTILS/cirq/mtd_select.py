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

class controlled_Qij(cft.SelectOracle):
    """
    controlled by |ctl> application of 
    |k>|l>|psi_up>-> |k>|l> i γ_{k,0} γ_{l,1} |psi_up>|
    |psi_up>: spin-orbital register with only spin-up (i.e. equivalent to spacial orbital register)
    input: N = number of spacial orbitals
    """
    N : int

    @property
    def control_registers(self) -> cft.Registers:
        return cft.Registers.build(control=1)

    @property
    def selection_registers(self) -> cft.SelectionRegisters:
        i_reg = spacial_orbital_sel_register(N, "i")
        j_reg = spacial_orbital_sel_register(N, "j")

        return merge_registers(i_reg, j_reg)

    @property
    def target_registers(self) -> cft.Registers:
        return cft.Registers.build(psi_up = self.N)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:

        i_qubs = quregs["i"]
        j_qubs = quregs["j"]
        psi_qubs = quregs["psi_up"]
        control = quregs["control"]

        yield SelectedMajoranaFermionGate.make_on(target_gate=cirq.Y, selection=i_qubs, control=control, target=psi_qubs)
        yield cirq.S.on(*control)
        yield SelectedMajoranaFermionGate.make_on(target_gate=cirq.X, selection=j_qubs, control=control, target=psi_qubs)

class controlled_swap_Qija(cft.SelectOracle):
    """
    controlled application of Qija by using controlled_Qij and swapping spin registers
    """
    N : int

    @property
    def control_registers(self) -> cft.Registers:
        return cft.Registers.build(control=1)

    @property
    def selection_registers(self) -> cft.SelectionRegisters:
        i_reg = spacial_orbital_sel_register(N, "i")
        j_reg = spacial_orbital_sel_register(N, "j")
        spin_reg = spin_sel_register("spin")
        return merge_registers(i_reg, j_reg, spin_reg)

    @property
    def target_registers(self) -> cft.Registers:
        return cft.Registers.build(psi_up = self.N, psi_down = self.N)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:

        i_qubs = quregs["i"]
        j_qubs = quregs["j"]
        psi_up_qubs = quregs["psi_up"]
        psi_down_qubs = quregs["psi_down"]
        control = quregs["control"]
        spin = quregs["spin_reg"]

        yield swap_network.MultiTargetCSwap.make_on(control=spin, target_x=psi_up_qubs, target_y=psi_down_qubs)

        Qij = controlled_Qij(N = self.N)
        yield Qij.on_registers(i=i_qubs, j=j_qubs, psi_qubs=psi_up_qubs, control=control)

        yield swap_network.MultiTargetCSwap.make_on(control=spin, target_x=psi_up_qubs, target_y=psi_down_qubs)

class MTD_Select(cft.SelectOracle):
    """
    MTD select circuit
    """
    N : int
    @property
    def control_registers(self) -> cft.Registers:
        return cft.Registers.build(control=1)

    @property
    def selection_registers(self) -> cft.SelectionRegisters:
        i_reg = spacial_orbital_sel_register(N, "i")
        j_reg = spacial_orbital_sel_register(N, "j")
        k_reg = spacial_orbital_sel_register(N, "k")
        l_reg = spacial_orbital_sel_register(N, "l")
        v_reg = cft.SelectionRegister("v", 1, 2)
        return merge_registers(i_reg, j_reg, k_reg, l_reg, v_reg)

    @property
    def target_registers(self) -> cft.Registers:
        return cft.Registers.build(psi_up = self.N, psi_down = self.N)

    @property
    def extra_registers(self) -> cft.Registers:
        s1_reg = spin_sel_register("s1")
        s2_reg = spin_sel_register("s2")
        v_and_ctl = cft.Register("v_and_ctl", 1)

        return merge_registers(s1_reg, s2_reg, v_and_ctl)

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
        psi_up_qubs = quregs["psi_up"]
        psi_down_qubs = quregs["psi_down"]
        control = quregs["control"]
        s1 = quregs["s1"]
        s2 = quregs["s2"]
        v_and_ctl = quregs["v_and_ctl"]

        yield cirq.H.on(*s1)
        yield cirq.H.on(*s2)
        Qijs1 = controlled_swap_Qija(N = self.N)

        yield Qijs1.on_registers(i=i_qubs, j=j_qubs, psi_up=psi_up_qubs, psi_down=psi_down_qubs,
            control=control, spin=s1)

        yield cft.And((1,1)).on_registers(control=[*control, v_qubs], ancilla=[], target=v_and_ctl)

        yield Qkls2.on_registers(i=k_qubs, j=l_qubs, psi_up=psi_up_qubs, psi_down=psi_down_qubs,
            control=v_and_ctl, spin=s2)


        


