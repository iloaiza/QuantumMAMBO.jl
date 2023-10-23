import cirq
import cirq_ft as cft
import sys
import os
my_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(my_dir + '/cirq/')

from utils import *
from sparse import *
from givens import *
from df import *
from ac import *


def controlled_majorana_vec(num_spacial_orbs, orb_name = "i", spin_name = "σ", is_x = True,
    ctl_reg = cft.Registers([cft.Register("control",1)])):
    ORBS_REG = cft.SelectionRegister(orb_name, num_spacial_orbs, num_spacial_orbs+1)
    SPINS_REG = cft.SelectionRegister(spin_name, 1, 1)
    SEL_REGS = cft.SelectionRegisters([ORBS_REG, SPINS_REG])

    if is_x:
        which_gate = cirq.X
    else:
        which_gate = cirq.Y

    gate = cft.SelectedMajoranaFermionGate(SEL_REGS,ctl_reg,target_gate=which_gate)
    circuit = cirq.Circuit(gate.on_registers(**gate.registers.get_named_qubits()))
    qubits = list(q for v in gate.registers.get_named_qubits().values() for q in v)

    return circuit, gate, qubits


def Qija(num_spacial_orbs):
    c1, ia_ctl_gate, ia_ctl_qub = controlled_majorana_vec(num_spacial_orbs, "i", "σ", False)
    CTL_REG = ia_ctl_gate.control_regs
    c2, ja_ctl_gate, ja_ctl_qub = controlled_majorana_vec(num_spacial_orbs, "j", "σ", True, CTL_REG)
    
    S_gate = cft.GateWithRegisters(cirq.S)

    s_cir = cirq.Circuit(S_gate.on(**CTL_REG.get_named_qubits()))

    cir = c1 + s_cir + c2

    return cir