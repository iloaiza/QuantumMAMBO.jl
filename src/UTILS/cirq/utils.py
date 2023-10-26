import attr
import cirq
import cirq_ft as cft
import numpy as np
from numpy.typing import NDArray
from math import ceil, log2
from typing import Sequence

def to_operation(G):
    #turns a gate G into an operation by targeting registers on G
    return G.on_registers(**G.registers.get_named_qubits())

def to_circuit(G):
    #recursively turns a gate G, or list of gates, into a circuit, will use default G.registers for target qubits
    if type(G) == list:
        g_sum = to_circuit(G[0])
        for i in range(1,len(G)):
            g_sum += to_circuit(G[i])
        return g_sum
    elif issubclass(type(G),cirq.Operation):
        return cirq.Circuit(G)
    
    return cirq.Circuit(to_operation(G))

def spacial_orbital_sel_register(num_spacial_orbs, orb_name="i"):
    return cft.SelectionRegister(orb_name, ceil(log2(num_spacial_orbs)), num_spacial_orbs)

def spin_sel_register(spin_name="σ"):
    return cft.SelectionRegister(spin_name, 1, 2)

def get_names(**kwargs):
    #iterator to get all the keys in a kwargs dictionary
    keys = []
    for key, _ in kwargs.items():
        keys += [key]
    
    return keys

def merge_registers(*regs):
    reg_list = []
    is_sel = []
    
    for reg in regs:
        if issubclass(type(reg), cft.Registers):
            for sub_reg in reg:
                reg_list += [sub_reg]
                is_sel += [issubclass(type(sub_reg), cft.SelectionRegisters)]
        elif issubclass(type(reg), cft.Register):
            reg_list += [reg]
            is_sel += [issubclass(type(reg), cft.SelectionRegister)]
        else:
            print("Trying to merge registers but input has non-register object!")

    is_sel = bool(np.prod(is_sel))
    if is_sel:
        return cft.SelectionRegisters(reg_list)
    else:
        return cft.Registers(reg_list)       

def get_regs_names(*regs):
    REGS = merge_registers(*regs)
    
    names_list = []

    for reg in REGS:
        names_list += reg.name
        
    return names_list

def get_qubits(*regs):
    REGS = merge_registers(*regs)
    
    qub_list = REGS.get_named_qubits()
    qub_names = get_names(**qub_list)
    
    qub_arr = []
    for name in qub_names:
        qub_arr = np.append(qub_arr,qub_list[name])
        
    return qub_arr

def recursive_circuit(GWR) -> cirq.Circuit:
    """Diagram info that uses underlying operators from decompose_from_registers to represent circuit
    Takes GateWithRegisters as input
    """
    context = cirq.DecompositionContext(cirq.ops.SimpleQubitManager())

    op_list = []
    for op in GWR.decompose_from_registers(**GWR.registers.get_named_qubits(), context=context):
        op_list += [op]
    
    return to_circuit(op_list)