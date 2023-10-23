from givens import *
from utils import *
from math import ceil, log2

class Superposition_mn(cft.GateWithRegisters):
    def __init__(self, M : int, br : int):
        num_qubs = ceil(log2(M+1))
        self.mu_reg = cft.Register("mu", num_qubs)
        self.nu_reg = cft.Register("nu", num_qubs)
        self.br_reg = cft.Register("br", br)
        self.ancillas_reg = cft.Register("superposition_ancilla", 5)
        self.succes_reg = cft.Register("success", 1)

    def registers(self) -> cft.Registers:
        return merge_registers(self.mu_reg, self.nu_reg, self.br_reg, self.ancillas_reg, self.succes_reg)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:

        mu_qubs = get_qubits(self.mu_reg)
        nu_qubs = get_qubits(self.nu_reg)
        br_qubs = get_qubits(self.br_reg)
        ancillas_qubs = get_qubits(self.ancillas_reg)
        success_qubs = get_qubits(self.success_reg)

        yield cirq.H.on_each(*mu_qubs)
        yield cirq.H.on_each(*nu_qubs)
        