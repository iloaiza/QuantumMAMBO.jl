from givens import *
from utils import *
from pauli_utils import *
from sparse import *
import cirq
import cirq_ft as cft
import numpy as np
from math import log2, ceil, pi
from coeff_prep import *


def find_rotation_angles(coeffs, tiny = 1e-8):
    # for a given set of anticommuting operators {g_i}_{i=1}^{N} and a corresponding unitary
    # A = sum_i coeffs[i] * g_i
    # This function finds the rotation angles such that:
    # A = prod_{i=N-1}^{1} exp(theta_i g_i*g_{i+1}) * g_1 * prod_{i=1}^{N-1} exp(theta_i g_i*g_{i+1})
    N = len(coeffs)
    norm_coeffs = coeffs / np.sqrt(np.sum(np.abs(coeffs) ** 2))
    thetas = 0.5 * pi * np.ones(N-1)
    thetas[0] = np.arccos(norm_coeffs[0]) / 2
    for i in range(1,N-1):
        cum_coef = np.sum(np.abs(norm_coeffs[0:i] ** 2))

        if np.abs(1 - cum_coef) < tiny:
            break

        if np.abs(norm_coeffs[i]) > tiny:
            ccur = norm_coeffs[i]/np.prod(np.sin(2*thetas[0:i]))
        else:
            ccur = 0
        thetas[i] = np.arccos(ccur)/2

    if np.abs(norm_coeffs[-1]) > tiny:
        thetas[-1] *= np.sign(norm_coeffs[-1])

    return thetas

def bit_product(bit_1, bit_2):
    """
    product of two bits representing pauli matrices
    input: integers for pauli matrix (e.g. 1,2 -> X,Y)

    output: resulting pauli matrix as integer, along phase (e.g. Z, i)
    """
    if bit_1 == bit_2:
        return 0, 1
    elif bit_1 == 0:
        return bit_2, 1
    elif bit_2 == 0:
        return bit_1, 1
    else:
        if bit_1 == 1 and bit_2 == 2:
            return 3, 1j
        elif bit_1 == 2 and bit_2 == 1:
            return 3, -1j
        elif bit_1 == 2 and bit_2 == 3:
            return 1, 1j
        elif bit_1 == 3 and bit_2 == 2:
            return 1, -1j
        elif bit_1 == 3 and bit_2 == 1:
            return 2, 1j
        elif bit_1 == 1 and bit_2 == 3:
            return 2, -1j


def pauli_prod(pauli_1, pauli_2):
    """
    input: pauli words as integer vectors (e.g. X1*Y2*Z4 = [1,2,0,3])

    output: product of pauli words as integer vector along phase value
    """
    assert len(pauli_1) == len(pauli_2), f"Pauli vectors should have same length for calculating product"

    phase = 1.0 + 0j
    prod_bits = np.zeros(len(pauli_1), dtype=int)
    for i in range(len(pauli_1)):
        prod_bits[i], prod_phase = bit_product(pauli_1[i], pauli_2[i])
        phase *= prod_phase

    return prod_bits, phase



class Select_AC(cft.UnaryIterationGate, cft.SelectOracle):
    """
    Implement select oracle for anticommuting grouping
    Corresponds to H = sum_{n=1}^Nac an An, where An is group of anticommuting Paulis
    For Pauli decomposition H = sum_k ck Pk, each group n holds a subset of {ck}'s
    An = (1/an) * sum_{k in group(n)} ck Pk
    an = sqrt(sum_{k in group(n)} ck^2)

    input: ac_groups: list of length Nac, where the n-th element corresponds to:
        ac_vecs[n][i,j] contains integer i of Pauli word number j of group n
        ac_coeffs[n][j] contains coefficient of Pauli word number j of group n
        (for integer representation of Pauli words, see MultiplexedPauliExponential in givens.py) 
    """
    def __init__(self, ac_vecs, ac_coeffs, n_reg = None, targ_reg = None, ctl_reg = None):
        self.Nac = len(ac_coeffs)
        Nac = self.Nac

        ac_groups = []
        for n in range(Nac):
            ac_groups += [(ac_coeffs[n], ac_vecs[n])]
        self.ac_groups = ac_groups

        num_paulis = np.zeros(self.Nac, dtype=int)
        pauli_len = np.zeros(self.Nac, dtype=int)
        for n in range(self.Nac):
            paulis = ac_groups[n][1]
            pauli_len[n], num_paulis[n] = np.shape(paulis)

        self.num_paulis = num_paulis

        num_targs = pauli_len[0]
        assert (pauli_len - num_targs == 0).any(), f"All Pauli words should have the same length!"            

        if type(n_reg) == type(None):
            self.n_register = cft.SelectionRegister("n", ceil(log2(Nac)), Nac)
        else:
            assert n_reg.iteration_length == Nac
            self.n_register = n_reg

        if type(targ_reg) == type(None):
            self.target_register = cft.Register("target", (num_targs,))
        else:
            assert targ_reg.total_bits() == num_targs
            self.target_register = targ_reg

        if type(ctl_reg) == type(None):
            self.control_register = cft.Register("control", 1)
        else:
            assert ctl_reg.total_bits() == 1
            self.control_register = ctl_reg
            

    @cached_property
    def control_registers(self) -> cft.Registers:
        return merge_registers(self.control_register)

    @cached_property
    def selection_registers(self) -> cft.SelectionRegisters:
        return merge_registers(self.n_register)

    @cached_property
    def target_registers(self) -> cft.Registers:
        return merge_registers(self.target_register)

    def nth_operation(
        self, context: cirq.DecompositionContext, control: cirq.Qid, **kwargs
    ) -> cirq.OP_TREE:
        """Apply nth operation on the target registers when selection registers store `n`.

        The `UnaryIterationGate` class is a mixin that represents a coherent for-loop over
        different indices (i.e. selection registers). This method denotes the "body" of the
        for-loop, which is executed `self.selection_registers.total_iteration_size` times and each
        iteration represents a unique combination of values stored in selection registers. For each
        call, the method should return the operations that should be applied to the target
        registers, given the values stored in selection registers.

        The derived classes should specify the following arguments as `**kwargs`:
            1) `control: cirq.Qid`: A qubit which can be used as a control to selectively
            apply operations when selection register stores specific value.
            2) Register names in `self.selection_registers`: Each argument corresponds to
            a selection register and represents the integer value stored in the register.
            3) Register names in `self.target_registers`: Each argument corresponds to a target
            register and represents the sequence of qubits that represent the target register.
            4) Register names in `self.extra_regs`: Each argument corresponds to an extra
            register and represents the sequence of qubits that represent the extra register.
        """
        n = kwargs["n"]
        my_coeffs = self.ac_groups[n][0]
        my_paulis = self.ac_groups[n][1]
        pauli_len, num_paulis = np.shape(my_paulis)
        null_qubs = np.array([], dtype=object)
        target_qubs = get_qubits(self.target_registers)

        if n == 0:
            print("Phases from Pauli multiplication not being considered, angles might need correction")

        if num_paulis > 1:
            #calculate products of adjacent pauli terms
            paulis_i_iplus = np.zeros((pauli_len, num_paulis - 1), dtype = int)
            phases_i_iplus = np.zeros(num_paulis - 1, dtype=complex)
            for i in range(num_paulis - 1):
                paulis_i_iplus[:, i], phases_i_iplus[i] = pauli_prod(my_paulis[:, i], my_paulis[:, i+1])
            
            #calculate angles for decomposing ac unitary A as:
            #U = prod_{i ascending}(exp(theta_i * p_i * p_(i+1))), with p_i i-th pauli word
            #A = U^{dagger} * p_1 * U
            my_thetas = find_rotation_angles(my_coeffs)# * phases_i_iplus

            #implement AC unitary. First U is applied
            target_qubs = get_qubits(self.target_register)
            for i in reversed(range(num_paulis - 1)):
                yield PauliExponential.on_qubits(target_qubs, null_qubs, paulis_i_iplus[:, i], my_thetas[i])

            #controlled application of p_1
            yield PauliExponential.on_qubits(target_qubs, control, my_paulis[:,0], -pi)

            #application of U^dagger
            for i in range(num_paulis - 1):
                yield PauliExponential.on_qubits(target_qubs, null_qubs, paulis_i_iplus[:, i], my_thetas[i])
        else:
            yield PauliExponential.on_qubits(target_qubs, control, my_paulis[:,0], -pi)



class Prepare_AC(cft.PrepareOracle):
    """
    Prepare circuit for anticommuting grouping

    inputs: 
        -ac_coeffs: same as for Select_AC oracle
        -n_reg: register over which coherent coefficients state will be prepared
        -probability_epsilon: accuracy with which coefficients will be prepared

    """
    def __init__(self, ac_coeffs, n_reg = None, probability_epsilon: float = 1.0e-5):
        self.Nac = len(ac_coeffs)
        Nac = self.Nac
        self.ac_coeffs = ac_coeffs

        an_arr = np.zeros(self.Nac)
        for n in range(self.Nac):
            coeffs = ac_coeffs[n]
            an_arr[n] = np.sqrt(np.sum(np.abs(coeffs) ** 2))
        self.an_arr = an_arr

        if type(n_reg) == type(None):
            self.n_register = cft.SelectionRegister("n", ceil(log2(Nac)), Nac)
        else:
            assert n_reg.iteration_length == Nac
            self.n_register = n_reg
        self.probability_epsilon = probability_epsilon

    @property
    def selection_registers(self) -> cft.SelectionRegisters:
        return merge_registers(self.n_register)

    @property
    def registers(self) -> cft.SelectionRegisters:
        return self.selection_registers

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:

        prep = TargetedCoefficients.from_lcu_probs(self.an_arr, probability_epsilon = self.probability_epsilon, sel_reg = self.n_register)
        yield prep.on(*get_qubits(prep.registers))
