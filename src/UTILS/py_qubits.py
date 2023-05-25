from openfermion import QubitOperator
import numpy as np

#Pauli to symplectic vector and back functions
def pauli_word_to_binary_vector(pauli_word, n_qubits):
    if isinstance(pauli_word, QubitOperator):
        pauli_word_tuple = list(pauli_word.terms.keys())[0]
    else:
        pauli_word_tuple = pauli_word
    binary_vec = np.zeros(2*n_qubits, dtype='int')
    for sigma in pauli_word_tuple:
        if sigma[1] == 'X':
            binary_vec[sigma[0]] = 1
        elif sigma[1] == 'Z':
            binary_vec[n_qubits+sigma[0]]= 1
        elif sigma[1] == 'Y':
            binary_vec[sigma[0]] = 1
            binary_vec[n_qubits+sigma[0]]= 1
    return binary_vec

def binary_vector_to_pauli_word(binary_vec):
    qubit_op_str = ''
    n_qubits = len(binary_vec) // 2
    for i in range(n_qubits):
        if binary_vec[i] == 1 and binary_vec[n_qubits+i] == 0:
            qubit_op_str += 'X{} '.format(i)
        elif binary_vec[i] == 1 and binary_vec[n_qubits+i] == 1:
            qubit_op_str += 'Y{} '.format(i)
        elif binary_vec[i] == 0 and binary_vec[n_qubits+i] == 1:
            qubit_op_str += 'Z{} '.format(i)
    return QubitOperator(qubit_op_str)

def is_anticommuting(string1, string2):
    #check if two binary vectors (i.e. Pauli words) are anticommuting
    n = len(string1) // 2

    x1 = string1[:n]
    z1 = string1[n:]
    x2 = string2[:n]
    z2 = string2[n:]

    return sum([x1[c] * z2[c] + z1[c] * x2[c] for c in range(n)]) % 2

def get_pauliword_list(H: QubitOperator, ignore_identity=True):
    """Obtain a list of pauli words in H. 
    """
    pws = []
    coefs = []
    for pw, val in H.terms.items():
        if ignore_identity:
            if len(pw) == 0:
                continue
        pws.append(QubitOperator(term=pw, coefficient=val))
        coefs.append(val)
    return pws, coefs

def pauli_l1(H: QubitOperator):
    pws, coefs = get_pauliword_list(H)
    return np.sum(np.abs(coefs))