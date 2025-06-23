import cirq
import cirq_ft as cft
import sys
import os
my_dir = os.path.dirname(os.path.realpath(__file__))

sys.path.append(my_dir + '/cirq/')
from utils import *
from sparse import *
from givens import *
from pauli_utils import *
from df import *
from ac import *

from mtd_prepare import *
from mtd_select import *

def superposition_t_complexity_(self):
    #fix t_complexity of PrepareUniformSuperposition gate
    """Prepares a uniform superposition over first $n$ basis states using $O(log(n))$ T-gates.

    Performs a single round of amplitude amplification and prepares a uniform superposition over
    the first $n$ basis states $|0>, |1>, ..., |n - 1>$. The expected T-complexity should be
    $10 * log(L) + 2 * K$ T-gates and $2$ single qubit rotation gates, where $n = L * 2^K$.
    """
    n, k = self.n, 0
    while n > 1 and n % 2 == 0:
        k += 1
        n = n // 2
    L = self.n / (2**k)
    logL = ceil(log2(L))
    
    return cft.TComplexity(t=10*logL + 2*k, rotations=2)

#cft.algos.prepare_uniform_superposition.PrepareUniformSuperposition._t_complexity_ = superposition_t_complexity_