from typing import Tuple
import numpy as np
from numpy.lib.scimath import arccos, arcsin  # has analytc continuation to cplx
from openfermion.resource_estimates.utils import QR, QI, power_two
#from openfermion.resource_estimates.df import compute_cost as cc
import openfermion.resource_estimates.df.compute_cost_df as df

def compute_cost(n, lam, L, Lxi, chi, beta, stps, verbose=False):
    toffoli_cost, _, ancilla_cost = df.compute_cost(n, lam, 1, L, Lxi, chi, beta, stps, verbose)

    return 4*toffoli_cost, ancilla_cost
