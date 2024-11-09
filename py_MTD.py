from src.UTILS.ham_utils import *

import openfermion as of
from openfermion.resource_estimates import df, thc
import math

mol_name = "h2"
cutoff = 1e-5

if mol_name == "h2":
    xyz = ["H 0.0 0.0 0.0", "H 0.0 0.0 1.0"]
elif mol_name == "lih":
    xyz = ["H 0.0 0.0 0.0", "Li 0.0 0.0 1.0"]
elif mol_name == "beh2":
    xyz = ["H 0.0 0.0 0.0", "Be 0.0 0.0 1.0", "H 0.0 0.0 2.0"]
elif mol_name == "h2o":
    angle = 0.9389871375729493 #107.6/2 in radians
    xDistance = math.sin(angle)
    yDistance = math.cos(angle)
    xyz = ["O 0.0 0.0 0.0", "H -{}, {}, 0.0", "H {}, {}, 0.0".format(xDistance, yDistance, xDistance, yDistance)]
elif mol_name == "nh3":
    bondAngle = 1.8675022996339325 #107 in radians
    cosval = math.cos(bondAngle)
    sinval = math.sin(bondAngle)
    thirdyRatio = (cosval - cosval**2) / sinval
    thirdxRatio = math.sqrt(1 - cosval**2 - thirdyRatio**2)
    xyz = ["H 0.0 0.0 1.0", "H 0.0 {} {}", "H {} {} {}", "N 0.0 0.0 0.0".format(sinval, cosval, thirdxRatio, thirdyRatio, cosval)]


h_const, obt_hf, tbt_hf, obt_fb, tbt_fb, η, mf = localized_ham_from_xyz(xyz, return_mf = True)

print("Starting DF routine...")
factorized_eris, df_factors, _, _ = df.factorize(tbt_hf, cutoff)
df_lambda  = df.compute_lambda(mf, df_factors)
_, number_toffolis, num_logical_qubits = df.compute_cost(np.shape(obt_hf)[0] * 2, df_lambda)

print("Number of T-gates = {}".format(7*number_toffolis))
print("Numfer of logical qubits = {}".format(num_logical_qubits))