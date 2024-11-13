using Pkg
Pkg.add("Arpack")
Pkg.activate(".")
Pkg.instantiate()
Pkg.add("PythonCall")
using PythonCall
pyscf = pyimport("pyscf")
scipy = pyimport("scipy")
of = pyimport("openfermion")
np = pyimport("numpy")
juliacall = pyimport("juliacall")
UTILS_DIR = @__DIR__
sys = pyimport("sys")
sys.path.append(UTILS_DIR)
fermionic = pyimport("ferm_utils")

function bliss_test(F1::F_OP, F2::F_OP, n_elec)
  h1e = h1e_to_OF(F1)
  h2e = h2e_to_OF(F1)
  h3e = h3e_to_OF(F1)

  # Construct the Hamiltonian
  hamil = h1e + h2e + h3e
  sparse_op = of.get_sparse_operator(hamil)
  energy, vect = of.jw_get_ground_state_at_particle_number(sparse_op, n_elec)

  sparse_op_max = of.get_sparse_operator(-hamil)
  max_energy, vect = of.jw_get_ground_state_at_particle_number(sparse_op_max, n_elec)

  h1e_bliss = h1e_to_OF(F2)
  h2e_bliss = h2e_to_OF(F2)
  h3e_bliss = h3e_to_OF(F2)

  hamil_bliss = h1e_bliss + h2e_bliss + h3e_bliss
  sparse_op_bliss = of.get_sparse_operator(hamil_bliss)
  energy_bliss, vect_bliss = of.jw_get_ground_state_at_particle_number(sparse_op_bliss, n_elec)

  sparse_op_max_bliss = of.get_sparse_operator(-hamil_bliss)
  max_energy_bliss, vect_bliss = of.jw_get_ground_state_at_particle_number(sparse_op_max_bliss, n_elec)

  println("Original Min Energy: ", energy)
  println("Original Core energy: ", F1.mbts[1][1])

  println("Bliss Min Energy: ", energy_bliss)
  println("Bliss Core energy: ", F2.mbts[1][1])

  println("Min energy difference:", energy + F1.mbts[1][1] - energy_bliss - F2.mbts[1][1])


  println("|Subspace Min Energy original - Energy after Bliss| < 1E-4 ", abs(energy + F1.mbts[1][1] - energy_bliss - F2.mbts[1][1]) < 1E-4)

  println("Spectral range difference: ", -max_energy - energy + max_energy_bliss + energy_bliss)
  println("|Spectral Range difference| < 1E-4 ", abs(-max_energy - energy + max_energy_bliss + energy_bliss) < 1E-4)

end


function h1e_to_OF(F::F_OP)
  h1e = F.mbts[2]
  TOT_OP = of.FermionOperator.zero()
  TOT_OP += fermionic.get_ferm_op_one(h1e, spin_orb=false)

  return TOT_OP
end

function h2e_to_OF(F::F_OP)
  h2e = F.mbts[3]
  TOT_OP = of.FermionOperator.zero()
  TOT_OP += fermionic.get_ferm_op_two(h2e, spin_orb=false)

  return TOT_OP
end

function h3e_to_OF(F::F_OP)
  h3e = F.mbts[4]
  TOT_OP = of.FermionOperator.zero()
  TOT_OP += fermionic.get_ferm_op_three(h3e)

  return TOT_OP
end