# Functions for working with the chemical benchmark data

"""
load_chem_benchmark_hamiltonian(datafile::String)    
    Load one body tensor (aka one electron integrals), 
    two body tensor (aka two electron integrals), 
    and attributes from an hdf5 file formatted 
    according to the benchmark standard.
"""
function load_chem_benchmark_hamiltonian(datafile::String)
    fid = h5open(datafile, "r")
    one_body_tensor = read(fid, "one_body_tensor")
    two_body_tensor = read(fid, "two_body_tensor")
    # attributes = read(fid, "attributes")

    return one_body_tensor, two_body_tensor#, attributes
end