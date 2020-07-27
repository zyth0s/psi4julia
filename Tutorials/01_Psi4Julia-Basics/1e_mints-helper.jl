# ---
# jupyter:
#   jupytext:
#     formats: jl:light,ipynb
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Julia 1.4.2
#     language: julia
#     name: julia-1.4
# ---

# # MintsHelper: Generating 1- and 2-electron Integrals with <span style='font-variant: small-caps'> Psi4 </span>
#
# In all of quantum chemistry, one process which is common to nearly every method is the evaluation of one-
# and two-electron integrals.  Fortunately, we can leverage infrastructure in <span style='font-variant: small-caps'> 
# Psi4 </span> to perform this task for us.  This tutorial will discuss the [``psi4.core.MintsHelper``](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.MintsHelper "Go to API") class, which is an
# interface for the powerful Psi4 ``libmints`` library which wraps the `libint` library, where these integrals are actually computed.  
#
# ## MintsHelper Overview
# In order to compute 1- and 2-electron integrals, we first need a molecule and basis set with which to work.  So, 
# before diving into `MintsHelper`, we need to build these objects.  In the cell below, we have imported
# <span style='font-variant: small-caps'> Psi4 </span> and NumPy, defined a water molecule, and set the basis to
# cc-pVDZ.  We've also set the memory available to <span style='font-variant: small-caps'> Psi4</span>, as well as
# defined a variable `numpy_memory` which we will discuss later.

# +
# ==> Setup <==
# Import statements
using PyCall: pyimport
psi4 = pyimport("psi4")
np   = pyimport("numpy")
using Formatting: printfmt

# Memory & Output file
psi4.set_memory(Int(2e9))
numpy_memory = 2
psi4.core.set_output_file("output.dat", false)

# Molecule definition
h2o = psi4.geometry("""
O
H 1 0.96
H 1 0.96 2 104.5
""")

# Basis Set
psi4.set_options(Dict("basis" => "cc-pvdz"))
# -

# Now, we are ready to create an instance of the `MintsHelper` class.  To do this, we need to pass a `BasisSet`
# object to the `MintsHelper` initializer.  Fortunately, from the previous tutorial on the `Wavefunction` class, we know
# that we can obtain such an object from an existing wavefunction.  So, let's build a new wavefunction for our molecule,
# get the basis set object, and build an instance of `MintsHelper`:

# +
# ==> Build MintsHelper Instance <==
# Build new wavefunction
wfn = psi4.core.Wavefunction.build(h2o, psi4.core.get_global_option("basis"))

# Initialize MintsHelper with wavefunction's basis set
mints = psi4.core.MintsHelper(wfn.basisset())
# -

# Below are summarized several commonly computed quantities and how to obtain them using a `MintsHelper` class method:
#
# | Quantity | Function | Description |
# |----------|----------|-------------|
# | AO Overlap integrals | [mints.ao_overlap()](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.MintsHelper.ao_overlap "Go to Documentation") | Returns AO overlap matrix as a `psi4.core.Matrix` object |
# | AO Kinetic Energy | [mints.ao_kinetic()](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.MintsHelper.ao_kinetic "Go to Documentation") | Returns AO kinetic energy matrix as a `psi4.core.Matrix` object |
# | AO Potential Energy | [mints.ao_potential()](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.MintsHelper.ao_potential "Go to Documentation") | Returns AO potential energy matrix as a `psi4.core.Matrix` object |
# | AO Electron Repulsion Integrals | [mints.ao_eri()](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.MintsHelper.ao_eri "Go to Documentation") | Returns AO electron repulsion integrals as a `psi4.core.Matrix` object 

# As discussed previously, any of these `psi4.core.Matrix` objects can be accessed as Julia arrays, which is the preferred 
# method in Psi4Julia.  For a Psi4 matrix `A`, we can access a Julia view using `psi4view(A)`, or we can make a
# copy of the matrix using `np.array(A)`.  This works as one would expect, converting square matrices into arrays of Array{Float64,2} type, for the overlap (S), kinetic energy (T), and potential energy (V) matrices.  In Psi4, the electron repulsion integrals 
# (ERIs) are handled somewhat differently; `mints.ao_eri()` returns the rank-4 ERI tensor packed into a 2D matrix.  If the 
# four indices of the ERI are p, q, r, s, then this element of the Psi4 Matrix can be accessed by first computing composite 
# indices `pq = p * nbf + q` and `rs = r * nbf + s`, and then accessing element `A.get(pq,rs)`.  However, for convenience, 
# the Julia view is a rank-4 tensor, and a particular ERI is more simply accessed like this:
# ~~~python
# I = mints.ao_eri()
# I = psi4view(I)
# val = I[p,q,r,s]
# ~~~

function psi4view(psi4matrix)
   # Assumes Float64 type, C ordering
   if !hasproperty(psi4matrix,:__array_interface__)
      throw(ArgumentError("Input matrix cannot be accessed. Try assigning to a variable first"))
   end
   array_interface = psi4matrix.__array_interface__
   array_interface["data"][2] == false   || @warn "Not writable"
   array_interface["strides"] == nothing || @warn "Different ordering than C"
   array_interface["typestr"] == "<f8"   || @warn "Not little-endian Float64 eltype"
   ptr   = array_interface["data"][1]
   shape = reverse(array_interface["shape"])
   ndims = length(shape)
   unsafe_wrap(Array{Float64,ndims}, Ptr{Float64}(ptr), shape)
end

# In addition to these methods, another which is worth mentioning is the `MintsHelper.mo_eri()` ([Go to documentation](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.MintsHelper.mo_eri)) function, which can transform 
# the four-index, two-electron repulsion integrals from the atomic orbital (AO) to the molecular orbital (MO) basis,
# which will be important in MP2 theory.  

# ## Memory Considerations

# Before moving forward to computing any 1- or 2-electron integrals, we must first discuss the memory requirements of
# these objects.  Whenever these quantities are computed, they are stored directly in memory (a.k.a. RAM,
# *not* on the hard drive) which, for a typical laptop or personal computer, usually tops out at around 16 GB of 
# space.  The storage space required by the two-index AO overlap integrals and four-index ERIs scales as ${\cal O}(N^2)$ 
# and ${\cal O}(N^4)$, respectively, where $N$ is the number of AO basis functions.  This means that for a
# system with 500 AO basis functions, while the AO overlap integrals will only require 1 MB of memory to store,
# the ERIs will require a staggering **500 GB** of memory!! This can be reduced to **62.5 GB** of memory if integral permutational symmetry is used. 
# However, this complicates the bookkeeping, and is not used in the `mints` functions discussed above.  For this reason, as well as the steep computational 
# scaling of many of the methods demonstrated here, we limit ourselves to small systems ($\sim50$ basis functions)
# which should not require such egregious amounts of memory.  Additionally, we will employ a "memory check" to catch
# any case which could potentially try to use more memory than is available:
# ~~~python
# # Memory check for ERI tensor
# I_size = nbf^4 * 8.e-9
# printfmt("Size of the ERI tensor will be {:4.2f} GB.\n", I_size)
# memory_footprint = I_size * 1.5
# if I_size > numpy_memory
#     psi4.core.clean()
#     throw(OutOfMemoryError("Estimated memory utilization ($memory_footprint GB) exceeds allotted memory " *
#                            "limit of $numpy_memory GB."))
# ~~~
# In this example, we have somewhat arbitrarily assumed that whatever other matrices we may need, in total their memory
# requirement will not exceed 50% of the size of the ERIs (hence, the total memory footprint of `I_size * 1.5`)
# Using the `numpy_memory` variable, we are able to control whether the ERIs will be computed, based on the amount of
# memory required to store them. 
#
# <font color="red">**NOTE: DO NOT EXCEED YOUR SYSTEM'S MEMORY.  THIS MAY RESULT IN YOUR PROGRAM AND/OR COMPUTER CRASHING!**</font>

# ## Examples: AO Overlap, AO ERIs, Core Hamiltonian
# The cell below demonstrates obtaining the AO overlap integrals, conducting the
# above memory check, and computing the ERIs and core Hamiltonian matrix for our water molecule.

# +
# ==> Integrals galore! <==
# AO Overlap
S = mints.ao_overlap()
S = psi4view(S) # psi4view(mints.ao_overlap()) would fail, first assign to a variable

# Number of basis functions
nbf = size(S)[1]

# Memory check
I_size = nbf^4 * 8.e-9
printfmt("Size of the ERI tensor will be {:4.2f} GB.\n", I_size)
memory_footprint = I_size * 1.5
if I_size > numpy_memory
    psi4.core.clean()
    throw(OutOfMemoryError("Estimated memory utilization ($memory_footprint GB) exceeds allotted memory " *
                           "limit of $numpy_memory GB."))
end

# Compute AO-basis ERIs
I = mints.ao_eri()

# Compute AO Core Hamiltonian
T = mints.ao_kinetic()
V = mints.ao_potential()
T = psi4view(T)
V = psi4view(V)
H = T + V
