# -*- coding: utf-8 -*-
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

# + [markdown] run_control={"frozen": false, "read_only": false}
# # Wavefunctions in <span style='font-variant: small-caps'> Psi4 </span>
#
# One very advantageous feature of <span style='font-variant: small-caps'> Psi4 </span> is the
# ability to generate, return, and manipulate wavefunctions both from computations and as independent entities.
# This is particularly useful because of the depth of information carried by a wavefunction -- which is formally
# an instance of the [`psi4.core.Wavefunction`](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.Wavefunction
# "Go to API") class.  This tutorial introduces these objects
# and provides an overview of their capabilities that will be leveraged in future tutorials.
#
# Let's begin our discussion by importing <span style='font-variant: small-caps'> Psi4 </span> and NumPy, and setting
# some basic options for <span style='font-variant: small-caps'> Psi4</span>, like the memory, to direct output to a file
# named `output.dat`, and options to be used when performing a computation.

# + run_control={"frozen": false, "read_only": false}
# ==> Basic Options <==
using PyCall: pyimport
psi4 = pyimport("psi4")
np   = pyimport("numpy")

# Memory & Output File
psi4.set_memory(Int(2e9))
psi4.core.set_output_file("output.dat", false)

# Computation options
psi4.set_options(Dict("basis" => "aug-cc-pvdz",
                      "scf_type" => "df",
                      "e_convergence" => 1e-8,
                      "d_convergence" => 1e-8))

# + [markdown] run_control={"frozen": false, "read_only": false}
# Now that we've set the basics, let's use what we learned in the Molecule tutorial to define a water molecule, in Z-matrix
# format, specifying that we want $C_1$ symmetry (instead of letting <span style='font-variant: small-caps'> Psi4
# </span> detect the real symmetry $C_{\text{2v}}$):

# + run_control={"frozen": false, "read_only": false}
# ==> Define C_1 Water Molecule <==
h2o = psi4.geometry("""
O
H 1 0.96
H 1 0.96 2 104.5
symmetry c1
""")

# + [markdown] run_control={"frozen": false, "read_only": false}
# Other than molecules, of course, the quintessential object within quantum chemistry (and arguably **the**
# quintessential object in all of quantum mechanics) is the _wavefunction_.  Every method in quantum mechanics seeks to
# find the wavefunction which describes the state of the system of interest.  So, how can we create these objects with
# <span style='font-variant: small-caps'> Psi4</span>?  If we simply wish to perform a computation (e.g., Hartree–Fock 
# or MP2), all we need to do is to define the molecule, call [``psi4.energy()``](http://psicode.org/psi4manual/master/api/psi4.driver.energy.html#psi4.driver.energy "Go to API"), and <span style='font-variant: small-
# caps'> Psi4 </span> will do the rest.  What about if we need a wavefunction _before_ performing a computation, or in
# order to implement a method?  Fortunately, the class method [`Wavefunction.build()`](http://psicode.org/psi4manual
# /master/psi4api.html#psi4.core.Wavefunction.build "Go to Documentation") allows us to build one from scratch, given
# a molecule and a basis set.  In the cell below, we've illustrated how to invoke this function:

# + run_control={"frozen": false, "read_only": false}
# ==> Build wavefunction for H2O from scratch with Wavefunction.build() <==
h2o_wfn = psi4.core.Wavefunction.build(h2o, psi4.core.get_global_option("basis"))

# + [markdown] run_control={"frozen": false, "read_only": false}
# Notice that we have passed the variable `h2o` (itself an instance of the `psi4.core.Molecule` class) and the AO basis
# set we wish to use to construct the wavefunction for this molecule.  We could have just as easily passed the string
# `'aug-cc-pvdz'` as an argument, but then we would have to remember to change the argument if we ever changed the
# <span style='font-variant: small-caps'> Psi4 </span> option in the `psi4.set_options()` block above. Generally, when 
# creating something like a wavefunction or a basis <span style='font-variant: small-caps'> Psi4</span>-side, the class 
# instances themselves are what is used to do so.  (Don't worry too much about creating basis sets yet, we'll cover
# these in more detail later.) 
#
# Now that we have built an instance of the `Wavefunction` class, we can access our wavefunction's information by 
# calling any of the member functions of the `Wavefunction` class on our object.  For instance, the number of spin-up
# ($\alpha$) electrons in our wavefuntion can be found using the [`Wavefunction.nalpha()`](http://psicode.org/psi4manual
# /master/psi4api.html#psi4.core.Wavefunction.nalpha "Go to Documentation") function:  
# ~~~python
# h2o_wfn.nalpha()
# ~~~
# Since the water molecule
# above was defined to be a neutral singlet, we expect that the total number of electrons in our wavefunction should
# be $\alpha + \beta = 2\alpha$.  Let's check:
# -

# Compute the number of electrons in water 
println("Water has $(Int(2h2o_wfn.nalpha())) electrons, according to our wavefunction.") 

# + [markdown] run_control={"frozen": false, "read_only": false}
# Good, <span style='font-variant: small-caps'> Psi4 </span> and every General Chemistry textbook on the planet agree.
# What other information can be gleaned from our wavefunction object? For now, not a whole lot.  Since we have built our wavefunction from our molecule and choice of basis set but haven't yet computed anything, the wavefunction doesn't have the orbitals, electron density, energy, or Psi variables attributes set.  Once a computation has been performed,
# however, all this information may be accessed.  This may be accomplised by _returning the wavefunction_ from a
# successful computation, a concept referred to in <span style='font-variant: small-caps'> Psi4</span>-lingo as 
# _wavefunction passing_.  To run a computation with <span style='font-variant: small-caps'> Psi4</span>, the function
# [`psi4.energy()`](http://psicode.org/psi4manual/master/api/psi4.driver.energy.html#psi4.driver.energy "Go to 
# Documentation") is invoked with a particular quantum chemical method, like `'scf'`, `'mp2'`, or `'ccsd(t)`'.  To
# return the wavefunction from that method, the additional argument `return_wfn=True` can be specified:
# ~~~python
# # Returning a CCSD(T) wavefunction
# energy, wfn = psi4.energy('ccsd(t)', return_wfn=True)
# ~~~
# Then, both the energy and the wavefunction are returned.  Give it a try yourself, for the Hartree–Fock computation
# in the cell below:

# + run_control={"frozen": false, "read_only": false}
# Get the SCF wavefunction & energies for H2O

scf_e, scf_wfn = psi4.energy("scf", return_wfn=true)
println("A float and a Wavefunction object returned: ", scf_e, "\n", scf_wfn)

# + [markdown] run_control={"frozen": false, "read_only": false}
# Now, we can access information you would expect a wavefunction to carry — basically everything we couldn't before.
# Below is summarized several quantities which will be used throughout the modules and tutorials to come.  All these  wavefunction attributes are available after a Hartree–Fock computation; make sure to try them out on our `scf_wfn`!
#
# | Quantity | Function(s) | Description |
# |----------|-------------|-------------|
# | Orbital Coefficients, **C** | [wfn.Ca()](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.Wavefunction.Ca "Go to Documentation"), [wfn.Cb()](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.Wavefunction.Cb "Go to Documentation") | Returns orbital coefficient matrix for $\alpha$ (Ca) or $\beta$ (Cb) orbitals. (Identical for restricted orbitals) |
# | Electron Density, **D** | [wfn.Da()](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.Wavefunction.Da "Go to Documentation"), [wfn.Db()](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.Wavefunction.Db "Go to Documentation") | One-particle density matrices for $\alpha$ (Da) and $\beta$ (Db) electrons. (Identical for restricted orbitals) |
# | Fock Matrix, **F** | [wfn.Fa()](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.Wavefunction.Fa "Go to Documentation"), [wfn.Fb()](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.Wavefunction.Fb "Go to Documentation") | Returns the Fock matrix. For wavefunction with unrestricted orbitals, distinct Fock matrices $F^{\alpha}$ and $F^{\beta}$ for $\alpha$ and $\beta$ orbitals, respectively, are created.|
# | Basis Set | [wfn.basisset()](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.Wavefunction.basisset "Go to Documentation") | Returns basis set associated with the wavefunction. |
# | $\alpha$ ($\beta$) electrons | [wfn.nalpha()](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.Wavefunction.nalpha "Go to Documentation"), [wfn.nbeta()](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.Wavefunction.nbeta "Go to Documentation") | Returns number of $\alpha$ ($\beta$) electrons of the wavefunction. |
# | Irreducible Representations | [wfn.nirrep()](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.Wavefunction.nirrep "Go to Documentation") | Returns number of irreducible representations (number of symmetry elements). Several objects can utilize molecular symmetry in the wavefunction. |
# | Occupied Orbitals | [wfn.doccpi()](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.Wavefunction.doccpi "Go to Documentation") | Returns number of doubly occupied orbitals per irrep in the wavefunction. |
# | Psi Variables | [wfn.variables()](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.Wavefunction.variables "Go to Documentation") | Returns all Psi variables associated with the method which computed the wavefunction. |
# | Energy   | [wfn.energy()](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.Wavefunction.energy "Go to Documentation") | Returns current energy of the wavefunction. |
# | Orbital Energies, $\boldsymbol{\epsilon}$ | [wfn.epsilon_a()](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.Wavefunction.epsilon_a "Go to Documentation"), [wfn.epsilon_b()](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.Wavefunction.epsilon_b "Go to Documentation") | Returns $\alpha$ (a) and $\beta$ (b) orbital energies. (Identical for restricted orbitals) |
#
# Note: The functions returning any of the matrices mentioned above (**C**, **D**, $\boldsymbol{\epsilon}$), actually
# return instances of the `psi4.core.Matrix` class (noticing a pattern here?) and not viewable arrays.  Fortunately,
# the previous tutorial introduced how to modify these arrays Julia-side, using Julia views created through our `psi4view()`.
#
# The full list is quite extensive; however, this likely comprises the most utilized functions. It should be noted that the "a" stands for alpha and conversely the beta quantities can be accessed with the letter "b". For now let's ensure that all computations have C1 symmetry; molecular symmetry can be utilized in Psi4NumPy computations but adds significant complexity to the code.

# + run_control={"frozen": false, "read_only": false}
# Try out the wavefunction class member functions!
