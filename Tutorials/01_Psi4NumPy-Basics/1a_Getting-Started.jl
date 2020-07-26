# ---
# jupyter:
#   jupytext:
#     formats: jl,ipynb
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

# # <span style="font-family: Optima, sans-serif; color: #273896;">P<span style="font-size: 82%;">SI<span>4</span> Acquisition

# ### Easy choice: clone and compile
#
# * Clone from [GitHub](https://github.com/psi4/psi4)
# * Compile via [CMake](https://github.com/psi4/psi4/blob/master/CMakeLists.txt#L16-L123)
# * Build FAQ and guide [here](http://psicode.org/psi4manual/master/build_faq.html)

# ### Easier choice: conda
#
# 0. Get [miniconda](https://conda.io/miniconda.html) or [anaconda](https://www.continuum.io/downloads) installer or script
# 0. Run the installer or "bash" the script on the command line, accepting its license and allowing it to add its installation directory to your `PATH` environment variable: _e.g._, ``bash Anaconda3-4.3.0-Linux-x86_64.sh``
# 0. Create a <span style="font-family: Optima, sans-serif; color: #273896;">P<span style="font-size: 82%;">SI</span>4</span> environment named "p4env". Until the 1.1 release, instructions may be found [at the end of this thread](https://github.com/psi4/psi4/issues/466#issuecomment-272589229)
# 0. Activate the environment: ``source activate p4env``
# 0. See [guide](http://psicode.org/psi4manual/master/build_planning.html#faq-runordinarymodule) for any configuration trouble

# ### Easiest choice: conda installer
#
# * **Not available until 1.1 release**
# * ~~Get Psi4+miniconda installer script from [psicode](http://psicode.org/downloads.html) and follow its directions~~

# ### Online choice: binder button
# * **Not available until 1.1 release**

# # <span style="font-family: Optima, sans-serif; color: #273896;">P<span style="font-size: 82%;">SI</span>4</span> Boilerplate
#
#

# ### Eponymous Julia module imports

using PyCall: pyimport

psi4 = pyimport("psi4")
np   = pyimport("numpy")

# ### Direct output and scratch
#
# * Output goes to file ``output.dat``
# * Boolean directs overwriting (`true`) rather than appending (`false`).
# * Optionally, redirect scratch away from ``/tmp`` to existing, writable directory

# +
psi4.set_output_file("output.dat", true)

# optional
#psi4.core.IOManager.shared_object().set_default_path("/scratch")
# -

# ### Set memory limits
#
# * Give 500 Mb of memory to <span style="font-family: Optima, sans-serif; color: #273896;">P<span style="font-size: 82%;">SI</span>4</span>
# * Give 2 Gb of memory for NumPy arrays (quantity for Psi4Julia project, *not* passed to NumPy)
# * Sum of these two should nowhere approach the RAM of your computer

psi4.set_memory(Int(5e8))
numpy_memory = 2

# ### Molecule and Basis
#
# * Covered in detail in subsequent tutorials. This is the quick reference
# * Running _without_ symmetry recommended in Psi4NumPy for simplest expressions

# +
psi4.geometry("""
O 0.0 0.0 0.0 
H 1.0 0.0 0.0
H 0.0 1.0 0.0
symmetry c1
""")

psi4.set_options(Dict("basis" => "cc-pvdz"))
