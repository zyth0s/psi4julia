Interactive Tutorials
=====================

These tutorials use the Jupyter notebook environment to offer an interactive,
step-by-step introduction to several important concepts in quantum chemistry.
Their goal is to provide the reader with the necessary background to
effectively program quantum chemical methods using the machinery provided by
Psi4 and Julia's ecosystem.  

Below is a list of the available interactive tutorials, grouped by module:

1. Psi4Julia Basics
    * Molecule: Overview of the Molecule class and geometry specification in Psi4
    * Psi4-Julia Data Sharing: Describes the integration of Psi4 Matrix and Julia array objects
    * Wavefunction: Building, passing, and using wavefunction objects from Psi4
    * MintsHelper: Generating one- and two-electron integrals with Psi4
    * Tensor Manipulation: Overview of commonly used tensor engines throughout Psi4NumPy 
    * BasisSet: Building and manipulating basis sets within Psi4

2. Linear Algebra

3. Hartree-Fock Molecular Orbital Theory
    * Restricted Hartree-Fock: Implementation of the closed-shell, restricted orbital formulation of Hartree-Fock theory
    * Direct Inversion of the Iterative Subspace: Theory and integration of the DIIS convergence acceleration method into an RHF program
    * Unrestricted Hartree-Fock: Implementation of the open-shell, unrestricted orbital formulation of Hartree-Fock theory, utilizing DIIS convergence acceleration
    * Density Fitting: Overview of the theory and construction of density-fitted ERIs in Psi4, and an illustrative example of a density-fitted Fock build.

4. Density Functional Theory (requires Psi4 1.2, beta)
    * DFT Grid: Briefly outlines several details of the DFT Grid.
    * LDA Kernel: Discusses how the local density approximation (LDA) is formed and computed.
    * GGA and Meta Kernels: Focuses on higher-rung DFT variants and examines how to compute various functionals.
    * VV10: A guide how to compute the non-local kernel VV10.

5. Møller–Plesset Perturbation Theory 
    * Conventional MP2: Overview and implementation of the conventional formulation of second-order Møller-Plesset Perturbation Theory (MP2), using full 4-index ERIs.
    * Density Fitted MP2: Discusses the implementation of the density-fitted formulation of MP2 with an efficient, low-memory algorithm utilizing permutational ERI symmetry.

13. Geometry Optimization Theory
    * Internal coordinates: the B-matrix and coordinate transformations.
    * Hessians: estimating and transforming second derivatives.
    * Hessians: update schemes with the example of BFGS.
    * RFO: Rational Function Optimization minimizations.
    * The Step-Backtransformation: From displacements in internals to displacements in Cartesians.



Note: These tutorials are under active construction.

Julia scripts have the extension `.jl`. They be executed from the command line
```
julia example.jl
```
or, better, from Julia's REPL (avoids recompilation)
```
julia
julia> include("example.jl")
```

Jupyter notebooks have the file extension `.ipynb`.  In order to use these
tutorials, Jupyter must first be installed.  Jupyter is available with the
[Anaconda](https://www.continuum.io/downloads) Python distribution.  Once
installed, a Jupyter notebook `example.ipynb` may be opened from the command
line with
```
jupyter-notebook example.ipynb
```

These modules and the tutorials contained therein assume familiarity
scientific programming and some packages to focus more closely on
the intricacies of programming quantum chemistry.  Before jumping 
into Module 1, it is therefore advantageous to at
the very least skim through the [Julia learning resources](https://julialang.org/learning/) and 
[LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/).  For an
introduction to TensorOperations package, please refer to the 
[TensorOperations.jl](https://jutho.github.io/TensorOperations.jl/stable/indexnotation/).  Good luck and happy
programming!
