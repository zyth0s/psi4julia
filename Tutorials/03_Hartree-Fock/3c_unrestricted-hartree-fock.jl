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

# # Unrestricted Open-Shell Hartree-Fock
#
# In the first two tutorials in this module, we wrote programs which implement a closed-shell formulation of Hartree-Fock theory using restricted orbitals, aptly named Restricted Hartree-Fock (RHF).  In this tutorial, we will abandon strictly closed-shell systems and the notion of restricted orbitals, in favor of a more general theory known as Unrestricted Hartree-Fock (UHF) which can accommodate more diverse molecules.  In UHF, the orbitals occupied by spin up ($\alpha$) electrons and those occupied by spin down ($\beta$) electrons no longer have the same spatial component, e.g., 
#
# $$\chi_i({\bf x}) = \begin{cases}\psi^{\alpha}_j({\bf r})\alpha(\omega) \\ \psi^{\beta}_j({\bf r})\beta(\omega)\end{cases},$$
#
# meaning that they will not have the same orbital energy.  This relaxation of orbital constraints allows for more variational flexibility, which leads to UHF always being able to find a lower total energy solution than RHF.  
#
# ## I. Theoretical Overview
# In UHF, we seek to solve the coupled equations
#
# \begin{align}
# {\bf F}^{\alpha}{\bf C}^{\alpha} &= {\bf SC}^{\alpha}{\bf\epsilon}^{\alpha} \\
# {\bf F}^{\beta}{\bf C}^{\beta} &= {\bf SC}^{\beta}{\bf\epsilon}^{\beta},
# \end{align}
#
# which are the unrestricted generalizations of the restricted Roothan equations, called the Pople-Nesbet-Berthier equations.  Here, the one-electron Fock matrices are given by
#
# \begin{align}
# F_{\mu\nu}^{\alpha} &= H_{\mu\nu} + (\mu\,\nu\mid\lambda\,\sigma)[D_{\lambda\sigma}^{\alpha} + D_{\lambda\sigma}^{\beta}] - (\mu\,\lambda\,\mid\nu\,\sigma)D_{\lambda\sigma}^{\beta}\\
# F_{\mu\nu}^{\beta} &= H_{\mu\nu} + (\mu\,\nu\mid\,\lambda\,\sigma)[D_{\lambda\sigma}^{\alpha} + D_{\lambda\sigma}^{\beta}] - (\mu\,\lambda\,\mid\nu\,\sigma)D_{\lambda\sigma}^{\alpha},
# \end{align}
#
# where the density matrices $D_{\lambda\sigma}^{\alpha}$ and $D_{\lambda\sigma}^{\beta}$ are given by
#
# \begin{align}
# D_{\lambda\sigma}^{\alpha} &= C_{\sigma i}^{\alpha}C_{\lambda i}^{\alpha}\\
# D_{\lambda\sigma}^{\beta} &= C_{\sigma i}^{\beta}C_{\lambda i}^{\beta}.
# \end{align}
#
# Unlike for RHF, the orbital coefficient matrices ${\bf C}^{\alpha}$ and ${\bf C}^{\beta}$ are of dimension $M\times N^{\alpha}$ and $M\times N^{\beta}$, where $M$ is the number of AO basis functions and $N^{\alpha}$ ($N^{\beta}$) is the number of $\alpha$ ($\beta$) electrons.  The total UHF energy is given by
#
# \begin{align}
# E^{\rm UHF}_{\rm total} &= E^{\rm UHF}_{\rm elec} + E^{\rm BO}_{\rm nuc},\;\;{\rm with}\\
# E^{\rm UHF}_{\rm elec} &= \frac{1}{2}[({\bf D}^{\alpha} + {\bf D}^{\beta}){\bf H} + 
# {\bf D}^{\alpha}{\bf F}^{\alpha} + {\bf D}^{\beta}{\bf F}^{\beta}].
# \end{align}
#
# ## II. Implementation
#
# In any SCF program, there will be several common elements which can be abstracted from the program itself into separate modules, classes, or functions to 'clean up' the code that will need to be written explicitly; examples of this concept can be seen throughout the Psi4Julia reference implementations.  For the purposes of this tutorial, we can achieve some degree of code cleanup without sacrificing readabilitiy and clarity by focusing on abstracting only the parts of the code which are both 
# - Lengthy subroutines, and 
# - Used repeatedly.  
#
# In our UHF program, let's use what we've learned in the last tutorial by also implementing DIIS convergence accelleration for our SCF iterations.  With this in mind, two subroutines in particular would benefit from abstraction are
#
# 1. Orthogonalize & diagonalize Fock matrix
# 2. Extrapolate previous trial vectors for new DIIS solution vector
#
# Before we start writing our UHF program, let's try to write functions which can perform the above tasks so that we can use them in our implementation of UHF.  Recall that defining functions in Julia has the following syntax:
# ~~~julia
# function function_name(args; kwargs)
#     # function block
#     return_values
# end
# ~~~
# A thorough discussion of defining functions in Julia can be found [here](https://docs.julialang.org/en/v1/manual/functions/index.html "Go to Julia docs").  First, let's write a function which can diagonalize the Fock matrix and return the orbital coefficient matrix **C** and the density matrix **D**.  From our RHF tutorial, this subroutine is executed with:
# ~~~julia
# F_p =  A * F * A
# e, C_p = eigen(Hermitian(F_p))
# C = A * C_p
# C_occ = C[:, 1:ndocc]
# D = C_occ * C_occ'
# ~~~
# Examining this code block, there are three quantities which must be specified beforehand:
# - Fock matrix, **F**
# - Orthogonalization matrix, ${\bf A} = {\bf S}^{-1/2}$
# - Number of doubly occupied orbitals, `ndocc`
#
# However, since the orthogonalization matrix **A** is a static quantity (only built once, then left alone) we may choose to leave **A** as a *global* quantity, instead of an argument to our function.  In the cell below, using the code snippet given above, write a function `diag_F()` which takes **F** and the number of orbitals `norb` as arguments, and returns **C** and **D**:

# ==> Define function to diagonalize F <==
function diag_F(F, norb, A)
    F_p = A * F * A
    e, C_p = eigen(Hermitian(F_p))
    C = A * C_p
    C_occ = C[:, 1:norb]
    D = C_occ * C_occ'
    C, D
end


# Next, let's write a function to perform DIIS extrapolation and generate a new solution vector.  Recall that the DIIS-accellerated SCF algorithm is:
# #### Algorithm 1: DIIS within a generic SCF Iteration
# 1. Compute **F**, append to list of previous trial vectors
# 2. Compute AO orbital gradient **r**, append to list of previous residual vectors
# 3. Compute RHF energy
# 3. Check convergence criteria
#     - If RMSD of **r** sufficiently small, and
#     - If change in SCF energy sufficiently small, break
# 4. Build **B** matrix from previous AO gradient vectors
# 5. Solve Pulay equation for coefficients $\{c_i\}$
# 6. Compute DIIS solution vector **F_DIIS** from $\{c_i\}$ and previous trial vectors
# 7. Compute new orbital guess with **F_DIIS**
#
# In our function, we will perform steps 4-6 of the above algorithm.  What information will we need to provide our function in order to do so?  To build **B** (step 4 above) in the previous tutorial, we used:
# ~~~julia
# # Build B matrix
# B_dim = length(F_list) + 1
# B = zeros(B_dim, B_dim)
# B[end,   :] .= -1
# B[:  , end] .= -1
# B[end, end]  = 0
# for i in eachindex(F_list), j in eachindex(F_list)
#     B[i, j] = dot(DIIS_RESID[i], DIIS_RESID[j])
# end
# ~~~
# Here, we see that we must have all previous DIIS residual vectors (`DIIS_RESID`), as well as knowledge about how many previous trial vectors there are (for the dimension of **B**).  To solve the Pulay equation (step 5 above):
# ~~~julia
# # Build RHS of Pulay equation 
# rhs = zeros(B_dim)
# rhs[end] = -1
#       
# # Solve Pulay equation for c_i's with NumPy
# coeff = B \ rhs
# ~~~
# For this step, we only need the dimension of **B** (which we computed in step 4 above) and a Julia routine, so this step doesn't require any additional arguments.  Finally, to build the DIIS Fock matrix (step 6):
# ~~~julia
# # Build DIIS Fock matrix
# F = zeros(size(F_list[0]))
# for x in 1:length(coeff) - 1
#     F += coeff[x] * F_list[x]
# end
# ~~~
# Clearly, for this step, we need to know all the previous trial vectors (`F_list`) and the coefficients we generated in the previous step.  In the cell below, write a funciton `diis_xtrap()` according to Algorithm 1 steps 4-6, using the above code snippets, which takes a list of previous trial vectors `F_list` and residual vectors `DIIS_RESID` as arguments and returns the new DIIS solution vector `F_DIIS`:

# ==> Build DIIS Extrapolation Function <==
function diis_xtrap(F_list, DIIS_RESID)
    # Build B matrix
    B_dim = length(F_list) + 1
    B = zeros(B_dim, B_dim)
    B[end,   :] .= -1
    B[:  , end] .= -1
    B[end, end]  =  0
    for i in eachindex(F_list), j in eachindex(F_list)
       B[i, j] = dot(DIIS_RESID[i],  DIIS_RESID[j])
    end

    # Build RHS of Pulay equation 
    rhs = zeros(B_dim)
    rhs[end] = -1
    
    # Solve Pulay equation for c_i's with Julia
    coeff = B \ rhs
    
    # Build DIIS Fock matrix
    F = zeros(size(F_list[1]))
    for i in 1:length(coeff) - 1
       F += coeff[i] * F_list[i]
    end
    F
end

# We are now ready to begin writing our UHF program!  Let's begin by importing <span style='font-variant: small-caps'> Psi4 </span>, NumPy, TensorOperations, LinearAlgebra, and defining our molecule & basic options:

# ==> Import Psi4 & NumPy <==
using PyCall: pyimport
psi4 = pyimport("psi4")
np   = pyimport("numpy") # used only to cast to Psi4 arrays
using TensorOperations: @tensor
using LinearAlgebra: Diagonal, Hermitian, eigen, tr, norm, dot
using Printf: @printf

# +
# ==> Set Basic Psi4 Options <==
# Memory specification
psi4.set_memory(Int(5e8))
numpy_memory = 2

# Set output file
psi4.core.set_output_file("output.dat", false)

# Define Physicist's water -- don't forget C1 symmetry!
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

# Set computation options
psi4.set_options(Dict("basis"         => "cc-pvdz",
                      "scf_type"      => "pk",
                      "e_convergence" => 1e-8,
                      "guess"         => "core",
                      "reference"     => "uhf"))
# -

# You may notice that in the above `psi4.set_options()` block, there are two additional options -- namely, `'guess': 'core'` and `'reference': 'uhf'`.  These options make sure that when we ultimately check our program against <span style='font-variant: small-caps'> Psi4</span>, the options <span style='font-variant: small-caps'> Psi4 </span> uses are identical to our implementation.  Next, let's define the options for our UHF program; we can borrow these options from our RHF implementation with DIIS accelleration that we completed in our last tutorial.

# ==> Set default program options <==
# Maximum SCF iterations
MAXITER = 40
# Energy convergence criterion
E_conv = 1.0e-6
D_conv = 1.0e-3

# Static quantities like the ERI tensor, core Hamiltonian, and orthogonalization matrix have exactly the same form in UHF as in RHF.  Unlike in RHF, however, we will need the number of $\alpha$ and $\beta$ electrons.  Fortunately, both these values are available through querying the Wavefunction object.  In the cell below, generate these static objects and compute each of the following:
# - Number of basis functions, `nbf`
# - Number of alpha electrons, `nalpha`
# - Number of beta electrons, `nbeta`
# - Number of doubly occupied orbitals, `ndocc` (Hint: In UHF, there can be unpaired electrons!)

# +
# ==> Compute static 1e- and 2e- quantities with Psi4 <==
# Class instantiation
wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option("basis"))
mints = psi4.core.MintsHelper(wfn.basisset())

# Overlap matrix
S = np.asarray(mints.ao_overlap()) # we only need a copy

# Number of basis Functions, alpha & beta orbitals, and # doubly occupied orbitals
nbf = wfn.nso()
nalpha = wfn.nalpha()
nbeta = wfn.nbeta()
ndocc = min(nalpha, nbeta)

println("Number of basis functions: ", nbf)
println("Number of singly occupied orbitals: ", abs(nalpha-nbeta))
println("Number of doubly occupied orbitals: ", ndocc)

# Memory check for ERI tensor
I_size = nbf^4 * 8.e-9
println("\nSize of the ERI tensor will be $I_size GB.")
memory_footprint = I_size * 1.5
if I_size > numpy_memory
    psi4.core.clean()
    throw(OutOfMemoryError("Estimated memory utilization ($memory_footprint GB) exceeds " * 
                           "allotted memory limit of $numpy_memory GB."))
end

# Build ERI Tensor
I = np.asarray(mints.ao_eri()) # we only need a copy

# Build core Hamiltonian
T = np.asarray(mints.ao_kinetic()) # we only need a copy
V = np.asarray(mints.ao_potential()) # we only need a copy
H = T + V;

# Construct AO orthogonalization matrix A
A = mints.ao_overlap()
A.power(-0.5, 1.e-16) # ≈ Julia's A^(-0.5) after psi4view()
A = np.asarray(A);
# -

# Unlike the static quantities above, the CORE guess in UHF is slightly different than in RHF.  Since the $\alpha$ and $\beta$ electrons do not share spatial orbitals, we must construct a guess for *each* of the $\alpha$ and $\beta$ orbitals and densities.  In the cell below, using the function `diag_F()`, construct the CORE guesses and compute the nuclear repulsion energy:
#
# (Hint: The number of $\alpha$ orbitals is the same as the number of $\alpha$ electrons!)

# +
# ==> Build alpha & beta CORE guess <==
Ca, Da = diag_F(H, nalpha, A)
Cb, Db = diag_F(H, nbeta, A)

# Get nuclear repulsion energy
E_nuc = mol.nuclear_repulsion_energy()
# -

# We are almost ready to perform our SCF iterations; beforehand, however, we must initiate variables for the current & previous SCF energies, and the lists to hold previous residual vectors and trial vectors for the DIIS procedure.  Since, in UHF, there are Fock matrices ${\bf F}^{\alpha}$ and ${\bf F}^{\beta}$ for both $\alpha$ and $\beta$ orbitals, we must apply DIIS to each of these matrices separately.  In the cell below, define empty lists to hold previous Fock matrices and residual vectors for both $\alpha$ and $\beta$ orbitals:

# ==> Pre-Iteration Setup <==
# SCF & Previous Energy
SCF_E = 0.0
E_old = 0.0

# We are now ready to write the SCF iterations.  The algorithm for UHF-SCF iteration, with DIIS convergence accelleration, is:
# #### Algorithm 2: DIIS within UHF-SCF Iteration
# 1. Build ${\bf F}^{\alpha}$ and ${\bf F}^{\beta}$, append to trial vector lists
# 2. Compute the DIIS residual for $\alpha$ and $\beta$, append to residual vector lists
# 3. Compute UHF energy
# 4. Convergence check
#     - If average of RMSD of $\alpha$ and $\beta$ residual sufficiently small, and
#     - If change in UHF energy sufficiently small, break
# 5. DIIS extrapolation of ${\bf F}^{\alpha}$ and ${\bf F}^{\beta}$ to form new solution vector
# 6. Compute new ${\alpha}$ and ${\beta}$ orbital & density guesses
#
# In the cell below, write the UHF-SCF iteration according to Algorithm 2:
#
# (Hint: Use your functions `diis_xtrap()` and `diag_F` for Algorithm 2 steps 5 & 6, respectively)

SCF_E = let SCF_E = SCF_E, E_old = E_old, Da = Da, Db = Db, A = A, I = I, H = H, S = S

   # Trial & Residual Vector Lists -- one each for α & β
   F_list_a = []
   F_list_b = []
   R_list_a = []
   R_list_b = []

   # ==> UHF-SCF Iterations <==
   println("==> Starting SCF Iterations <==")

   # Begin Iterations
   for scf_iter in 1:MAXITER
       # Build Fa & Fb matrices
       @tensor Ja[p,q] := I[p,q,r,s] * Da[r,s]
       @tensor Jb[p,q] := I[p,q,r,s] * Db[r,s]
       @tensor Ka[p,q] := I[p,r,q,s] * Da[r,s]
       @tensor Kb[p,q] := I[p,r,q,s] * Db[r,s]
       Fa = H + (Ja + Jb) - Ka
       Fb = H + (Ja + Jb) - Kb

       # Compute DIIS residual for Fa & Fb
       diis_r_a = A * (Fa * Da * S - S * Da * Fa) * A
       diis_r_b = A * (Fb * Db * S - S * Db * Fb) * A
       
       # Append trial & residual vectors to lists
       push!(F_list_a, Fa)
       push!(F_list_b, Fb)
       push!(R_list_a, diis_r_a)
       push!(R_list_b, diis_r_b)
       
       # Compute UHF Energy
       SCF_E = 0.5*tr( H*(Da + Db) + Fa*Da + Fb*Db) + E_nuc
       
       dE = SCF_E - E_old
       dRMS = 0.5(norm(diis_r_a) + norm(diis_r_b))
       @printf("SCF Iteration %3d: Energy = %4.16f dE = %1.5e dRMS = %1.5e \n",
                           scf_iter,        SCF_E, SCF_E - E_old,     dRMS)
       
       # Convergence Check
       if abs(dE) < E_conv && dRMS < D_conv
           break
       end
       E_old = SCF_E
       
       # DIIS Extrapolation
       if scf_iter >= 2
           Fa = diis_xtrap(F_list_a, R_list_a)
           Fb = diis_xtrap(F_list_b, R_list_b)
       end
       
       # Compute new orbital guess
       Ca, Da = diag_F(Fa, nalpha, A)
       Cb, Db = diag_F(Fb, nbeta, A)
       
       # MAXITER exceeded?
       if scf_iter == MAXITER
           psi4.core.clean()
           throw(MethodError("Maximum number of SCF iterations exceeded."))
       end
   end
   SCF_E
end
# Post iterations
println("\nSCF converged.")
println("Final RHF Energy: $SCF_E [Eh]")
println()

# Congratulations! You've written your very own Unrestricted Hartree-Fock program with DIIS convergence accelleration!  Finally, let's check your final UHF energy against <span style='font-variant: small-caps'> Psi4</span>:

# Compare to Psi4
SCF_E_psi = psi4.energy("SCF")
SCF_E
psi4.compare_values(SCF_E_psi, SCF_E, 6, "SCF Energy")

# ## References
# 1. A. Szabo and N. S. Ostlund, *Modern Quantum Chemistry*, Introduction to Advanced Electronic Structure Theory. Courier Corporation, 1996.
# 2. I. N. Levine, *Quantum Chemistry*. Prentice-Hall, New Jersey, 5th edition, 2000.
# 3. T. Helgaker, P. Jorgensen, and J. Olsen, *Molecular Electronic Structure Theory*, John Wiley & Sons Inc, 2000.
