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

# # Hartree-Fock Self-Consistent Field Theory

# +
"""Tutorial implementing a basic Hartree--Fock SCF program."""

__authors__ = ["D. Menendez", "D. A. Sirianni"]
__credits__ = ["D. Menendez", "D. G. A. Smith"]
__email__   = "danielmail7@gmail.com"

__copyright__ = "(c) 2014-2020, The Psi4Julia Developers"
__license__   = "BSD-3-Clause"
__date__      = "07/28/2020"
# -

# ## I. Theoretical Overview
# In this tutorial, we will seek to introduce the theory and implementation of the quantum chemical method known as Hartree-Fock Self-Consistent Field Theory (HF-SCF) with restricted orbitals and closed-shell systems (RHF).  This theory seeks to solve the pseudo-eigenvalue matrix equation 
#
# $$\sum_{\nu} F_{\mu\nu}C_{\nu i} = \epsilon_i\sum_{\nu}S_{\mu\nu}C_{\nu i}$$
# $${\bf FC} = {\bf SC\epsilon},$$
#
# called the Roothan equations, which can be solved self-consistently for the orbital coefficient matrix **C** and the orbital energy eigenvalues $\epsilon_i$.  The Fock matrix, **F**, has elements $F_{\mu\nu}$ given (in the atomic orbital basis) as
#
# $$F_{\mu\nu} = H_{\mu\nu} + 2(\mu\,\nu\left|\,\lambda\,\sigma)D_{\lambda\sigma} - (\mu\,\lambda\,\right|\nu\,\sigma)D_{\lambda\sigma},$$
#
# where $D_{\lambda\sigma}$ is an element of the one-particle density matrix **D**, constructed from the orbital coefficient matrix **C**:
#
# $$D_{\lambda\sigma} = C_{\sigma i}C_{\lambda i}$$
#
# Formally, the orbital coefficient matrix **C** is a $N\times M$ matrix, where $N$ is the number of atomic basis functions, and $M$ is the total number of molecular orbitals.  Physically, this matrix describes the contribution of every atomic basis function (columns) to a particular molecular orbital (e.g., the $i^{\rm th}$ row).  The density matrix **D** is a square matrix describing the electron density contained in each orbital.  In the molecular orbital basis, the density matrix has elements
#
# $$D_{pq} = \left\{
# \begin{array}{ll}
# 2\delta_{pq} & p\; {\rm occupied} \\
# 0 & p\; {\rm virtual} \\
# \end{array}\right .$$
#
# The total RHF energy is given by
#
# $$E^{\rm RHF}_{\rm total} = E^{\rm RHF}_{\rm elec} + E^{\rm BO}_{\rm nuc},$$
#
# where $E^{\rm RHF}_{\rm elec}$ is the final electronic RHF energy, and $E^{\rm BO}_{\rm nuc}$ is the total nuclear repulsion energy within the Born-Oppenheimer approximation.  To compute the electronic energy, we may use the density matrix in the AO basis:
#
# $$E^{\rm RHF}_{\rm elec} = (F_{\mu\nu} + H_{\mu\nu})D_{\mu\nu},$$
#
# and the nuclear repulsion energy is simply
#
# $$E^{\rm BO}_{\rm nuc} = \sum_{A>B}\frac{Z_AZ_B}{r_{AB}}$$
#
# where $Z_A$ is the nuclear charge of atom $A$, and the sum runs over all unique nuclear pairs.

# ## II. Implementation
#
# Using the above overview, let's write a RHF program using <span style="font-variant: small-caps"> Psi4 </span>, NumPy, TensorOperations, and Julia's LinearAlgebra standard library.  First, we need to import these modules: 

# ==> Import Psi4 & NumPy <==
using PyCall: pyimport
psi4 = pyimport("psi4")
np   = pyimport("numpy") # used only to cast to Psi4 arrays
using TensorOperations: @tensor
using LinearAlgebra: Diagonal, Hermitian, eigen, tr
using Printf: @printf

# Next, using what you learned in the previous tutorial module, set the following <span style="font-variant: small-caps"> Psi4 </span> and molecule options.
#
# Memory & Output specifications:
# - Give 500 Mb of memory to Psi4
# - Set Psi4 output file to "output.dat"
# - Set a variable `numpy_memory` to an acceptable amount of available memory for the working computer to use for storing tensors
#
# Molecule definition:
# - Define the "physicist's water molecule" (O-H bond length = 1.1 Angstroms, HOH bond angle = 104 degrees)
# - Molecular symmetry C1
#
# Computation options:
# - basis set cc-pVDZ
# - SCF type PK
# - Energy convergence criterion to 0.00000001
#

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
psi4.set_options(Dict("basis" => "cc-pvdz",
                      "scf_type" => "pk",
                      "e_convergence" => 1e-8))
# -

# Since we will be writing our own, iterative RHF procedure, we will need to define options that we can use to tweak our convergence behavior.  For example, if something goes wrong and our SCF doesn't converge, we don't want to spiral into an infinite loop.  Instead, we can specify the maximum number of iterations allowed, and store this value in a variable called `maxiter`.  Here are some good default options for our program:
# ~~~python
# MAXITER = 40
# E_conv = 1.0e-6
# ~~~
# These are by no means the only possible values for these options, and it's encouraged to try different values and see for yourself how different choices affect the performance of our program.  For now, let's use the above as our default.

# ==> Set default program options <==
# Maximum SCF iterations
MAXITER = 40
# Energy convergence criterion
E_conv = 1.0e-6;

# Before we can build our Fock matrix, we'll need to compute the following static one- and two-electron quantities:
#
# - Electron repulsion integrals (ERIs) **I** between our AOs
# - Overlap matrix **S**
# - Core Hamiltonian matrix **H**
#
# Fortunately for us, we can do this using the machinery in <span style='font-variant: small-caps'> Psi4</span>.  In the first module, you learned about `psi4.core.Wavefunction` and `psi4.core.MintsHelper` classes.  In the cell below, use these classes to perform the following:
#
# 1. Create Class Instances
#
#     a. Build a wavefunction for our molecule and basis set
#     
#     b. Create an instance of the `MintsHelper` class with the basis set for the wavefunction
#
# 2. Build overlap matrix, **S**
#
#     a. Get the AO overlap matrix from `MintsHelper`, and cast it into a Julia array
#     
#     b. Get the number of AO basis functions and number of doubly occupied orbitals from S and the wavefunciton
#
# 3. Compute ERI Tensor, **I**
#
#     a. Get ERI tensor from `MintsHelper`, and cast it into a Julia array
#
# 4. Build core Hamiltonian, **H**
#
#     a. Get AO kinetic energy matrix from `MintsHelper`, and cast it into a Julia array
#
#     b. Get AO potential energy matrix from `MintsHelper`, and cast it into a Julia array
#
#     c. Build core Hamiltonian from kinetic & potential energy matrices

# +
# ==> Compute static 1e- and 2e- quantities with Psi4 <==
# Class instantiation
wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option("basis"))
mints = psi4.core.MintsHelper(wfn.basisset())

# Overlap matrix
S = np.asarray(mints.ao_overlap()) # we only need a copy

# Number of basis Functions & doubly occupied orbitals
nbf = size(S)[1]
ndocc = wfn.nalpha()

println("Number of occupied orbitals: ", ndocc)
println("Number of basis functions: ", nbf)

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
# -

# The Roothan equations
#
# $${\bf FC} = {\bf SC\epsilon}$$
#
# are only *pseudo*-eigenvalue equations due to the presence of the overlap matrix **S** on the right hand side of the equation.  Normally, the AO basis set will not be orthonormal, so the overlap matrix **S** will not be unity and therefore cannot be ignored.  Let's check to see whether our AO basis is orthonormal:

# ==> Inspecting S for AO orthonormality <==
hope = S ≈ Diagonal(ones(size(S)[1]))
println("\nDo we have any hope that our AO basis is orthonormal? ", hope)

# Just as we'd expected -- looks like we can't ignore the AO overlap matrix.  Therefore, the Fock matrix **F** cannot simply be diagonalized to solve for the orbital coefficient matrix **C**.  There is still hope, however!  We can overcome this issue by transforming the AO basis so that all of our basis functions are orthonormal.  In other words, we seek a matrix **A** such that the transformation 
#
# $${\bf A}^{\dagger}{\bf SA} = {\bf 1}$$
#
# One method of doing this is called *symmetric orthogonalization*, which lets ${\bf A} = {\bf S}^{-1/2}$.  Then, 
#
# $${\bf A}^{\dagger}{\bf SA} = {\bf S}^{-1/2}{\bf SS}^{-1/2} = {\bf S}^{-1/2}{\bf S}^{1/2} = {\bf S}^0 = {\bf 1},$$
#
# and we see that this choice for **A** does in fact yield an orthonormal AO basis.  In the cell below, construct this transformation matrix using <span style='font-variant: small-caps'> Psi4</span>'s built-in `Matrix` class member function `power()` just like the following:
# ~~~python
# A = mints.ao_overlap()
# A.power(-0.5, 1.e-16)
# A = np.asarray(A)
# ~~~

# +
# ==> Construct AO orthogonalization matrix A <==
A = mints.ao_overlap()
A.power(-0.5, 1.e-16) # ≈ Julia's A^(-0.5) after psi4view()
A = np.asarray(A) # we only need a copy

# Check orthonormality
S_p = A * S * A
new_hope = S ≈ Diagonal(ones(size(S)[1]))

if new_hope
    println("There is a new hope for diagonalization!")
else
    println("Whoops...something went wrong. Check that you've correctly built the transformation matrix.")
    @show sum(S_p) - tr(S_p)
end
# -

# The drawback of this scheme is that we would now have to either re-compute the ERI and core Hamiltonian tensors in the newly orthogonal AO basis, or transform them using our **A** matrix (both would be overly costly, especially transforming **I**).  On the other hand, substitute ${\bf C} = {\bf AC}'$ into the Roothan equations:
#
# \begin{align}
# {\bf FAC'} &= {\bf SAC}'{\bf \epsilon}\\
# {\bf A}^{\dagger}({\bf FAC}')&= {\bf A}^{\dagger}({\bf SAC}'){\bf \epsilon}\\
# ({\bf A}^{\dagger}{\bf FA}){\bf C}'&= ({\bf A}^{\dagger}{\bf SA}){\bf C}'{\bf \epsilon}\\
# {\bf F}'{\bf C}' &= {\bf 1C}'{\bf \epsilon}\\
# {\bf F}'{\bf C}' &= {\bf C}'{\bf \epsilon}\\
# \end{align}
#
# Clearly, we have arrived at a canonical eigenvalue equation.  This equation can be solved directly for the transformed orbital coefficient matrix ${\bf C}'$ by diagonalizing the transformed Fock matrix, ${\bf F}'$, before transforming ${\bf C}'$ back into the original AO basis with ${\bf C} = {\bf AC}'$.  
#
# Before we can get down to the business of using the Fock matrix **F** to compute the RHF energy, we first need to compute the orbital coefficient **C** matrix.  But, before we compute the **C** matrix, we first need to build **F**.  Wait...hold on a second.  Which comes first, **C** or **F**?  Looking at the Roothan equations more closely, we see that that both sides depend on the **C** matrix, since **F** is a function of the orbitals:
#
#
# $${\bf F}({\bf C}){\bf C} = {\bf SC\epsilon}\,;\;\;F_{\mu\nu} = H_{\mu\nu} + 2(\mu\,\nu\mid\lambda\,\sigma)C_{\sigma i}C_{\lambda i} - (\mu\,\lambda\,\mid\nu\,\sigma)C_{\sigma i}C_{\lambda i}.$$
#
# Therefore technically, *neither* **F** nor **C** can come first!  In order to proceed, we instead begin with a *guess* for the Fock matrix, from which we obtain a guess at the **C** matrix.  Without orbital coefficients (and therefore without electron densities), the most logical starting point for obtaining a guess at the Fock matrix is to begin with the only component of **F** that does *not* involve densities: the core Hamiltonian, **H**.  Below, using the `eigen()` function, and forcing hermitianess with `Hermitian()`, obtain coefficient and density matrices using the core guess:
#
# 1. Obtain ${\bf F}'$ by transforming the core Hamiltonian with the ${\bf A}$ matrix
# 2. Diagonalize the transformed Fock matrix for $\epsilon$ and ${\bf C}'$
# 3. Use doubly-occupied slice of coefficient matrix to build density matrix

# +
# ==> Compute C & D matrices with CORE guess <==
# Transformed Fock matrix
F_p = A * H * A

# Diagonalize F_p for eigenvalues & eigenvectors with NumPy
e, C_p = eigen(Hermitian(F_p))

# Transform C_p back into AO basis
C = A * C_p

# Grab occupied orbitals
C_occ = C[:, 1:ndocc]

# Build density matrix from occupied orbitals
D = C_occ * C_occ' ;
# -

# The final quantity we need to compute before we can proceed with our implementation of the SCF procedure is the Born-Oppenheimer nuclear repulsion energy, $E^{\rm BO}_{\rm nuc}$.  We could use the expression given above in $\S$1, however we can also obtain this value directly from <span style='font-variant: small-caps'> Psi4</span>'s `Molecule` class.  In the cell below, compute the nuclear repulsion energy using either method. 

# ==> Nuclear Repulsion Energy <==
E_nuc = mol.nuclear_repulsion_energy()

# Within each SCF iteration, we'll have to perform a number of tensor contractions when building the Fock matrix, computing the total RHF energy, and performing several transformations.  Since the computational expense of this process is related to the number of unique indices, the most intensive step of computing the total electronic energy will be performing the four-index contractions corresponding to building Coulomb and Exchange matrices **J** and **K**, with elements
#
# \begin{align}
# J[D_{\lambda\sigma}]_{\mu\nu} &= (\mu\,\nu\mid\lambda\,\sigma)D_{\lambda\sigma}\\
# K[D_{\lambda\sigma}]_{\mu\nu} &= (\mu\,\lambda\mid\nu\,\sigma)D_{\lambda\sigma},
# \end{align}
#
# when building the Fock matrix.  Fortunately, once **J** and **K** have been built, the Fock matrix may be computed as a simple matrix addition, instead of element-wise:
#
# $$ {\bf F} = {\bf H} + 2{\bf J} - {\bf K} = {\bf H} + {\bf G}.$$
#
# Formation of the **J** and **K** matrices will be the most expensive step of the RHF procedure, scaling with respect to the number of AOs as ${\cal O}(N^4)$.  Strategies for building these marices efficiently, as well as different methods for handling these tensor contractions, will be discussed in greater detail in tutorials 2c and 2d in this module, respectively. 
#
# Let's now write our SCF iterations according to the following algorithm:
#
# #### Algorithm 1: SCF Iteration
# for `scf_iter` less than `MAXITER`, do:
# 1. Build Fock matrix
#     - Build the two-electron Coulomb & Exchange matrix **G** 
#     - Form the Fock matrix
# 2. RHF Energy
#     - Compute total RHF energy   
#     - If change in RHF energy less than `E_conv`, break    
#     - Save latest RHF energy as `E_old`
# 3. Compute new orbital guess
#     - Transform Fock matrix to orthonormal AO basis    
#     - Diagonalize ${\bf F}'$ for $\epsilon$ and ${\bf C}'$    
#     - Back transform ${\bf C}'$ to AO basis    
#     - Form **D** from occupied orbital slice of **C**
#

# +
# ==> SCF Iterations <==
# Output and pre-iteration energy declarations
SCF_E = let SCF_E = 0.0, E_old = 0.0, D = D

   print("==> Starting SCF Iterations <==\n")

   # Begin Iterations
   for scf_iter in 1:MAXITER
       # Build Fock matrix
       @tensor G[p,q] := (2I[p,q,r,s] - I[p,r,q,s]) * D[r,s]
       F = H + G
       
       # Compute RHF energy
       SCF_E = tr((H + F) * D) + E_nuc
       @printf("SCF Iteration %3d: Energy = %4.16f dE = %1.5e \n",scf_iter, SCF_E, SCF_E - E_old)
       
       # SCF Converged?
       if abs(SCF_E - E_old) < E_conv
           break
       end
       E_old = SCF_E
       
       # Compute new orbital guess
       F_p =  A * F * A
       e, C_p = eigen(Hermitian(F_p))
       C = A * C_p
       C_occ = C[:, 1:ndocc]
       D = C_occ * C_occ'
       
       # MAXITER exceeded?
       if scf_iter == MAXITER
           psi4.core.clean()
           throw(MethodError("Maximum number of SCF iterations exceeded."))
       end
   end
   SCF_E # return RHF SCF energy
end

# Post iterations
println("\nSCF converged.")
println("Final RHF Energy: $SCF_E [Eh]")
# -

# Congratulations! You've written your very own Restricted Hartree-Fock program!  Finally, let's check your final RHF energy against <span style='font-variant: small-caps'> Psi4</span>:

# Compare to Psi4
SCF_E_psi = psi4.energy("SCF")
psi4.compare_values(SCF_E_psi, SCF_E, 6, "SCF Energy")

# ## References
# 1. [[Szabo:1996](http://store.doverpublications.com/0486691861.html)] A. Szabo and N. S. Ostlund, *Modern Quantum Chemistry*, Introduction to Advanced Electronic Structure Theory. Courier Corporation, 1996.
# 2. [[Levine:2000](https://books.google.com/books?id=80RpQgAACAAJ&dq=levine%20quantum%20chemistry%205th%20edition&source=gbs_book_other_versions)] I. N. Levine, *Quantum Chemistry*. Prentice-Hall, New Jersey, 5th edition, 2000.
# 3. [[Helgaker:2000](https://books.google.com/books?id=lNVLBAAAQBAJ&pg=PT1067&dq=helgaker+molecular+electronic+structure+theory&hl=en&sa=X&ved=0ahUKEwj37I7MkofUAhWG5SYKHaoPAAkQ6AEIKDAA#v=onepage&q=helgaker%20molecular%20electronic%20structure%20theory&f=false)] T. Helgaker, P. Jorgensen, and J. Olsen, *Molecular Electronic Structure Theory*, John Wiley & Sons Inc, 2000.
