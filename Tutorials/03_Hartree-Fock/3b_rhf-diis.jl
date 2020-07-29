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

# # Direct Inversion of the Iterative Subspace
#
# When solving systems of linear (or nonlinear) equations, iterative methods are often employed.  Unfortunately, such methods often suffer from convergence issues such as numerical instability, slow convergence, and significant computational expense when applied to difficult problems.  In these cases, convergence accelleration methods may be applied to both speed up, stabilize and/or reduce the cost for the convergence patterns of these methods, so that solving such problems become computationally tractable.  One such method is known as the direct inversion of the iterative subspace (DIIS) method, which is commonly applied to address convergence issues within self consistent field computations in Hartree-Fock theory (and other iterative electronic structure methods).  In this tutorial, we'll introduce the theory of DIIS for a general iterative procedure, before integrating DIIS into our previous implementation of RHF.
#
# ## I. Theory
#
# DIIS is a widely applicable convergence acceleration method, which is applicable to numerous problems in linear algebra and the computational sciences, as well as quantum chemistry in particular.  Therefore, we will introduce the theory of this method in the general sense, before seeking to apply it to SCF.  
#
# Suppose that for a given problem, there exist a set of trial vectors $\{\mid{\bf p}_i\,\rangle\}$ which have been generated iteratively, converging toward the true solution, $\mid{\bf p}^f\,\rangle$.  Then the true solution can be approximately constructed as a linear combination of the trial vectors,
# $$\mid{\bf p}\,\rangle = \sum_ic_i\mid{\bf p}_i\,\rangle,$$
# where we require that the residual vector 
# $$\mid{\bf r}\,\rangle = \sum_ic_i\mid{\bf r}_i\,\rangle\,;\;\;\; \mid{\bf r}_i\,\rangle 
# =\, \mid{\bf p}_{i+1}\,\rangle - \mid{\bf p}_i\,\rangle$$
# is a least-squares approximate to the zero vector, according to the constraint
# $$\sum_i c_i = 1.$$
# This constraint on the expansion coefficients can be seen by noting that each trial function ${\bf p}_i$ may be represented as an error vector applied to the true solution, $\mid{\bf p}^f\,\rangle + \mid{\bf e}_i\,\rangle$.  Then
# \begin{align}
# \mid{\bf p}\,\rangle &= \sum_ic_i\mid{\bf p}_i\,\rangle\\
# &= \sum_i c_i(\mid{\bf p}^f\,\rangle + \mid{\bf e}_i\,\rangle)\\
# &= \mid{\bf p}^f\,\rangle\sum_i c_i + \sum_i c_i\mid{\bf e}_i\,\rangle
# \end{align}
# Convergence results in a minimization of the error (causing the second term to vanish); for the DIIS solution vector $\mid{\bf p}\,\rangle$ and the true solution vector $\mid{\bf p}^f\,\rangle$ to be equal, it must be that $\sum_i c_i = 1$.  We satisfy our condition for the residual vector by minimizing its norm,
# $$\langle\,{\bf r}\mid{\bf r}\,\rangle = \sum_{ij} c_i^* c_j \langle\,{\bf r}_i\mid{\bf r}_j\,\rangle,$$
# using Lagrange's method of undetermined coefficients subject to the constraint on $\{c_i\}$:
# $${\cal L} = {\bf c}^{\dagger}{\bf Bc} - \lambda\left(1 - \sum_i c_i\right)$$
# where $B_{ij} = \langle {\bf r}_i\mid {\bf r}_j\rangle$ is the matrix of residual vector overlaps.  Minimization of the Lagrangian with respect to the coefficient $c_k$ yields (for real values)
# \begin{align}
# \frac{\partial{\cal L}}{\partial c_k} = 0 &= \sum_j c_jB_{jk} + \sum_i c_iB_{ik} - \lambda\\
# &= 2\sum_ic_iB_{ik} - \lambda
# \end{align}
# which has matrix representation
# \begin{equation}
# \begin{pmatrix}
#   B_{11} & B_{12} & \cdots & B_{1m} & -1 \\
#   B_{21} & B_{22} & \cdots & B_{2m} & -1 \\
#   \vdots  & \vdots  & \ddots & \vdots  & \vdots \\
#   B_{n1} & B_{n2} & \cdots & B_{nm} & -1 \\
#   -1 & -1 & \cdots & -1 & 0
# \end{pmatrix}
# \begin{pmatrix}
# c_1\\
# c_2\\
# \vdots \\
# c_n\\
# \lambda
# \end{pmatrix}
# =
# \begin{pmatrix}
# 0\\
# 0\\
# \vdots\\
# 0\\
# -1
# \end{pmatrix},
# \end{equation}
#
# which we will refer to as the Pulay equation, named after the inventor of DIIS.  It is worth noting at this point that our trial vectors, residual vectors, and solution vector may in fact be tensors of arbitrary rank; it is for this reason that we have used the generic notation of Dirac in the above discussion to denote the inner product between such objects.
#
# ## II. Algorithms for DIIS
# The general DIIS procedure, as described above, has the following structure during each iteration:
# #### Algorithm 1: Generic DIIS procedure
# 1. Compute new trial vector, $\mid{\bf p}_{i+1}\,\rangle$, append to list of trial vectors
# 2. Compute new residual vector, $\mid{\bf r}_{i+1}\,\rangle$, append to list of trial vectors
# 3. Check convergence criteria
#     - If RMSD of $\mid{\bf r}_{i+1}\,\rangle$ sufficiently small, and
#     - If change in DIIS solution vector $\mid{\bf p}\,\rangle$ sufficiently small, break
# 4. Build **B** matrix from previous residual vectors
# 5. Solve Pulay equation for coefficients $\{c_i\}$
# 6. Compute DIIS solution vector $\mid{\bf p}\,\rangle$
#
# For SCF iteration, the most common choice of trial vector is the Fock matrix **F**; this choice has the advantage over other potential choices (e.g., the density matrix **D**) of **F** not being idempotent, so that it may benefit from extrapolation.  The residual vector is commonly chosen to be the orbital gradient in the AO basis,
# $$g_{\mu\nu} = ({\bf FDS} - {\bf SDF})_{\mu\nu},$$
# however the better choice (which we will make in our implementation!) is to orthogonormalize the basis of the gradient with the inverse overlap metric ${\bf A} = {\bf S}^{-1/2}$:
# $$r_{\mu\nu} = ({\bf A}^{\rm T}({\bf FDS} - {\bf SDF}){\bf A})_{\mu\nu}.$$
# Therefore, the SCF-specific DIIS procedure (integrated into the SCF iteration algorithm) will be:
# #### Algorithm 2: DIIS within an SCF Iteration
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

# ## III. Implementation
#
# In order to implement DIIS, we're going to integrate it into an existing RHF program.  Since we just-so-happened to write such a program in the last tutorial, let's re-use the part of the code before the SCF integration which won't change when we include DIIS:

# +
# ==> Basic Setup <==
# Import statements
using PyCall: pyimport
psi4 = pyimport("psi4")
np   = pyimport("numpy") # used only to cast to Psi4 arrays
using TensorOperations: @tensor
using LinearAlgebra: Diagonal, Hermitian, eigen, tr, norm, dot
using Printf: @printf

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

# Maximum SCF iterations
MAXITER = 40
# Energy convergence criterion
E_conv = 1.0e-6
D_conv = 1.0e-3

# +
# ==> Static 1e- & 2e- Properties <==
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

# +
# ==> CORE Guess <==
# AO Orthogonalization Matrix
A = mints.ao_overlap()
A.power(-0.5, 1.e-16) # ≈ Julia's A^(-0.5) after psi4view()
A = np.asarray(A)

# Transformed Fock matrix
F_p = A * H * A

# Diagonalize F_p for eigenvalues & eigenvectors with Julia
e, C_p = eigen(Hermitian(F_p))

# Transform C_p back into AO basis
C = A * C_p

# Grab occupied orbitals
C_occ = C[:, 1:ndocc]

# Build density matrix from occupied orbitals
D = C_occ * C_occ'

# Nuclear Repulsion Energy
E_nuc = mol.nuclear_repulsion_energy()
# -

# Now let's put DIIS into action.  Before our iterations begin, we'll need to create empty lists to hold our previous residual vectors (AO orbital gradients) and trial vectors (previous Fock matrices), along with setting starting values for our SCF energy and previous energy:

# ==> Pre-Iteration Setup <==
# SCF & Previous Energy
SCF_E = 0.0
E_old = 0.0;

# Now we're ready to write our SCF iterations according to Algorithm 2.  Here are some hints which may help you along the way:
#
# #### Starting DIIS
# Since DIIS builds the approximate solution vector $\mid{\bf p}\,\rangle$ as a linear combination of the previous trial vectors $\{\mid{\bf p}_i\,\rangle\}$, there's no need to perform DIIS on the first SCF iteration, since there's only one trial vector for DIIS to use!
#
# #### Building **B**
# 1. The **B** matrix in the Lagrange equation is really $\tilde{\bf B} = \begin{pmatrix} {\bf B} & -1\\ -1 & 0\end{pmatrix}$.
# 2. Since **B** is the matrix of residual overlaps, it will be a square matrix of dimension equal to the number of residual vectors.  If **B** is an $N\times N$ matrix, how big is $\tilde{\bf B}$?
# 3. Since our residuals are real, **B** will be a symmetric matrix.
# 4. To build $\tilde{\bf B}$, make an empty array of the appropriate dimension, then use array indexing to set the values of the elements.
#
# #### Solving the Pulay equation
# 1. Use built-in Julia functionality to make your life easier.
# 2. The solution vector for the Pulay equation is $\tilde{\bf c} = \begin{pmatrix} {\bf c}\\ \lambda\end{pmatrix}$, where $\lambda$ is the Lagrange multiplier, and the right hand side is $\begin{pmatrix} {\bf 0}\\ -1\end{pmatrix}$.  

# +
# Start from fresh orbitals
F_p = A * H * A
e, C_p = eigen(Hermitian(F_p))
C = A * C_p
C_occ = C[:, 1:ndocc]
D = C_occ * C_occ' ;

# Trial & Residual Vector Lists
F_list = []
DIIS_RESID = []

# ==> SCF Iterations w/ DIIS <==
println("==> Starting SCF Iterations <==")
SCF_E = let SCF_E = SCF_E, E_old = E_old, D = D

   # Begin Iterations
   for scf_iter in 1:MAXITER
      # Build Fock matrix
      @tensor G[p,q] := (2I[p,q,r,s] - I[p,r,q,s]) * D[r,s]
      F = H + G
      
      # Build DIIS Residual
      diis_r = A * (F * D * S - S * D * F) * A
      
      # Append trial & residual vectors to lists
      push!(F_list, F)
      push!(DIIS_RESID, diis_r)
      
      # Compute RHF energy
      SCF_E = tr((H + F) * D) + E_nuc
      dE = SCF_E - E_old
      dRMS = norm(diis_r)
      @printf("SCF Iteration %3d: Energy = %4.16f dE = %1.5e dRMS = %1.5e \n",
                          scf_iter,        SCF_E, SCF_E - E_old,     dRMS)
      
      # SCF Converged?
      if abs(SCF_E - E_old) < E_conv && dRMS < D_conv
          break
      end
      E_old = SCF_E
      
      if scf_iter >= 2
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
         F = zeros(size(F))
         for i in 1:length(coeff) - 1
            F += coeff[i] * F_list[i]
         end
      end
      
      # Compute new orbital guess with DIIS Fock matrix
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
   SCF_E
end

# Post iterations
println("\nSCF converged.")
println("Final RHF Energy: $SCF_E [Eh]")
# -

# Congratulations! You've written your very own Restricted Hartree-Fock program with DIIS convergence accelleration!  Finally, let's check your final RHF energy against <span style='font-variant: small-caps'> Psi4</span>:

# Compare to Psi4
SCF_E_psi = psi4.energy("SCF")
psi4.compare_values(SCF_E_psi, SCF_E, 6, "SCF Energy")

# ## References
# 1. P. Pulay. *Chem. Phys. Lett.* **73**, 393-398 (1980)
# 2. C. David Sherrill. *"Some comments on accellerating convergence of iterative sequences using direct inversion of the iterative subspace (DIIS)".* Available at: vergil.chemistry.gatech.edu/notes/diis/diis.pdf. (1998)
