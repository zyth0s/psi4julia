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

# +
"""
Tutorial: A reference implementation of orbital optimized second-order Moller-Plesset perturbation theory.
"""

__authors__   = ["D. Menendez", "Boyi Zhang"]
__credits__   = ["D. Menendez", "Boyi Zhang", "Justin M. Turney"]

__copyright_amp__ = "(c) 2014-2020, The Psi4Julia Developers"
__license__   = "BSD-3-Clause"
__date__      = "2020-08-03"
# -

# # Orbital-Optimized Second-Order Moller Plesset Perturbation Theory (OMP2)

# In this tutorial, we will implement the orbital-optimized second-order Moller-Plesset method in the spin orbital notation. The groundwork for working in the spin orbital notation has been laid out in "Introduction to the Spin Orbital Formulation of Post-HF methods" [tutorial](../08_CEPA0_and_CCD/8a_Intro_to_spin_orbital_postHF.ipynb). It is highly recommended to work through that introduction before starting this tutorial. 

# ## I. Theoretical Overview

# ### The general orbital optimization procedure
#
# In orbital optimization methods, the energy is minimized with respect to(w.r.t) an orbital rotation parameter $\textbf{X}$ and can be expanded to second-order as:
#
# \begin{equation}
# E(\textbf{X}) = E(\textbf{X}) + \textbf{X}^\dagger \textbf{w} + \frac{1}{2}\textbf{X}^\dagger\textbf{A}\textbf{X}
# \end{equation}
#
# Here, $\textbf{w}$ is the orbital gradient (derivative of E w.r.t. $\textbf{X}^\dagger$ evaluated at zero and $\textbf{A}$ is the orbital Hessian matrix (second derivative of E w.r.t. $\textbf{X}^\dagger\textbf{X}$ evaluated at zero).
#
# It can be shown that $\textbf{X} = -\textbf{A}^{-1}\textbf{w}$, which gives us the equation used in the Newton-Raphson step of the orbital optimization. 
#
# We define the unitary rotation matrix to be $\textbf{U} = exp(\textbf{X}-\textbf{X}^\dagger)$ and use this to rotate the orbitals (using the cofficient matrix). 
#
# We then transform the 1 and 2-electron integrals using the new cofficient matrix and evaluate the energy. 
#
# This process is repeated until the energy convergence satisfies a specified convergence parameter. 
#
# A detailed algorithm for OMP2 is provided in the implementation section. 
#

# ### A note on the MP2 amplitude equation
#
# The MP2 amplitude equation can be explicitly written as 
#
# \begin{equation}
#  t_{ab}^{ij} = (\mathcal{E}_{ab}^{ij})^{-1} \left(
#      \bar{g}_{ab}^{ij} + P_{(a/b)}f'{}_{a}^{c} t_{cb}^{ij} -
#      P^{(i/j)}f'{}_k^it_{ab}^{kj} \right)
# \end{equation}
#
# where f' is the off-digonal Fock matrix.
#
# Indices p, q, r... are used to indicate arbitrary orbitals, indices a, b, c... are used to indicate virtual orbitals, and indices i, j, k... are used to indicate occupied orbitals.
#
# In conventional MP2, the use canonical orbitals result in a diagonal Fock matrix and the last two terms of the t amplitude equation goes to zero. In OMP2, however, the orbitals are no longer canonical due to orbital rotation, and we have to include these terms in the equation.  
#

# ## II. Implementation

# As with previous tutorials, let's begin by importing Psi4, NumPy, TensorOperations, and LinearAlgebra and setting memory and output file options.
# Note that we will also be importing SciPy, which is another library that builds on NumPy and has additional capabilities that we will use.

# +
# ==> Import Psi4, NumPy, & TensorOperations <==
using PyCall: pyimport
psi4 = pyimport("psi4")
np   = pyimport("numpy") # used only to cast to Psi4 arrays
using TensorOperations: @tensor
import LinearAlgebra
eye(n) = LinearAlgebra.I(n)
using Formatting: printfmt


# ==> Set Basic Psi4 Options <==

# Memory specifications
psi4.set_memory(Int(2e9))
numpy_memory = 2

# Output options
psi4.core.set_output_file("output.dat", false)
# -

# We now define the molecule and set Psi4 options:

# +
mol = psi4.geometry("""
0 1
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

psi4.set_options(Dict("basis"         => "6-31g",
                      "scf_type"      => "pk",
                      "reference"     => "rhf",
                      "mp2_type"      => "conv",
                      "e_convergence" => 1e-8,
                      "d_convergence" => 1e-8))
# -

# We use Psi4 to compute the RHF energy and wavefunction and store them in variables `scf_e` and `scf_wfn`. We also check the memory requirements for computation:

# +
# Get the SCF wavefunction & energies
scf_e, scf_wfn = psi4.energy("scf", return_wfn=true)

# ==> Nuclear Repulsion Energy <==
E_nuc = mol.nuclear_repulsion_energy()

# Check memory requirements
nmo = scf_wfn.nmo()
I_size = nmo^4 * 8e-9
println("\nSize of the ERI tensor will be $I_size GB.")
memory_footprint = I_size * 1.5
if I_size > numpy_memory
    psi4.core.clean()
    throw(OutOfMemoryError("Estimated memory utilization ($memory_footprint GB) exceeds " * 
                           "allotted memory limit of $numpy_memory GB."))
end
# -

# We need to set the maximum number of iterations for the OMP2 code as well as the energy convergence criteria:

# ==> Set default program options <==
# Maximum OMP2 iterations
MAXITER = 40
# Energy convergence criterion
E_conv = 1.0e-8

# We first obtain orbital information from our wavefunction. We also create an instance of MintsHelper to help build our molecular integrals:

# +
# Create instance of MintsHelper class
mints = psi4.core.MintsHelper(scf_wfn.basisset())

# Get basis and orbital information
nbf = mints.nbf()          # Number of basis functions
nalpha = scf_wfn.nalpha()  # Number of alpha electrons
nbeta = scf_wfn.nbeta()    # Number of beta electrons
nocc = nalpha + nbeta      # Total number of electrons
nso = 2nbf                 # Total number of spin orbitals
nvirt = nso - nocc         # Number of virtual orbitals
# -

# We now build our 2-electron integral, a 4D tensor, in the spin orbital formulation. We also convert it into physicist's notation and antisymmetrize for easier manipulation of the tensor later on. 

# +
"""
Spin blocks 2-electron integrals
Using np.kron, we project I and I tranpose into the space of the 2x2 ide
The result is our 2-electron integral tensor in spin orbital notation
"""
function spin_block_tei(I)
    identity = eye(2)
    I = np.kron(identity, I)
    np.kron(identity, permutedims(I, reverse(1:4)))
end
 
I = np.asarray(mints.ao_eri())
I_spinblock = spin_block_tei(I)
 
# Convert chemist's notation to physicist's notation, and antisymmetrize
# (pq|rs) ---> ⟨pr|qs⟩
# ⟨pr||qs⟩ = ⟨pr|qs⟩ - ⟨pr|sq⟩
gao = permutedims(I_spinblock, (1, 3, 2, 4)) - permutedims(I_spinblock, (1, 3, 4, 2));
# -

# We get the core Hamiltonian from the reference wavefunction and build it in the spin orbital formulation. The Julia function `kron` is used to project the core Hamiltonian into the space of a 2x2 identity matrix. Note that `np.kron` was used for spin-blocking the 2-electron integral. In the current case, `kron` is only called once because the core Hamltonian is a 2D matrix. 

# +
# ==> core Hamiltoniam <==

h = np.asarray(scf_wfn.H())

# Using np.kron, we project h into the space of the 2x2 identity
# The result is the core Hamiltonian in the spin orbital formulation
hao = kron(eye(2), h);
# -

# We get the orbital energies from alpha and beta electrons and append them together. We spin-block the coefficients obtained from the reference wavefunction and convert them into Julia arrays. There is a set corresponding to coefficients from alpha electrons and a set of coefficients from beta electrons. We then sort them according to the order of the orbital energies using `sortperm()`:

# +
# Get orbital energies, cast into Julia array, and extend eigenvalues
eps_a = np.asarray(scf_wfn.epsilon_a())
eps_b = np.asarray(scf_wfn.epsilon_b())
eps = vcat(eps_a, eps_b)

# Get coefficients, block, and sort
Ca = np.asarray(scf_wfn.Ca())
Cb = np.asarray(scf_wfn.Cb())
C = [Ca zero(Ca); zero(Cb) Cb]; # direct sum

# Sort the columns of C according to the order of orbital energies
C = C[:, sortperm(eps)]; 
# -

# We now define two functions that will transform the core Hamiltonian and the 2-electron integral from the AO basis into the MO basis using the coefficients:
#
# \begin{align}
# h_p^q &= \sum_{\mu \nu} C_{\mu p}^* h_{\mu \nu} C_{\nu q} \\
# \bar{g}_{pq}^{rs} &= \sum_{\mu \nu \rho \sigma} 
#                 C_{\mu p}^* C_{\nu q}^* \langle \mu \nu || \rho \sigma \rangle C_{\rho r}C_{\sigma s}
# \end{align}
#
# Note that we transform the core Hamiltonian twice because it has two dimensions. We use these functions to transform the `hao` and `gao` previously defined:

# +
# ==> AO to MO transformation functions <==


"""
Transform hao, which is the core Hamiltonian in the spin orbital basis,
into the MO basis using MO coefficients
"""
function ao_to_mo(hao, C)
    
    @tensor begin
       hmo[P,Q] := hao[p,Q] * C[p,P]
       hmo[p,Q] := hmo[p,q] * C[q,Q]
    end
end


"""
Transform gao, which is the spin-blocked 4d array of physicist's notation,
antisymmetric two-electron integrals, into the MO basis using MO coefficients
"""
function ao_to_mo_tei(gao, C)
    
   @tensor begin
      gmo[P,Q,R,S] := gao[p,Q,R,S] * C[p,P]
      gmo[p,Q,R,S] := gmo[p,q,R,S] * C[q,Q]
      gmo[p,q,R,S] := gmo[p,q,r,S] * C[r,R]
      gmo[p,q,r,S] := gmo[p,q,r,s] * C[s,S]
   end
end

# Transform gao and hao into MO basis
hmo = ao_to_mo(hao, C)
gmo = ao_to_mo_tei(gao, C);
# -

# Here we define slices corresponding to the number and position of occupied and virtual indices. We will use these later in the code to access occupied and virtual blocks of relevant arrays. For example, to get $\bar{g}_{ab}^{ij}$, we call:
# ~~~julia
# gmo[v, v, o, o]
# ~~~

# Make slices
x = [CartesianIndex()]
o = [p ≤ nocc for p in 1:nso]
v = [p > nocc for p in 1:nso];

# **OMP2 iteration algorithm:**
#
# 1. Build the fock matrix
#
#    \begin{equation}
#    f_p^q = h_p^q +\bar{g}_{pi}^{qi} 
#    \end{equation}   
#    
# 2. Build the off-diagonal Fock matrix and the orbital energies, where off-diagonal Fock matrix(`fprime`) is just the Fock matrix with its diagonal elements set to zero, and the orbital energies (`eps`) are just the diagonal elements of the Fock matrix
#
#     \begin{equation}
#     \epsilon_p = f_p^p
#     \end{equation}
#
#     \begin{equation}
#     f'{}_p^q =(1 - \delta_p^q)f_p^q
#     \end{equation}
#
# 3. Update the amplitudes (`t_amp`)
#
#     \begin{equation}
#      t_{ab}^{ij} = (\mathcal{E}_{ab}^{ij})^{-1} \left(
#      \bar{g}_{ab}^{ij} + P_{(a/b)}f'{}_{a}^{c} t_{cb}^{ij} -
#      P^{(i/j)}f'{}_k^it_{ab}^{kj} \right)
#     \end{equation}
#
#    Here, P is a permutation operator that permutes the indices indicated. For example, $P_{(a/b)}$ would give all    possible permutations of a and b. Thus, 
#     
#     \begin{equation}
#     P_{(a/b)}f'{}_{a}^{c} t_{cb}^{ij} = f'{}_{a}^{c} t_{cb}^{ij} - f'{}_{b}^{c} t_{ca}^{ij}
#     \end{equation}   
#    
#    where the minus sign arises as a result of antisymmetric properties due to the interchange of the two indices
#    The amplitudes terms in the code are assigned as `t1`, `t2`, and `t3`, respectively.
#    
#    To take in account the permutation terms, we evaluate the term and then transpose the relevant indices. 
#    For example, for the second term in the amplitude equation we first evaluate it as it:
#    ~~~julia
#    @tensor t2[a,b,i,j] := (fprime[v, v])[a,c] * t_amp[c,b,i,j]
#    ~~~
#    Then, to account for the permutation, we transpose the two dimensions corresponding to the permuted indices. Since    a and b are in the first two dimensions of `t2`, we switch 0 and 1: 
#    ~~~julia
#    t2 = t2 - permutedims(t2, (2, 1, 3, 4))
#    ~~~
# 4. Build the one-particle density matrix (`opdm`)
#
#     \begin{equation}
#     \gamma_q^p = \tilde{\gamma}_q^p + \mathring{\gamma}_q^p
#     \end{equation}
#
#    The one-particle density matrix(opdm) is a sum of the reference opdm ($\mathring{\gamma}_q^p$) and a correlation opdm ($\tilde{\gamma}_q^p$).
#     
#     $\mathring{\gamma}_q^p$ is assigned as the variable `odm_ref` and defined as:
#      \begin{align}
#      & \, \delta^i_j \, \text{for $p=i$, $q=j$}, \\
#      & 0 \,  \text{otherwise}  
#     \end{align}
#
#     The virtual block of $\tilde{\gamma}_q^p$ (assigned as `odm_corr`) is defined as:
#     \begin{equation}
#     \tilde{\gamma}_b^a  = \frac{1}{2} t_{ij}^{ab*}t_{bc}^{ij}
#     \end{equation}
#
#      The occupied block of $\tilde{\gamma}_q^p$ is defined as:
#     \begin{equation}
#     \tilde{\gamma}_j^i  = -\frac{1}{2} t_{jk}^{ab*}t_{ab}^{ik}
#     \end{equation}
#
#     As seen before, we used our defined slices to pick out these specific blocks: 
#     ~~~julia 
#     @tensor (opdm_corr[v, v])[b,a] :=  0.5(permutedims(t_amp, reverse(1:4)))[i,j,a,c] * t_amp[b,c,i,j]
#     @tensor (opdm_corr[o, o])[j,i] := -0.5(permutedims(t_amp, reverse(1:4)))[j,k,a,b] * t_amp[a,b,i,k]
#     ~~~
#     
# 5. Build the two-particle density matrix (`tpdm`)  
#
#     \begin{equation}
#     \Gamma_{rs}^{pq} = \tilde{\Gamma}_{rs}^{pq} + P_{(r/s)}^{(p/q)}\tilde{\gamma}_r^p\mathring{\gamma}_s^q 
#     +P_{(r/s)}\mathring{\gamma}_r^p\mathring{\gamma}_s^q
#     \end{equation}
#     
#      where as before, P is the permutation operator
#  
#  $\tilde{\Gamma}_{rs}^{pq}$ (`tdm_corr`) can be separated into two components: 
#  
#  \begin{align}
#  \tilde{\Gamma}_{ij}^{ab} = & t_{ij}^{ab*}\\
#  \tilde{\Gamma}_{ab}^{ij} = & t_{ab}^{ij}
#  \end{align}
#  
# 6. Compute the Newton-Raphson step 
#
#    First, form a generalized-Fock matrix using the one and two particle density matrices. This will be used to form the MO gradient matrix needed for the rotation matrix:
#    
#    \begin{equation}
#    (\textbf{F})_p^q \equiv h_p^r \gamma_r^q + \frac{1}{2} \bar{g}_{pr}^{st}\Gamma_{st}^{qr}
#    \end{equation}
#    
#    We have seen in the theoretical overview that the X matrix while paramtetrizes the orbital rotations can be expressed in terms of the orbital gradient matrix and orbital Hessian matrix. It can be shown that the individual elements of X can be computed by:
#    
#    \begin{equation}
#     x_a^i = \frac{(\textbf{F} - \textbf{F}^\dagger)_a^i}{\epsilon_i - \epsilon_a}
#     \end{equation}
#     
#      Here we only consider rotations between the occupied and virtual orbitals, since rotations within each block are redudant since energy is invariant to rotations within those spaces. 
#      
#      Rather than computing individual elements we can compute the whole virtual-occupied block:
#      
#     \begin{equation}
#     \textbf{X}_v^o = (\textbf{F} - \textbf{F}^\dagger)_v^o (\mathcal{E}_v^o)^{-1}
#     \end{equation}
#     Translating this to code, this becomes:
#     ~~~julia
#     X[v, o] = ((F - F')[v, o]) ./ (-eps[v, x] .+ eps[x, o])
#     ~~~
# 7. We can now build the Newton-Raphson orbital rotation matrix from $\textbf{X}$:
#
#     \begin{equation}
#     \textbf{U} = exp(\textbf{X} - \textbf{X}^\dagger)
#     \end{equation}
#     
# 8. Use the rotation matrix to rotate the MO coefficients
#    \begin{equation}
#    \textbf{C} \leftarrow \textbf{CU}
#    \end{equation}
#    
# 9. Transform the 1-electron (`hmo`) and 2-electron (`gmo`) integrals to the MO basis using the new coefficient matrix. We can use our previously defined transformation functions for this step.
#
#     \begin{align}
#     h_p^q &= \sum_{\mu \nu} C_{\mu p}^* h_{\mu \nu} C_{\nu q} \\
#     \bar{g}_{pq}^{rs} &= \sum_{\mu \nu \rho \sigma} 
#     C_{\mu p}^* C_{\nu q}^* \langle \mu \nu || \rho \sigma \rangle C_{\rho r}C_{\sigma s}
#     \end{align}
# 10. Evaluate the energy (`E_OMP2`)
#     \begin{equation}
#     E = h_p^q \gamma_q^p + \frac{1}{4} \bar{g}_{pq}^{rs}\Gamma_{rs}^{pq}
#     \end{equation}
#
# 11. If the energy is converged according to the convergence criterion defined above, quit. Otherwise, loop over the algorithm again. 

# Before beginning the iterations, we initialize OMP2 energy and the t amplitudes $t_{ab}^{ij}$ (`t_amp`) to be zero. We also initialize the correlation and reference one-particle density matrix and the correlation two-particle density matrix. Finally we intialize `X`, which is the parameter used to optimize our orbitals in the Newton-Raphson step. 
#

# +
# Intialize t amplitude and energy 
t_amp = zeros(nvirt, nvirt, nocc, nocc)
E_OMP2_old = 0.0 

# Initialize the correlation one particle density matrix
opdm_corr = zeros(nso, nso)

# Build the reference one particle density matrix
opdm_ref = zeros(nso, nso)
opdm_ref[o, o] = eye(nocc)

# Initialize two particle density matrix
tpdm_corr = zeros(nso, nso, nso, nso)

# Initialize the rotation matrix parameter 
E_OMP2 = let hmo=hmo, gmo=gmo, tpdm_corr=tpdm_corr, opdm_corr=opdm_corr, opdm_ref=opdm_ref,
   E_OMP2_old = E_OMP2_old, t_amp=t_amp, C=C

   E_OMP2 = 0.0
   X = zeros(nso, nso)

   for iteration in 1:MAXITER

       # Build the Fock matrix
       @tensor f[p,q] := hmo[p,q] + (gmo[:, o, :, o])[p,i,q,i]

       # Build off-diagonal Fock Matrix and orbital energies
       fprime = copy(f)
       fprime[LinearAlgebra.diagind(fprime)] .= 0
       eps = LinearAlgebra.diag(f)

       # Update t amplitudes
       t1 = @view gmo[v, v, o, o]
       @tensor t2[a,b,i,j] := (fprime[v, v])[a,c] * t_amp[c,b,i,j]
       @tensor t3[a,b,i,j] := (fprime[o, o])[k,i] * t_amp[a,b,k,j]
       t_amp = t1 .+ t2 .- permutedims(t2, (2, 1, 3, 4)) .-
               t3 .+ permutedims(t3, (1, 2, 4, 3))
       
       # Divide by a 4D tensor of orbital energies
       @. t_amp /= (- eps[v, x, x, x] - eps[x, v, x, x] +
                      eps[x, x, o, x] + eps[x, x, x, o])
      
       # Build one particle density matrix
       @tensor (opdm_corr[v, v])[b,a] :=  0.5(permutedims(t_amp, reverse(1:4)))[i,j,a,c] * t_amp[b,c,i,j]
       @tensor (opdm_corr[o, o])[j,i] := -0.5(permutedims(t_amp, reverse(1:4)))[j,k,a,b] * t_amp[a,b,i,k]
       opdm = opdm_corr + opdm_ref 

       # Build two particle density matrix
       tpdm_corr[v, v, o, o] = t_amp
       tpdm_corr[o, o, v, v] = permutedims(t_amp, reverse(1:4))
       @tensor tpdm2[r,s,p,q] := opdm_corr[r,p] * opdm_ref[s,q]
       @tensor tpdm3[r,s,p,q] := opdm_ref[r,p] * opdm_ref[s,q]
       tpdm = tpdm_corr +
           tpdm2 - permutedims(tpdm2, (2, 1, 3, 4)) -
           permutedims(tpdm2, (1, 2, 4, 3)) + permutedims(tpdm2, (2, 1, 4, 3)) +
           tpdm3 - permutedims(tpdm3, (2, 1, 3, 4))

       # Newton-Raphson step
       @tensor F[p,q] := hmo[p,r] * opdm[r,q] + 0.5gmo[p,r,s,t] * tpdm[s,t,q,r]
       X[v, o] = ((F - F')[v, o]) ./ (- eps[v, x] .+ eps[x, o])

       # Build Newton-Raphson orbital rotation matrix
       U = exp(X - X')

       # Rotate spin-orbital coefficients
       C = C * U

       # Transform one and two electron integrals using new C
       hmo = ao_to_mo(hao, C)
       gmo = ao_to_mo_tei(gao, C)

       # Compute the energy
       E_OMP2 = E_nuc + @tensor scalar(hmo[p,q] * opdm[q,p]) +
                  1/4 * @tensor scalar(gmo[p,q,r,s] * tpdm[r,s,p,q])
       printfmt("OMP2 iteration: {1:3d} Energy: {2:15.8f} dE: {3:2.5e}\n", iteration, E_OMP2, E_OMP2-E_OMP2_old)

       abs(E_OMP2-E_OMP2_old) < E_conv && break

       # Updating values
       E_OMP2_old = E_OMP2
   end
   E_OMP2
end
# -

# We compare the final energy with Psi4's OMP2 energy:

psi4.compare_values(psi4.energy("omp2"), E_OMP2, 6, "OMP2 Energy")

# ## References
#
# 1. Background paper:
#     >"Quadratically convergent algorithm for orbital optimization in the orbital-optimized
# coupled-cluster doubles method and in orbital-optimized second-order Møller-Plesset
# perturbation theory"[[Bozkaya:2011:135](http://aip.scitation.org/doi/10.1063/1.3631129)] U. Bozkaya, J. M. Turney, Y. Yamaguchi, H. F. Schaefer III, and C. D. Sherrill, *J. Chem. Phys.* **135**, 104103 (2011).
#
# 2. Useful notes on orbital rotation: 
# 	> A. V. Copan, "Orbital Relaxation" accessed with https://github.com/CCQC/chem-8950/tree/master/2017/.
#     
# 3. Algorithms from: 
# 	> A. V. Copan, "OMP2" accessed with https://github.com/CCQC/chem-8950/tree/master/2017/programming.
