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

# # Density Fitting
#
# Density fitting is an extremely useful tool to reduce the computational scaling of many quantum chemical methods.  
# Density fitting works by approximating the four-index electron repulsion integral (ERI) tensors from Hartree-Fock 
# theory, $g_{\mu\nu\lambda\sigma} = (\mu\nu|\lambda\sigma)$, by
#
# $$(\mu\nu|\lambda\sigma) \approx \widetilde{(\mu\nu|P)}[J^{-1}]_{PQ}\widetilde{(Q|\lambda\sigma)}$$
#
# where the Coulomb metric $[J]_{PQ}$ and the three-index integral $\widetilde{(Q|\lambda\sigma)}$ are defined as
#
# \begin{align}
# [J]_{PQ} &= \int P(r_1)\frac{1}{r_{12}}Q(r_2){\rm d}^3r_1{\rm d}^3r_2\\
# \widetilde{(Q|\lambda\sigma)} &= \int Q(r_1)\frac{1}{r_{12}}\lambda(r_2)\sigma(r_2){\rm d}^3r_1{\rm d}^3r_2
# \end{align}
#
# To simplify the density fitting notation, the inverse Coulomb metric is typically folded into the three-index tensor:
#
# \begin{align}
# (P|\lambda\sigma) &= [J^{-\frac{1}{2}}]_{PQ}\widetilde{(Q|\lambda\sigma)}\\
# g_{\mu\nu\lambda\sigma} &\approx (\mu\nu|P)(P|\lambda\sigma)
# \end{align}
#
# These transformed three-index tensors can then be used to compute various quantities, including the four-index ERIs, 
# as well as Coulomb (J) and exchange (K) matrices, and therefore the Fock matrix (F).  Before we go any further, let's
# see how to generate these transformed tensors using <span style='font-variant: small-caps'> Psi4</span>.  
#
# First, let's import <span style='font-variant: small-caps'> Psi4</span> and set up some global options, as well as
# define a molecule and initial wavefunction:

# +
# ==> Psi4 & NumPy options, Geometry Definition <==
using PyCall: pyimport
psi4 = pyimport("psi4")
np   = pyimport("numpy") # used only to cast to Psi4 arrays
using TensorOperations: @tensor

function psi4view(psi4matrix)
   # Assumes Float64 type, C ordering
   if !hasproperty(psi4matrix,:__array_interface__)
      throw(ArgumentError("Input matrix cannot be accessed. Cannot be an rvalue"))
   end
   array_interface = psi4matrix.__array_interface__
   array_interface["data"][2] == false   || @warn "Not writable"
   array_interface["strides"] == nothing || @warn "Different ordering than C"
   array_interface["typestr"] == "<f8"   || @warn "Not little-endian Float64 eltype"
   ptr   = array_interface["data"][1]
   shape = reverse(array_interface["shape"])
   ndims = length(shape)
   permutedims(unsafe_wrap(Array{Float64,ndims}, Ptr{Float64}(ptr), shape),reverse(1:ndims))
end

# Set Psi4 memory & output options
psi4.set_memory(Int(2e9))
psi4.core.set_output_file("output.dat", false)

# Geometry specification
mol = psi4.geometry("""
O
H 1 0.96
H 1 0.96 2 104.5
symmetry c1
""")

# Psi4 options
psi4.set_options(Dict("basis"         => "cc-pvdz",
                      "scf_type"      => "df",
                      "e_convergence" => 1e-10,
                      "d_convergence" => 1e-10))

wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option("basis"))
# -

# ## Building the Auxiliary Basis Set
#
# One component of our density-fitted tensors $g_{\mu\nu\lambda\sigma} \approx (\mu\nu|P)(P|\lambda\sigma)$ which
# is unique from their exact, canonical counterparts $(\mu\nu|\lambda\sigma)$ is the additional "auxiliary" index, $P$.
# This index corresponds to inserting a resolution of the identity, which is expanded in an auxiliary basis set $\{P\}$.
# In order to build our density-fitted integrals, we first need to generate this auxiliary basis set.  Fortunately,
# we can do this with the `psi4.core.BasisSet` object:
# ~~~julia
# # Build auxiliary basis set
# aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", "aug-cc-pVDZ")
# ~~~
#
# There are special fitting basis sets that are optimal for a given orbital basis. As we will be building J and K 
# objects we want the `JKFIT` basis associated with the orbital `aug-cc-pVDZ` basis. This basis is straightfowardly 
# named `aug-cc-pVDZ-jkfit`.

# Build auxiliary basis set
aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", "aug-cc-pVDZ")

# ## Building Density-Fitted ERIs
# Now, we can use our orbital and auxiliary basis sets to construct the `Qpq` object with the inverted metric. As the 
# tensors are very similar to full ERI's we can use the same computer for both with the aid of a "zero basis". If we 
# think carefully about the $\widetilde{(Q|\lambda\sigma)}$ and $(\mu\nu|\lambda\sigma)$ we should note that on the 
# right and left hand sides the two gaussian basis functions are contracted to a single density.
#
# Specifically, for $\widetilde{(Q|\lambda\sigma)}$ the right hand side is a single basis function without being 
# multiplied by another, so we can "trick" the MintsHelper object into computing these quanties if we have a "basis 
# set" which effectively does not act on another. This is, effectively, what a "zero basis" does.
#
# The $[J^{-\frac{1}{2}}]_{PQ}$ object can be built in a similar way where we use the Psi4 Matrix's built in `power` 
# function to raise this to the $-\frac{1}{2}$ power. The call `Matrix.power(-0.5, 1.e-14)` will invert the Matrix to 
# the $-\frac{1}{2}$ while guarding against values smaller than 1.e-14. Recall that machine epsilon is ~1.e-16, when 
# these small values are taken to a negative fractional power they could become very large and dominate the resulting 
# matrix even though they are effectively noise before the inversion.
#
# ~~~julia
# orb = wfn.basisset()
# zero_bas = psi4.core.BasisSet.zero_ao_basis_set()
#
# # Build MintsHelper Instance
# mints = psi4.core.MintsHelper(orb)
#
# # Build (P|pq) raw 3-index ERIs, dimension (1, Naux, nbf, nbf)
# Ppq = mints.ao_eri(zero_bas, aux, orb, orb)
# Ppq = psi4view(Ppq)
#
# # Build and invert the metric
# metric = mints.ao_eri(zero_bas, aux, zero_bas, aux)
# metric.power(-0.5, 1.e-14)
# metric = psi4view(metric)
#
# # Remove the excess dimensions of Ppq & metric
# Ppq = dropdims(Ppq, dims=1)
# metric = dropdims(metric, dims=(1,3))
#
# # Contract Ppq & Metric to build Qso
# @tensor Qso[Q,p,q] := metric[Q,P] * Ppq[P,p,q]
# ~~~

# + code_folding=[]
# ==> Build Density-Fitted Integrals <==
# Get orbital basis & build zero basis
orb = wfn.basisset()
zero_bas = psi4.core.BasisSet.zero_ao_basis_set()

# Build instance of MintsHelper
mints = psi4.core.MintsHelper(orb)

# Build (P|pq) raw 3-index ERIs, dimension (1, Naux, nbf, nbf)
Ppq = mints.ao_eri(zero_bas, aux, orb, orb)
Ppq = psi4view(Ppq)

# Build & invert Coulomb metric, dimension (1, Naux, 1, Naux)
metric = mints.ao_eri(zero_bas, aux, zero_bas, aux)
metric.power(-0.5, 1.e-14)
metric = psi4view(metric)

# Remove excess dimensions of Ppq, & metric
Ppq = dropdims(Ppq, dims=1)
metric = dropdims(metric, dims=(1,3))

# Build the Qso object
@tensor Qpq[Q,p,q] := metric[Q,P] * Ppq[P,p,q];
# -

# ## Example: Building a Density-Fitted Fock Matrix
# Now that we've obtained our `Qpq` tensors, we may use them to build the Fock matrix.  To do so, since we aren't 
# implementing a fully density-fitted RHF program, we'll first need to get a density matrix and one-electron Hamiltonian 
# from somewhere. Let's get them from a converged HF wavefunction, so we can check our work later:

# ==> Compute SCF Wavefunction, Density Matrix, & 1-electron H <==
scf_e, scf_wfn = psi4.energy("scf", return_wfn=true)
D = np.asarray(scf_wfn.Da())
H = scf_wfn.H();

# Now that we have our density-fitted integrals and a density matrix, we can build a Fock matrix.  There are several 
# different algorithms which we can successfully use to do so; for now, we'll use a simple algorithm and `@tensor` 
# to illustrate how to perform contractions with these density fitted tensors and leave a detailed discussion of those 
# algorithms/different tensor contraction methods elsewhere.  Recall that the Fock matrix, $F$, is given by
#
# $$F = H + 2J - K,$$
#
# where $H$ is the one-electron Hamiltonian matrix, $J$ is the Coulomb matrix, and $K$ is the exchange matrix.  The 
# Coulomb and Exchange matrices have elements guven by
#
# \begin{align}
# J[D_{\lambda\sigma}]_{\mu\nu} &= (\mu\nu|\lambda\sigma)D_{\lambda\sigma}\\
# K[D_{\lambda\sigma}]_{\mu\nu} &= (\mu\lambda|\nu\sigma)D_{\lambda\sigma}.
# \end{align}
#
# When employing conventional 4-index ERI tensors, computing both $J$ and $K$ involves contracting over four unique
# indices, which involves four distinct loops -- one over each unique index in the contraction.  Therefore, the 
# scaling of this procedure is $\mathcal{O}(N^4)$, where $N$ is the number of iterations in each loop (one for each 
# basis function).  The above expressions can be coded using `@tensor` to handle the tensor contractions:
#
# ~~~julia
# @tensor J[p,q] := I[p,q,r,s] * D[r,s]
# @tensor K[p,q] := I[p,r,q,s] * D[r,s]
# ~~~
#
# for exact ERIs `I_pqrs`.  If we employ density fitting, however, we can reduce this scaling by reducing the number 
# of unique indices involved in the contractions.  Substituting in the density-fitted $(P|\lambda\sigma)$ tensors into 
# the above expressions, we obtain the following:
#
# \begin{align}
# J[D_{\lambda\sigma}]_{\mu\nu} &= (\mu\nu|P)(P|\lambda\sigma)D_{\lambda\sigma}\\
# K[D_{\lambda\sigma}]_{\mu\nu} &= (\mu\lambda|P)(P|\nu\sigma)D_{\lambda\sigma}.
# \end{align}
#
# Naively, this seems like we have actually *increased* the scaling of our algorithm, because we have added the $P$ 
# index to the expression, bringing the total to five unique indices, meaning this would scale like .  We've actually 
# made our lives easier, however: with three different tensors to contract, we can perform one contraction at a time!  
#
# For $J$, this works out to the following two-step procedure:
#
# \begin{align}
# \chi_P &= (P|\lambda\sigma)D_{\lambda\sigma} \\
# J[D_{\lambda\sigma}]_{\mu\nu} &= (\mu\nu|P)\chi_P
# \end{align}
#
#
# In the cell below, using `@tensor` and our `Qpq` tensor, try to construct `J`:

# Two-step build of J with Qpq and D
@tensor X_Q[Q] := Qpq[Q,p,q] * D[p,q]
@tensor J[p,q] := Qpq[Q,p,q] * X_Q[Q];

# Each of the above contractions, first constructing the `X_Q` intermediate and finally the full Coulomb matrix `J`, only involve three unique indices.  Therefore, the Coulomb matrix build above scales as $\mathcal{O}(N_{\rm aux}N^2)$.  Notice that we have distinguished the number of auxiliary ($N_{\rm aux}$) and orbital ($N$) basis functions; this is because auxiliary basis sets are usually around double the size of their corresponding orbital counterparts.  
#
# We can play the same intermediate trick for building the Exchange matrix $K$:
#
# \begin{align}
# \zeta_{P\nu\lambda} &= (P|\nu\sigma)D_{\lambda\sigma} \\
# K[D_{\lambda\sigma}]_{\mu\nu} &= (\mu\lambda|P)\zeta_{P\nu\lambda}
# \end{align}
#
# Just like with $J$, try building $K$ in the cell below:

# Two-step build of K with Qpq and D
@tensor Z_Qqr[Q,r,q] := Qpq[Q,r,s] * D[s,q]
@tensor K[p,r]       := Qpq[Q,p,q] * Z_Qqr[Q,r,q];


# Unfortunately, our two-step $K$ build does not incur a reduction in the overall scaling of the algorithm, with each contraction above scaling as $\mathcal{O}(N^3N_{\rm aux})$. The major benefit of density fitting for $K$ builds comes in the form of the small storage overhead of the three-index `Qpq` tensors compared to the full four-index `I_pqrs` tensors.  Even when exploiting the full eight-fold symmetry of the $(\mu\nu|\lambda\sigma)$ integrals, storing `I_pqrs` for a system with 3000 AO basis functions will require 81 TB of space, compared to a mere 216 GB to store the full `Qpq` object when exploiting the twofold symmetry of $(P|\lambda\sigma)$.  
#
# Now that we've built density-fitted versions of the $J$ and $K$ matrices, let's check our work by comparing a Fock matrix built using our $J$ and $K$ with the fully converged Fock matrix from our original SCF/aug-cc-pVDZ computation.  
#
# Below, build F using the one-electron Hamiltonian from the converged SCF wavefuntion and our $J$ and $K$ matrices.  Then, get the converged $F$ from the SCF wavefunction:

# Build F from SCF 1 e- Hamiltonian and our density-fitted J & K
F = H + 2 * J - K
# Get converged Fock matrix from converged SCF wavefunction
scf_F = scf_wfn.Fa();

# Feeling lucky? Execute the next cell to see if you've computed $J$, $K$, and $F$ correctly:

if np.allclose(F, scf_F,atol=1e-4)
    println("Nicely done!! Your density-fitted Fock matrix matches Psi4!")
else
    println("Whoops...something went wrong.  Try again!")
end

# Finally, we can remember the identity of the $D$ matrix for SCF which will be $D_{\lambda\sigma} = C_{\lambda i} C_{\sigma i}$, where $i$ is the occupied index. We can factor our $K$ build once more:
# \begin{align}
# D_{\lambda\sigma} &= C_{\lambda i} C_{\sigma i} \\
# \zeta_{P\nu i} &= (P|\nu\sigma)C_{\sigma i} \\
# K[D_{\lambda\sigma}]_{\mu\nu} &= \zeta_{P\nu i}\zeta_{P\mu i}
# \end{align}
#
# Consider the ratio between the number of basis functions and the size of the occupied index. Why would the above be beneficial?

# ## References
# 1. F. Weigend, Phys. Chem. Chem. Phys. 4, 4285 (2002).
# 2. O. Vahtras, J. Alml ̈of, and M. W. Feyereisen, Chem. Phys. Lett. 213, 514 (1993).
# 3. B. I. Dunlap, J. W. D. Connolly, and J. R. Sabin, J. Chem. Phys. 71, 3396 (1979).
# 4. J. L. Whitten, J. Chem. Phys. 58, 4496 (1973).
