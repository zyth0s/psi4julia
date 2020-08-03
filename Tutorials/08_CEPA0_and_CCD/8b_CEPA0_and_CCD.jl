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
"""Tutorial: CEPA0 and CCD"""

__author__    = ["D. Menendez", "Adam S. Abbott"]
__credit__    = ["D. Menendez", "Adam S. Abbott", "Justin M. Turney"]

__copyright__ = "(c) 2014-2020, The Psi4Julia Developers"
__license__   = "BSD-3-Clause"
__date__      = "2020-08-02"
# -

# # Introduction
# In this tutorial, we will implement the coupled-electron pair approximation (CEPA0) and coupled-cluster doubles (CCD) methods using our spin orbital framework covered in the [previous tutorial](8a_Intro_to_spin_orbital_postHF.ipynb).
#
#
# ### I. Coupled Cluster Theory
#
# In single reference coupled cluster theory, dynamic correlation is acquired by operating an exponential operator on some reference determinant, such as a Hartree-Fock wavefunction, to obtain the coupled cluster wavefunction given by:
#
# \begin{equation}
# \mid \mathrm{\Psi_{CC}} \rangle = \exp(\hat{T}) \mid \mathrm{\Phi} \rangle 
# \end{equation}
#
# where $\hat{T} = T_1 + T_2 + ... + T_n$ is the sum of "cluster operators" which act on our reference wavefunction to excite electrons from occupied ($i, j, k$...) to virtual ($a, b, c$...) orbitals. In second quantization, these cluster operators are expressed as:
#
# \begin{equation}
# T_k = \left(\frac{1}{k!}\right)^2 \sum_{\substack{i_1 \ldots i_k \\ a_1 \ldots a_k }} t_{i_1 \ldots i_k}^{a_1 \ldots a_k}   a_{a_1}^{\dagger} \ldots a_{a_k}^{\dagger} a_{i_k} \ldots a_{i_1}
# \end{equation}
#
# where $t$ is the $t$-amplitude, and $a^{\dagger}$ and $a$ are creation and annihilation operators.
#
# ### II. Coupled Cluster Doubles
# For CCD, we only include the doubles cluster operator:
#
# \begin{equation}
# \mid \mathrm{\Psi_{CCD}} \rangle = \exp(T_2) \mid \mathrm{\Phi} \rangle
# \end{equation}
#
# The CCD Schr&ouml;dinger equation is
#
# \begin{equation}
# \hat{H} \mid \mathrm{\Psi_{CCD}} \rangle = E \mid \mathrm{\Psi_{CCD}}\rangle
# \end{equation}
#
# The details will not be covered here, but if we project the CCD Schr&ouml;dinger equation on the left by our Hartree-Fock reference determinant $ \langle \mathrm{\Phi}\mid $, assuming intermediate normalization $\langle \Phi \mid \mathrm{\Psi_{CCD}} \rangle = 1$, we obtain:
#
# \begin{equation}
#  \langle \Phi \mid \hat{H} \space \exp(T_2) \mid \Phi \rangle = E
# \end{equation}
#
# which is most easily evaluated with a diagrammatic application of Wick's theorem. Assuming Brillouin's theorem applies (that is, our reference is a Hartree-Fock wavefunction) we obtain:
#
# \begin{equation}
# E_{\mathrm{CCD}} = \tfrac{1}{4} \bar{g}_{ij}^{ab} t_{ab}^{ij}
# \end{equation}
#
# A somewhat more involved derivation is that of the $t$-amplitudes. These are obtained in a similar fashion to the energy expression, this time projecting the CCD Schr&ouml;dinger equation on the left by a doubly-excited reference determinant $ \langle\Phi_{ij}^{ab}\mid $:
#
# \begin{equation}
# \langle\Phi_{ij}^{ab}\mid \hat{H} \space \exp(T_2) \mid \Phi \rangle
# \end{equation}
#
# I will spare you the details of solving this expectation value as well. But, if one evaluates the diagrams via Wick's theorem and simplifies, the $t$-amplitudes are given by:
#
# \begin{equation}
# t_{ab}^{ij} = (\mathcal{E}_{ab}^{ij})^{-1} \left( \bar{g}_{ab}^{ij} + \tfrac{1}{2} \bar{g}_{ab}^{cd} t_{cd}^{ij} + \tfrac{1}{2} \bar{g}_{kl}^{ij} t_{ab}^{kl}  + \hat{P}_{(a \space / \space b)}^{(i \space / \space j)} \bar{g}_{ak}^{ic} t_{bc}^{jk} - \tfrac{1}{2}\hat{P}_{(a \space / \space b)} \bar{g}_{kl}^{cd} t_{ac}^{ij} t_{bd}^{kl} - \tfrac{1}{2}  \hat{P}^{(i \space / \space j)} \bar{g}_{kl}^{cd} t_{ab}^{ik} t_{cd}^{jl} + \tfrac{1}{4} \bar{g}_{kl}^{cd} t_{cd}^{ij} t_{ab}^{kl} +  \hat{P}^{(i \space / \space j)} \bar{g}_{kl}^{cd} t_{ac}^{ik} t_{bd}^{jl} \right)
# \end{equation}
#
# where $(\mathcal{E}_{ab}^{ij})^{-1}$ is the orbital energy denominator, more familiarly known as
#
# \begin{equation}
# (\mathcal{E}_{ab}^{ij})^{-1} = \frac{1}{\epsilon_i + \epsilon_j - \epsilon_a - \epsilon_b}
# \end{equation}
#
# and $\bar{g}_{pq}^{rs}$ is the antisymmetrized two-electron integral in physicist's notation $\langle pq \mid\mid rs \rangle$. $\hat{P}$ is the *antisymmetric permutation operator*. This operator acts on a term to produce the sum of the permutations of the indicated indices, with an appropriate sign factor. Its effect is best illustrated by an example. Consider the fourth term, which is really four terms in one. 
#
# $\hat{P}_{(a \space / \space b)}^{(i \space / \space j)} \bar{g}_{ak}^{ic} t_{bc}^{jk}$ produces: 
#
# 1. The original: $ \quad \bar{g}_{ak}^{ic} t_{bc}^{jk} \\ $
#
# 2. Permuation of $a$ and $b$: $ \quad  \textrm{-} \bar{g}_{bk}^{ic} t_{ac}^{jk} \\ $
#
# 3. Permuation of $i$ and $j$: $ \quad \, \, \textrm{-} \bar{g}_{ak}^{jc} t_{bc}^{ik} \\ $
#
# 4. Permuation of $a$ and $b$, $i$ and $j$: $ \quad \bar{g}_{bk}^{jc} t_{ac}^{ik} \\ $
#
#
# Note that each permutation adds a sign change. This shorthand notation keeps the equation in a more manageable form. 
#
# Since the $t$-amplitudes and the energy depend on $t$-amplitudes, we must iteratively solve these equations until they reach self consistency, and the energy converges to some threshold.
#
# ### III. Retrieving MP2 and CEPA0 from the CCD equations
# It is interesting to note that if we only consider the first term of the expression for the doubles amplitude $t_{ab}^{ij}$ and plug it into the energy expression, we obtain the MP2 energy expression:
#
# \begin{equation}
# t_{ab}^{ij} = (\mathcal{E}_{ab}^{ij})^{-1} \bar{g}_{ab}^{ij} 
# \end{equation}
#
# \begin{equation}
# E_{\mathrm{MP2}} = \tfrac{1}{4}  \bar{g}_{ij}^{ab} t_{ab}^{ij}  = \tfrac{1}{4} \bar{g}_{ij}^{ab} \bar{g}_{ab}^{ij}   (\mathcal{E}_{ab}^{ij})^{-1}
# \end{equation}
#
# Furthermore, if we leave out the quadratic terms in the CCD amplitude equation (terms containing two $t$-amplitudes), we obtain the coupled electron-pair approximation (CEPA0):
# \begin{equation}
# t_{ab}^{ij} = (\mathcal{E}_{ab}^{ij})^{-1} \left( \bar{g}_{ab}^{ij} + \tfrac{1}{2} \bar{g}_{ab}^{cd} t_{cd}^{ij} + \tfrac{1}{2} \bar{g}_{kl}^{ij} t_{ab}^{kl}  + \hat{P}_{(a \space / \space b)}^{(i \space / \space j)} \bar{g}_{ak}^{ic} t_{bc}^{jk} \right)
# \end{equation}
#
# The CEPA0 energy expression is identical:
#
# \begin{equation}
# E_{\mathrm{CEPA0}} = \tfrac{1}{4} \bar{g}_{ij}^{ab} t_{ab}^{ij}
# \end{equation}
#
# Using our spin orbital setup for the MO coefficients, orbital energies, and two-electron integrals used in the [previous tutorial](8a_Intro_to_spin_orbital_postHF.ipynb), we are equipped to program the expressions for the CEPA0 and CCD correlation energy.

# ### Implementation: CEPA0 and CCD
# As usual, we import Psi4, NumPy, and TensorOperations, and set the appropriate options. 

# +
# ==> Import statements & Global Options <==
using PyCall: pyimport
psi4 = pyimport("psi4")
np   = pyimport("numpy")
using TensorOperations: @tensor
using Formatting: printfmt

psi4.set_memory(Int(2e9))
numpy_memory = 2
psi4.core.set_output_file("output.dat", false)

# +
# ==> Molecule & Psi4 Options Definitions <==
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

# Note that since we are using a spin orbital setup, we are free to use any Hartree-Fock reference we want. Here we choose RHF. For convenience, we let Psi4 take care of the Hartree-Fock procedure, and return the wavefunction object.

# Get the SCF wavefunction & energies
scf_e, scf_wfn = psi4.energy("scf", return_wfn=true)

# Load in information about the basis set and orbitals using MintsHelper and the wavefunction:

mints = psi4.core.MintsHelper(scf_wfn.basisset())
nbf = mints.nbf()           # number of basis functions
nso = 2nbf                  # number of spin orbitals
nalpha = scf_wfn.nalpha()   # number of alpha electrons
nbeta = scf_wfn.nbeta()     # number of beta electrons
nocc = nalpha + nbeta       # number of occupied orbitals
nvirt = 2nbf - nocc         # number of virtual orbitals

# Spin-block our MO coefficients and two-electron integrals, just like in the spin orbital MP2 code:

# +
Ca = np.asarray(scf_wfn.Ca())
Cb = np.asarray(scf_wfn.Cb())
C = [Ca zero(Ca); zero(Cb) Cb]; # direct sum

# Result: | Ca  0 |
#         | 0   Cb|

# +
# Get the two electron integrals using MintsHelper
I = np.asarray(mints.ao_eri())

"""  
Function that spin blocks two-electron integrals
Using `np.kron`, we project I into the space of the 2x2 identity, tranpose the result
and project into the space of the 2x2 identity again. This doubles the size of each axis.
The result is our two electron integral tensor in the spin orbital form.
"""
function spin_block_tei(I)
    identity = [ 1.0 0.0; 0.0 1.0]
    I = np.kron(identity, I)
    np.kron(identity, permutedims(I, reverse(1:4)))
end

# Spin-block the two electron integral array
I_spinblock = spin_block_tei(I);
# -

# Convert two-electron integrals to antisymmetrized physicist's notation:

# Converts chemist's notation to physicist's notation, and antisymmetrize
# (pq|rs) ↦ ⟨pr|qs⟩
# Physicist's notation
tmp = permutedims(I_spinblock, (1, 3, 2, 4))
# Antisymmetrize:
# ⟨pr||qs⟩ = ⟨pr|qs⟩ - ⟨pr|sq⟩
gao = tmp - permutedims(tmp, (1, 2, 4, 3));

# Obtain the orbital energies, append them, and sort the columns of our MO coefficient matrix according to the increasing order of orbital energies. 

# +
# Get orbital energies 
eps_a = np.asarray(scf_wfn.epsilon_a())
eps_b = np.asarray(scf_wfn.epsilon_b())
eps = vcat(eps_a, eps_b)

# Before sorting the orbital energies, we can use their current arrangement to sort the columns
# of C. Currently, each element i of eps corresponds to the column i of C, but we want both
# eps and columns of C to be in increasing order of orbital energies

# Sort the columns of C according to the order of increasing orbital energies 
C = C[:, sortperm(eps)] 

# Sort orbital energies in increasing order
sort!(eps);
# -

# Finally, we transform our two-electron integrals to the MO basis. Here, we denote the integrals as `gmo` to differentiate from the chemist's notation integrals `I_mo`.

# Transform gao, which is the spin-blocked 4d array of physicist's notation, 
# antisymmetric two-electron integrals, into the MO basis using MO coefficients 
gmo = @tensor begin
   gmo[P,Q,R,S] := gao[p,Q,R,S] * C[p,P]
   gmo[p,Q,R,S] := gmo[p,q,R,S] * C[q,Q]
   gmo[p,q,R,S] := gmo[p,q,r,S] * C[r,R]
   gmo[p,q,r,S] := gmo[p,q,r,s] * C[s,S]
end
nothing

# Construct the 4-dimensional array of orbital energy denominators:

# Define slices, create 4 dimensional orbital energy denominator tensor
n = [CartesianIndex()]
o = [p ≤ nocc for p in 1:nso]
v = [p > nocc for p in 1:nso]
e_denom = @. inv(-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o]);

# We now have everything we need to construct our $t$-amplitudes and iteratively solve for our CEPA0 and CCD energy. To build the $t$-amplitudes, we first construct an empty 4-dimensional array to store them. 

# Create space to store t amplitudes
t_amp = zeros(nvirt, nvirt, nocc, nocc);

# # Implementation: CEPA0
# First we will program CEPA0. Recall the expression for the $t$-amplitudes:
#
# \begin{equation}
# t_{ab}^{ij} = (\mathcal{E}_{ab}^{ij})^{-1} \left( \bar{g}_{ab}^{ij} + \tfrac{1}{2} \bar{g}_{ab}^{cd} t_{cd}^{ij} + \tfrac{1}{2} \bar{g}_{kl}^{ij} t_{ab}^{kl}  + \hat{P}_{(a \space / \space b)}^{(i \space / \space j)} \bar{g}_{ak}^{ic} t_{bc}^{jk} \right)
# \end{equation}
#
# These terms translate naturally into code using Julia's `@tensor` function. To access only the occupied and virtual indices of `gmo` we use our slices defined above. The permutation operator terms can be easily obtained by transposing the original result accordingly. To construct each iteration's $t$-amplitude:  
#
# ~~~julia
# mp2    = @view gmo[v, v, o, o]
# @tensor cepa1[ a,b,i,j] := 0.5(gmo[v,v,v,v])[a,b,c,d] * t_amp[c,d,i,j]
# @tensor cepa2[ a,b,i,j] := 0.5(gmo[o,o,o,o])[k,l,i,j] * t_amp[a,b,k,l]
# @tensor cepa3a[a,b,i,j] :=    (gmo[v,o,o,v])[a,k,i,c] * t_amp[b,c,j,k]
# cepa3b = -permutedims(cepa3a, (2, 1, 3, 4)) # a <-> b
# cepa3c = -permutedims(cepa3a, (1, 2, 4, 3)) #          i <-> j
# cepa3d =  permutedims(cepa3a, (2, 1, 4, 3)) # a <-> b, i <-> j
# cepa3  =  cepa3a + cepa3b + cepa3c + cepa3d
#
# t_amp_new = @. e_denom * (mp2 + cepa1 + cepa2 + cepa3)
# ~~~
#
# To evaluate the energy, $E_{\mathrm{CEPA0}} = \tfrac{1}{4} \bar{g}_{ij}^{ab} t_{ab}^{ij}$,
#
# ~~~julia
# E_CEPA0 = 1/4 * @tensor scalar((gmo[o,o,v,v])[i,j,a,b] * t_amp_new[a,b,i,j])
# ~~~

# Putting it all together, we initialize the energy, set the max iterations, and iterate the energy until it converges to our convergence criterion:

# +
# Initialize energy
E_CEPA0 = let E_CEPA0 = 0.0, gmo=gmo, o=o,v=v, e_denom = e_denom, t_amp = t_amp

   MAXITER = 50

   for cc_iter in 1:MAXITER
       E_old = E_CEPA0
       
       # Collect terms
       mp2      = @view gmo[v,v,o,o]
       @tensor cepa1[ a,b,i,j] := 0.5(gmo[v,v,v,v])[a,b,c,d] * t_amp[c,d,i,j]
       @tensor cepa2[ a,b,i,j] := 0.5(gmo[o,o,o,o])[k,l,i,j] * t_amp[a,b,k,l]
       @tensor cepa3a[a,b,i,j] :=    (gmo[v,o,o,v])[a,k,i,c] * t_amp[b,c,j,k]
       cepa3b = -permutedims(cepa3a, (2, 1, 3, 4))
       cepa3c = -permutedims(cepa3a, (1, 2, 4, 3))
       cepa3d =  permutedims(cepa3a, (2, 1, 4, 3))
       cepa3  =  cepa3a + cepa3b + cepa3c + cepa3d

       # Update t amplitude
       t_amp_new = @. e_denom * (mp2 + cepa1 + cepa2 + cepa3)

       # Evaluate Energy
       E_CEPA0 = 1/4 * @tensor scalar((gmo[o,o,v,v])[i,j,a,b] * t_amp_new[a,b,i,j])
       t_amp = t_amp_new
       dE = E_CEPA0 - E_old
       printfmt("CEPA0 Iteration {1:3d}: Energy = {2:4.12f} dE = {3:1.5e}\n", cc_iter, E_CEPA0, dE)

       if abs(dE) < 1.e-8
           @info "CEPA0 Iterations have converged!"
           break
       end

       if cc_iter == MAXITER
           psi4.core.clean()
           error("Maximum number of iterations exceeded.")
       end
   end
   E_CEPA0
end

printfmt("\nCEPA0 Correlation Energy: {:5.15f}\n", E_CEPA0)
printfmt("CEPA0 Total Energy: {:5.15f}\n", E_CEPA0 + scf_e)
# -

# Since `t_amp` is initialized to zero, the very first iteration should be the MP2 correlation energy. We can check the final CEPA0 energy with Psi4. The method is called `lccd`, or linear CCD, since CEPA0 omits the terms with two cluster amplitudes.

psi4.compare_values(psi4.energy("lccd"), E_CEPA0 + scf_e, 6, "CEPA0 Energy")

# # Implementation: CCD
#
# To code CCD, we only have to add in the last four terms in our expression for the $t$-amplitudes: 
#
# \begin{equation}
# t_{ab}^{ij} = (\mathcal{E}_{ab}^{ij})^{-1} \left( \bar{g}_{ab}^{ij} + \tfrac{1}{2} \bar{g}_{ab}^{cd} t_{cd}^{ij} + \tfrac{1}{2} \bar{g}_{kl}^{ij} t_{ab}^{kl}  + \hat{P}_{(a \space / \space b)}^{(i \space / \space j)} \bar{g}_{ak}^{ic} t_{bc}^{jk} - \underline{\tfrac{1}{2}\hat{P}_{(a \space / \space b)} \bar{g}_{kl}^{cd} t_{ac}^{ij} t_{bd}^{kl} - \tfrac{1}{2}  \hat{P}^{(i \space / \space j)} \bar{g}_{kl}^{cd} t_{ab}^{ik} t_{cd}^{jl} + \tfrac{1}{4} \bar{g}_{kl}^{cd} t_{cd}^{ij} t_{ab}^{kl} +  \hat{P}^{(i \space / \space j)} \bar{g}_{kl}^{cd} t_{ac}^{ik} t_{bd}^{jl}} \right)
# \end{equation}
#
# which we readily translate into `@tensor`'s:
#
# ~~~julia
# @tensor ccd1a[a,b,i,j] := (gmo[o,o,v,v])[k,l,c,d] * t_amp[a,c,i,j] * t_amp[b,d,k,l]
# ccd1b  = -permutedims(ccd1a, (2, 1, 3, 4))
# ccd1   = -0.5(ccd1a + ccd1b)
#
# @tensor ccd2a[a,b,i,j] := (gmo[o,o,v,v])[k,l,c,d] * t_amp[a,b,i,k] * t_amp[c,d,j,l]
# ccd2b  = -permutedims(ccd2a, (1, 2, 4, 3))
# ccd2   = -0.5(ccd2a + ccd2b)
#
# @tensor ccd3[a,b,i,j] := 1/4 * (gmo[o,o,v,v])[k,l,c,d] * t_amp[c,d,i,j] * t_amp[a,b,k,l]
#
# @tensor ccd4a[a,b,i,j] := (gmo[o,o,v,v])[ k,l,c,d] * t_amp[a,c,i,k] * t_amp[b,d,j,l]
# ccd4b  = -permutedims(ccd4a, (1, 2, 4, 3))
# ccd4   = (ccd4a + ccd4b)
# ~~~
#
# and the energy expression is identical to CEPA0:
# \begin{equation}
# E_{CCD } = \tfrac{1}{4} \bar{g}_{ij}^{ab} t_{ab}^{ij}
# \end{equation}
#
# Adding the above terms to our CEPA0 code will compute the CCD correlation energy (may take a minute or two to run):

# +
# Initialize energy
E_CCD = let E_CCD = 0.0, o=o,v=v, e_denom=e_denom, t_amp=t_amp

   MAXITER = 50

   # Create space to store t amplitudes 
   t_amp = zeros(nvirt, nvirt, nocc, nocc)
   for cc_iter in 1:MAXITER
       E_old = E_CCD

       # Collect terms
       mp2      = @view gmo[v,v,o,o]
       @tensor cepa1[ a,b,i,j] := 0.5(gmo[v,v,v,v])[a,b,c,d] * t_amp[c,d,i,j]
       @tensor cepa2[ a,b,i,j] := 0.5(gmo[o,o,o,o])[k,l,i,j] * t_amp[a,b,k,l]
       @tensor cepa3a[a,b,i,j] :=    (gmo[v,o,o,v])[a,k,i,c] * t_amp[b,c,j,k]
       cepa3b = -permutedims(cepa3a, (2, 1, 3, 4))
       cepa3c = -permutedims(cepa3a, (1, 2, 4, 3))
       cepa3d =  permutedims(cepa3a, (2, 1, 4, 3))
       cepa3  =  cepa3a + cepa3b + cepa3c + cepa3d

       @tensor ccd1a_ref[a,b,i,j] := (gmo[o,o,v,v])[k,l,c,d] * t_amp[a,c,i,j] * t_amp[b,d,k,l]
       @tensor ccd1a_tmp[c,b]     := (gmo[o,o,v,v])[k,l,c,d] * t_amp[b,d,k,l]
       @tensor ccd1a[a,b,i,j]     := ccd1a_tmp[c,b]    * t_amp[a,c,i,j]
       println(isapprox(ccd1a_ref, ccd1a))
       
       ccd1b  = -permutedims(ccd1a, (2, 1, 3, 4))
       ccd1   = -0.5(ccd1a + ccd1b)

       @tensor ccd2a_ref[a,b,i,j] := (gmo[o,o,v,v])[k,l,c,d] * t_amp[a,b,i,k] * t_amp[c,d,j,l]
       @tensor ccd2a_tmp[j,k]     := (gmo[o,o,v,v])[k,l,c,d] * t_amp[c,d,j,l]
       @tensor ccd2a[a,b,i,j]     := ccd2a_tmp[j,k]    * t_amp[a,b,i,k]
       println(isapprox(ccd2a_ref, ccd2a))
       
       ccd2b  = -permutedims(ccd2a, (1, 2, 4, 3))
       ccd2   = -0.5(ccd2a + ccd2b)

       @tensor ccd3_ref[a,b,i,j] := 1/4 * (gmo[o,o,v,v])[k,l,c,d] * t_amp[c,d,i,j] * t_amp[a,b,k,l]
       @tensor ccd3_tmp[k,l,i,j] :=       (gmo[o,o,v,v])[k,l,c,d] * t_amp[c,d,i,j]
       @tensor ccd3[a,b,i,j]     := 1/4 * ccd3_tmp[k,l,i,j] * t_amp[a,b,k,l]
       println(isapprox(ccd3_ref, ccd3))

       @tensor ccd4a_ref[a,b,i,j] := (gmo[o,o,v,v])[ k,l,c,d] * t_amp[a,c,i,k] * t_amp[b,d,j,l]
       @tensor ccd4a_tmp[l,a,i,d] := (gmo[o,o,v,v])[ k,l,c,d] * t_amp[a,c,i,k]
       @tensor ccd4a[a,b,i,j]     := ccd4a_tmp[l,a,i,d] * t_amp[b,d,j,l]
       println(isapprox(ccd4a_ref, ccd4a))
       
       ccd4b  = -permutedims(ccd4a, (1, 2, 4, 3))
       ccd4   = ccd4a + ccd4b

       # Update Amplitude
       t_amp_new = @. e_denom * (mp2 + cepa1 + cepa2 + cepa3 + ccd1 + ccd2 + ccd3 + ccd4)

       # Evaluate Energy
       E_CCD = 1/4 * @tensor scalar((gmo[o,o,v,v])[i,j,a,b] * t_amp_new[a,b,i,j])
       t_amp = t_amp_new
       dE = E_CCD - E_old
       printfmt("CCD Iteration {1:3d}: Energy = {2:4.12f} dE = {3:1.5e}\n", cc_iter, E_CCD, dE)

       if abs(dE) < 1.e-8
           @info "CCD Iterations have converged!"
           break
       end

       if cc_iter == MAXITER
           psi4.core.clean()
           error("Maximum number of iterations exceeded.")
       end
   end
   E_CCD
end

printfmt("\nCCD Correlation Energy:    {:15.12f}\n", E_CCD)
printfmt("CCD Total Energy:         {:15.12f}\n", E_CCD + scf_e)
# -

# Unfortunately, Psi4 does not have a CCD code to compare this to. However, Psi4 does have Bruekner CCD, an orbital-optimized variant of CCD. We can qualitatively compare our energies to this energy. The Bruekner-CCD energy should be a little lower than our CCD energy due to the orbital optimization procedure.

psi4_bccd = psi4.energy("bccd", ref_wfn = scf_wfn)
printfmt("\nPsi4 BCCD Correlation Energy:    {:15.12f}\n", psi4_bccd - scf_e)
printfmt("Psi4 BCCD Total Energy:         {:15.12f}\n", psi4_bccd)

# ## References
#
# 1. Modern review of coupled-cluster theory, included diagrammatic derivations of the CCD equations:
# 	> [[Bartlett and Musial:2007](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.79.291)] Rodney J. Bartlett and Monika Musial,  "Coupled-cluster theory in quantum chemistry" *Rev. Mod. Phys.* **79**, 291 (2007)
#    
# 2. Background on CEPA:
#     >Kutzelnigg, Werner 1977 *Methods of Electronic Structure Theory* ed. H. F. Schaefer III (Plenum, New York), p 129
#
# 3. More CEPA:
#     > [Koch and Kutzelnigg:1981](https://link.springer.com/article/10.1007/BF00553396) S. Koch and W. Kutzelnigg, *Theor. Chim. Acta* **59**, 387 (1981). 
#
# 4. Original CCD Paper:
#     > [Čížek:1966](http://aip.scitation.org/doi/abs/10.1063/1.1727484) Jiří Čížek, "On the Correlation Problem in Atomic and Molecular Systems. Calculation of Wavefunction Components in Ursell‐Type Expansion Using Quantum‐Field Theoretical Methods" *J. Chem. Phys* **45**, 4256 (1966)  
#
# 5. Useful notes on diagrams applied to post-HF methods:
#     > A. V. Copan, "Diagram notation" accessed with https://github.com/CCQC/chem-8950/tree/master/2017
#
