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
"""Tutorial: Introduction to the Spin-Orbital Formulation of Post-HF Methods"""

__author__    = ["D. Menendez", "Adam S. Abbott"]
__credit__    = ["D. Menendez", "Adam S. Abbott", "Justin M. Turney"]

__copyright__ = "(c) 2014-2020, The Psi4Julia Developers"
__license__   = "BSD-3-Clause"
__date__      = "2020-08-02"
# -

# # Introduction to the Spin Orbital Formulation of Post-HF Methods
# ## Notation
#
# Post-HF methods such as MPn, coupled cluster theory, and configuration interaction improve the accuracy of our Hartree-Fock wavefunction by including terms corresponding to excitations of electrons from occupied (i, j, k..) to virtual (a, b, c...) orbitals. This recovers some of the dynamic electron correlation previously neglected by Hartree-Fock.
#
# It is convenient to introduce new notation to succinctly express the complex mathematical expressions encountered in these methods. This tutorial will cover this notation and apply it to a spin orbital formulation of conventional MP2. This code will also serve as a starting template for other tutorials which use a spin-orbital formulation, such as CEPA0, CCD, CIS, and OMP2. 
#
#
#
# ### I. Physicist's Notation for Two-Electron Integrals
# Recall from previous tutorials the form for the two-electron integrals over spin orbitals ($\chi$) and spatial orbitals ($\phi$):
# \begin{equation}
#  [pq|rs] = [\chi_p\chi_q|\chi_r\chi_s] = \int dx_{1}dx_2 \space \chi^*_p(x_1)\chi_q(x_1)\frac{1}{r_{12}}\chi^*_r(x_2)\chi_s(x_2) \\
# (pq|rs) = (\phi_p\phi_q|\phi_r\phi_s) = \int dx_{1}dx_2 \space \phi^*_p(x_1)\phi_q(x_1)\frac{1}{r_{12}}\phi^*_r(x_2)\phi_s(x_2)
# \end{equation}
#
# Another form of the spin orbital two electron integrals is known as physicist's notation. By grouping the complex conjugates on the left side, we may express them in Dirac ("bra-ket") notation:
# \begin{equation}
# \langle pq \mid rs \rangle = \langle \chi_p \chi_q \mid \chi_r \chi_s \rangle = \int dx_{1}dx_2 \space \chi^*_p(x_1)\chi^*_q(x_2)\frac{1} {r_{12}}\chi_r(x_1)\chi_s(x_2) 
# \end{equation}
#
# The antisymmetric form of the two-electron integrals in physcist's notation is given by
#
# \begin{equation}
# \langle pq \mid\mid rs \rangle = \langle pq \mid rs \rangle - \langle pq \mid sr \rangle
# \end{equation}
#
#
# ### II. Kutzelnigg-Mukherjee Tensor Notation and the Einstein Summation Convention
#
# Kutzelnigg-Mukherjee (KM) notation provides an easy way to express and manipulate the tensors (two-electron integrals, $t$-amplitudes, CI coefficients, etc.) encountered in post-HF methods. Indices which appear in the bra are expressed as subscripts, and indices which appear in the ket are expressed as superscripts:
# \begin{equation}
# g_{pq}^{rs} = \langle pq \mid rs \rangle \quad \quad \quad \overline{g}_{pq}^{rs} = \langle pq \mid\mid rs \rangle
# \end{equation}
#
# The upper and lower indices allow the use of the Einstein Summation convention. Under this convention, whenever an indice appears in both the upper and lower position in a product, that indice is implicitly summed over. As an example, consider the MP2 energy expression:
#
# \begin{equation}
# E_{MP2} = \frac{1}{4} \sum_{i a j b} \frac{ [ia \mid\mid jb] [ia \mid\mid jb]} {\epsilon_i - \epsilon_a + \epsilon_j - \epsilon_b}
# \end{equation}
# Converting to physicist's notation:
#
# \begin{equation}
# E_{MP2} = \frac{1}{4} \sum_{i j a b} \frac{ \langle ij \mid\mid ab \rangle \langle ij \mid \mid ab \rangle} {\epsilon_i - \epsilon_a + \epsilon_j - \epsilon_b}
# \end{equation}
# KM Notation, taking advantage of the permutational symmetry of $g$:
# \begin{equation}
# E_{MP2} = \frac{1}{4} \overline{g}_{ab}^{ij} \overline{g}_{ij}^{ab} (\mathcal{E}_{ab}^{ij})^{-1}
# \end{equation}
#
# where $\mathcal{E}_{ab}^{ij}$ is the sum of orbital energies $\epsilon_i - \epsilon_a + \epsilon_j - \epsilon_b$. Upon collecting every possible orbital energy sum into a 4-dimensional tensor, this equation can be solved with a simple tensor-contraction, as done in our MP2 tutorial.
#
# The notation simplication here is minor, but the value of this notation becomes obvious with more complicated expressions encountered in later tutorials such as CCD. It is also worth noting that KM notation is deeply intertwined with the second quantization and diagrammatic expressions of methods in advanced electronic structure theory. For our purposes, we will shy away from the details and simply use the notation to write out readily-programmable expressions.
#
#
# ### III. Coding Spin Orbital Methods Example: MP2
#
# In the MP2 tutorial, we used spatial orbitals in our two-electron integral tensor, and this appreciably decreased the computational cost. However, this code will only work when using an RHF reference wavefunction. We may generalize our MP2 code (and other post-HF methods) to work with any reference by expressing our integrals, MO coefficients, and orbital energies obtained from Hartree-Fock in a spin orbital formulation. As an example, we will code spin orbital MP2, and this will serve as a foundation for later tutorials.
#
#

# ### Implementation of Spin Orbital MP2
# As usual, we import Psi4, NumPy, and TensorOperations, and set the appropriate options. However, in this code, we will be free to choose open-shell molecules which require UHF or ROHF references. We will stick to RHF and water for now.

# +
# ==> Import statements & Global Options <==
using PyCall: pyimport
psi4 = pyimport("psi4")
np   = pyimport("numpy") # used only to cast to Psi4 arrays
using TensorOperations: @tensor
#using Einsum: @einsum

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

# For convenience, we let Psi4 take care of the Hartree-Fock procedure, and return the wavefunction object.

# Get the SCF wavefunction & energies
scf_e, scf_wfn = psi4.energy("scf", return_wfn=true)

# We also need information about the basis set and orbitals, such as the number of basis functions, number of spin orbitals, number of alpha and beta electrons, the number of occupied spin orbitals, and the number of virtual spin orbitals. These can be obtained with MintsHelper and from the wavefunction.

mints = psi4.core.MintsHelper(scf_wfn.basisset())
nbf = mints.nbf()
nso = 2nbf
nalpha = scf_wfn.nalpha()
nbeta = scf_wfn.nbeta()
nocc = nalpha + nbeta
nvirt = 2nbf - nocc

# For MP2, we need the MO coefficients, the two-electron integral tensor, and the orbital energies. But, since we are using spin orbitals, we have to manipulate this data accordingly. Let's get our MO coefficients in the proper form first. Recall in restricted Hartree-Fock, we obtain one MO coefficient matrix **C**, whose columns are the molecular orbital coefficients, and each row corresponds to a different atomic orbital basis function. But, in unrestricted Hartree-Fock, we obtain separate matrices for the alpha and beta spins, **Ca** and **Cb**. We need a general way to build one **C** matrix regardless of our Hartree-Fock reference. The solution is to put alpha and beta MO coefficients into a block diagonal form:

# +
Ca = np.asarray(scf_wfn.Ca())
Cb = np.asarray(scf_wfn.Cb())
C = hvcat((2,2),Ca,      zero(Ca),
                zero(Cb),     Cb); # direct sum

# Result: | Ca  0 |
#         | 0   Cb|
# -

# It's worth noting that for RHF and ROHF, the Ca and Cb given by Psi4 are the same.
#
# Now, for this version of MP2, we also need the MO-transformed two-electron integral tensor in physicist's notation. However, Psi4's default two-electron integral tensor is in the AO-basis, is not "spin-blocked" (like **C**, above!), and is in chemist's notation, so we have a bit of work to do. 
#
# First, we will spin-block the two electron integral tensor in the same way that we spin-blocked our MO coefficients above. Unfortunately, this transformation is impossible to visualize for a 4-dimensional array.
#
# Nevertheless, the math generalizes and can easily be achieved with NumPy's kronecker product function `np.kron`. Here, we take the 2x2 identity, and place the two electron integral array into the space of the 1's along the diagonal. Then, we transpose the result and do the same. The result doubles the size of each dimension, and we obtain a "spin-blocked" two electron integral array.

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

# From here, converting to antisymmetrized physicists notation is simply:

# Converts chemist's notation to physicist's notation, and antisymmetrize
# (pq|rs) ↦ ⟨pr|qs⟩
# Physicist's notation
tmp = permutedims(I_spinblock, (1, 3, 2, 4))
# Antisymmetrize:
# ⟨pr||qs⟩ = ⟨pr|qs⟩ - ⟨pr|sq⟩
gao = tmp - permutedims(tmp, (1, 2, 4, 3));

# We also need the orbital energies, and just as with the MO coefficients, we combine alpha and beta together. We also want to ensure that the columns of **C** are sorted in the same order as the corresponding orbital energies.

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

# Finally, we transform our two-electron integrals to the MO basis. For the sake of generalizing for other methods, instead of just transforming the MP2 relevant subsection as before:
# ~~~julia
# I_mo = @tensor begin
#    I_mo[i,q,r,s] := Cocc[p,i]     * I[p,q,r,s]
#    I_mo[i,a,r,s] := Cvirt[q,a]    * I_mo[i,q,r,s]
#    I_mo[i,a,j,s] :=                 I_mo[i,a,r,s] * Cocc[r,j]
#    I_mo[i,a,j,b] :=                 I_mo[i,a,j,s] * Cvirt[s,b]
# end
# ~~~
#
# we instead transform the full array so it can be used for terms from methods other than MP2. The nested `@tensor`'s work the same way as the method above. Here, we denote the integrals as `gmo` to differentiate from the chemist's notation integrals `I_mo`.

# Transform gao, which is the spin-blocked 4d array of physicist's notation, 
# antisymmetric two-electron integrals, into the MO basis using MO coefficients 
gmo = @tensor begin
   gmo[P,Q,R,S] := gao[p,Q,R,S] * C[p,P]
   gmo[p,Q,R,S] := gmo[p,q,R,S] * C[q,Q]
   gmo[p,q,R,S] := gmo[p,q,r,S] * C[r,R]
   gmo[p,q,r,S] := gmo[p,q,r,s] * C[s,S]
end
nothing

# And just as before, construct the 4-dimensional array of orbital energy denominators. An alternative to the old method:
# ~~~julia
# e_ijab = reshape(e_ij,1,1,1,:) .- reshape(e_ab',1,1,:) .+ (e_ij .- e_ab')
# e_ijab = permutedims(e_ijab, (1,2,4,3)) # 3 ↔ 4
# e_ijab = inv.(e_ijab)
# ~~~
# is the following:

# Define slices, create 4 dimensional orbital energy denominator tensor e_denom[a,b,i,j]
n = [CartesianIndex()]
o = [p ≤ nocc for p in 1:nso]
v = [p > nocc for p in 1:nso]
v = vcat(falses(nocc),  trues(nvirt))
e_denom = inv.(-eps[v, n, n, n] .- eps[n, v, n, n] .+ eps[n, n, o, n] .+ eps[n, n, n, o]);

# check
using Test
for i in 1:nocc, a in 1:nvirt, j in 1:nocc, b in 1:nvirt
    @test e_denom[a,b,i,j] ≈ 1 / (eps[i] + eps[j] - eps[a+nocc] - eps[b+nocc])
end

# These slices will also be used to define the occupied and virtual space of our two electron integrals. 
#
# For example, $\bar{g}_{ab}^{ij}$ can be accessed with `gmo[v, v, o, o]` 

# We now have all the pieces we need to compute the MP2 correlation energy. Our energy expression in KM notation is
#
# \begin{equation}
# E_{MP2} = \frac{1}{4} \bar{g}_{ab}^{ij} \bar{g}_{ij}^{ab} (\mathcal{E}_{ab}^{ij})^{-1}
# \end{equation}
#
# which may be easily read-off as a sum in Julia. Here, for clarity, we choose to read the tensors from left to right (bra to ket). We also are sure to take the appropriate slice of the two-electron integral array:

# +
# Compute MP2 Correlation Energy
gmo_vvoo = @view gmo[v,v,o,o]
gmo_oovv = permutedims(gmo[o,o,v,v], (3,4,1,2))
E_MP2_corr = (1 / 4) * sum(gmo_vvoo .* gmo_oovv .* e_denom)

#gmo_oovv = @view gmo[o,o,v,v]
#@einsum E_MP2_corr := (1 / 4) * gmo_vvoo[a,b,i,j] * gmo_oovv[i,j,a,b] * e_denom[a,b,i,j]

#@tensor gg[A,B,I,J] := gmo_vvoo[a,b,i,j] * gmo_oovv[i,j,a,b]
#E_MP2_corr = (1 / 4) * sum( gg .* e_denom)

E_MP2 = E_MP2_corr + scf_e

println("MP2 correlation energy: ", E_MP2_corr)
println("MP2 total energy: ", E_MP2)
# -

# Finally, compare our answer with Psi4:

# ==> Compare to Psi4 <==
psi4.compare_values(psi4.energy("mp2"), E_MP2, 6, "MP2 Energy")

# ## References
#
# 1. Notation and Symmetry of Integrals:
#     > C. David Sherill, "Permutational Symmetries of One- and Two-Electron Integrals" Accessed with http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.pdf
# 2. Useful Notes on Kutzelnigg-Mukherjee Notation: 
#     > A. V. Copan, "Kutzelnigg-Mukherjee Tensor Notation" Accessed with https://github.com/CCQC/chem-8950/tree/master/2017
#
# 3. Original paper on MP2: "Note on an Approximation Treatment for Many-Electron Systems"
# 	> [[Moller:1934:618](https://journals.aps.org/pr/abstract/10.1103/PhysRev.46.618)] C. Møller and M. S. Plesset, *Phys. Rev.* **46**, 618 (1934)
#     
#
