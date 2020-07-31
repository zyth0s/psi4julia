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
"""Tutorial: Second-Order Moller--Plesset Perturbation Theory (MP2)"""

__author__    = ["D. Menendez", "Dominic A. Sirianni"]
__credit__    = ["D. Menendez", "Dominic A. Sirianni", "Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2020, The Psi4Julia Developers"
__license__   = "BSD-3-Clause"
__date__      = "2020-07-30"
# -

# # Second-Order Moller-Plesset Perturbation Theory (MP2)
#
# Moller-Plesset perturbation theory [also referred to as many-body perturbation theory (MBPT)] is an adaptation of the more general Rayleigh-Schrodinger perturbation theory (RSPT), applied to problems in molecular electronic structure theory.  This tutorial will provide a brief overview of both RSPT and MBPT, before walking through an implementation of second-order Moller-Plesset perturbation theory (specifically referred to as MP2) which uses conventional, 4-index ERIs.  
#
# ### I. Overview of Rayleigh-Schrodinger Perturbation Theory
# Given the Hamiltonian operator $\hat{H}$ for a system, perturbation theory solves the Schrodinger equation for that system by rewriting $\hat{H}$ as
#
# \begin{equation}
# \hat{H} = \hat{H}{}^{(0)} + \lambda\hat{V},
# \tag{[Szabo:1996], pp. 322, Eqn. 6.3}
# \end{equation}
#
# were $\hat{H}{}^{(0)}$ is the Hamiltonian operator corresponding to a solved problem which resembles $\hat{H}$, and $\hat{V}$ is the *perturbation* operator, defined as $\hat{V} = \hat{H} - \hat{H}{}^{(0)}$.  Then the Schrodinger equation for the system becomes 
#
# \begin{equation}
# \hat{H}\mid\Psi_n\rangle = (\hat{H}{}^{(0)} + \lambda\hat{V})\mid\Psi_n\rangle = E_n\mid\Psi_n\rangle.
# \tag{[Szabo:1996], pp. 322, Eqn. 6.2}
# \end{equation}
#
# The energies $E_n$ and wavefunctions $\mid\Psi_n\rangle$ will both be functions of $\lambda$; they can therefore be written as a Taylor series expansion about $\lambda = 0$ ([Szabo:1996], pp. 322, Eqns. 6.4a & 6.4b):
#
# \begin{align}
# E_n &= E_n^{(0)} + \lambda E_n^{(1)} + \lambda^2E_n^{(2)} + \ldots;\tag{[Szabo:1996], pp. 322, Eqn. 6.4a}\\
# \mid\Psi_n\rangle &= \mid\Psi_n^{(0)}\rangle + \lambda\mid\Psi_n^{(1)}\rangle + \lambda^2\mid\Psi_n^{(2)}\rangle + \ldots,\tag{[Szabo:1996], pp. 322, Eqn. 6.4b}\\
# \end{align}
#
# in practice, these perturbation expansions may be truncated to a given power of $\lambda$.  Substituting the perturbation series above back into the Schrodinger equation yields
#
# \begin{equation*}
# (\hat{H}{}^{(0)} + \lambda\hat{V})(\mid\Psi_n^{(0)}\rangle + \lambda\mid\Psi_n^{(1)}\rangle + \lambda^2\mid\Psi_n^{(2)}\rangle + \ldots) = (E_n^{(0)} + \lambda E_n^{(1)} + \lambda^2E_n^{(2)} + \ldots)(\mid\Psi_n^{(0)}\rangle + \lambda\mid\Psi_n^{(1)}\rangle + \lambda^2\mid\Psi_n^{(2)}\rangle + \ldots),
# \end{equation*}
#
# which by equating powers of $\lambda$ ([Szabo:1996], pp. 323, Eqns. 6.7a-6.7d) gives expressions for the $E_n^{(i)}$and $\mid\Psi_n^{(i)}\rangle$. Note that for $\lambda^0$, $E_n^{(0)}$ and $\mid\Psi_n^{(0)}\rangle$ are known, as they are the solution to the zeroth-order problem $\hat{H}{}^{(0)}$.  For $\lambda^1$ and $\lambda^2$, the expressions for $E_n^{(1)}$ and $E_n^{(2)}$ are given by
#
# \begin{align}
# \lambda^1:\;\;\;\;E_n^{(1)} &= \langle\Psi_n^{(0)}\mid\hat{V}\mid\Psi_n^{(0)}\rangle,\tag{[Szabo:1996], pp. 323, Eqn. 6.8b}\\
# \lambda^2:\;\;\;\;E_n^{(2)} &= \sum_{\mu\neq n}\frac{\mid\langle\Psi_{n}^{(0)}\mid\hat{V}\mid\Psi_{\mu}^{(0)}\rangle\mid^2}{E_n^{(0)} - E_{\mu}^{(0)}}\tag{[Szabo:1996], pp. 324, Eqn. 6.12}\\
# \end{align}
#

# ### II. Overview of Moller-Plesset Perturbation Theory
#
# The exact electronic Hamiltonian for an N-electron molecule with a given atomic configuration is (in atomic units):
#
# \begin{equation}
# \hat{H}_{elec} = \sum_i\hat{h}(i) + \sum_{i<j}\frac{1}{r_{ij}}.\tag{[Szabo:1996], pp. 43, Eqn. 2.10}
# \end{equation}
#
# Moller-Plesset perturbation theory seeks to solve the molecular electronic Schrodinger equation with the above Hamiltonian using the techniques of Rayleigh-Schroding perturbation theory, by selecting the zeroth-order Hamiltonian and wavefunctions to be those from Hartree-Fock theory:
#
# \begin{align}
# \hat{H}{}^{(0)}  &= \sum_i \hat{f}(i) = \sum_i \hat{h}(i) + \upsilon^{HF}(i)\tag{[Szabo:1996], pp. 350, Eqn. 6.59}\\
# \hat{V} &= \hat{H} - \hat{H}{}^{(0)} = \sum_{i<j} \frac{1}{r_{ij}} - \upsilon^{HF}(i).\tag{[Szabo:1996], pp. 350, Eqn. 6.60}\\
# \end{align}
#
# With these choices of $\hat{H}{}^{(0)}$, $\hat{V}$, and $\mid\Psi_n^{(0)}\rangle$, the ground-state zeroth-order energy is given by $E_0^{(0)} = \sum_i\epsilon_i$ ([Szabo:1996], pp. 351, Eqn. 6.67).  Then, the first-order ground state energy may be computed by $E_0^{(1)} = \langle\Psi_0^{HF}\mid\hat{V}\mid\Psi_0^{HF}\rangle$ to find that the total ground-state energy through first order, $E_0 = E_0^{(0)} + E_0^{(1)}$, is exactly the Hartree-Fock energy. ([Szabo:1996], pp. 351, Eqn. 6.69)  Therefore, the first correction to the Hartree-Fock energy will occur at second-order in the perturbation series, i.e., with $E_0^{(2)}$; truncating the perturbation series at second order is commonly referred to as MP2.  The second order correction to the ground state energy will be given by
#
# \begin{equation}
# E_0^{(2)} = \sum_{\mu\neq 0}\frac{\left|\langle\Psi_{0}^{HF}\mid\hat{V}\mid\Psi_{\mu}^{HF}\rangle\right|^2}{E_0^{(0)} - E_{\mu}^{(0)}}\tag{[Szabo:1996], pp. 351, Eqn. 6.70}
# \end{equation}
#
# For brevity, we will now drop the "HF" from all zeroth-order wavefunctions.  This summation is over the eigenstate index $\mu$, each associated with a different eigenstate of the zeroth-order Hamiltonian.  For a single-determinant wavefunction constructed from spin orbitals, the summation over the eigenstate index $\mu\neq 0$ therefore refers to determinants which are constructed from *different* spin orbitals than the ground state determinant.  To distinguish such determinants, we will denote MOs occupied in the ground state with indices $i,\,j,\,\ldots$, and MOs which are unoccupied in the ground state with indices $a,\,b,\,\ldots\,$.  Then a determinant where orbital $a$ is substituted for orbital $i$ is denoted $\mid\Psi_i^a\rangle$, and so on.  Before substituting this new notation into the above energy expression, however, we may immediately recognize that many terms $\langle\Psi_{0}\mid\hat{V}\mid\Psi_{\mu}\rangle$ will not contribute to the second order energy:
#
# | Term         | Determinant                               |  Contribution to $E_0^{(2)}$                            |
# |--------------|-------------------------------------------|---------------------------------------------------------|
# | Singles      | $\mid\Psi_i^a\rangle$                     | 0; Brillouin's Theorem                                  |
# | Doubles      | $\mid\Psi_{ij}^{ab}\rangle$               | Survive                                                 |
# | Higher-order | $\mid\Psi_{ijk\ldots}^{abc\ldots}\rangle$ | 0; $\hat{V}$ is a two-particle operator            |
#
# Hence we see that only doubly-substituted determinants $\mid\Psi_{ij}^{ab}\rangle$ will contribute to $E_0^{(2)}$.  From Hartree-Fock theory, we know that 
#
# \begin{equation}
# \langle\Psi_0\mid r_{ij}^{-1}\mid\Psi_{ij}^{ab}\rangle = [ia\| jb],\tag{[Szabo:1996], pp. 72, Tbl. 2.6}
# \end{equation}
#
# where $[ia\| jb] = [ia\mid jb] - [ib\mid ja]$ is an antisymmetrized two-electron integral, and the square brackets "$[\ldots ]$" indicate that we are employing chemists' notation.  What about the energies of these doubly substituted determinants?  Recognizing that the difference between the energies of the newly- and formerly-occupied orbitals in each substitution must modify the total energy of the ground state determinant, 
#
# \begin{equation}
# E_{ij}^{ab} = E_0 - (\epsilon_i - \epsilon_a + \epsilon_j - \epsilon_b).
# \end{equation}
#
# Substituting these expressions into the one for the second-order energy, we have that
#
# \begin{equation}
# E_0^{(2)} = \sum_{i<j}\sum_{a<b} \frac{\left|\,[ia\|jb]\,\right|^2}{\epsilon_i - \epsilon_a + \epsilon_j - \epsilon_b}.\tag{[Szabo:1996], pp. 352, Eqn. 6.71}
# \end{equation}
#
# So far, our discussion has used spin-orbitals instead of the more familiar spatial orbitals.  Indeed, significant speedups are achieved when using spatial orbitals.  Integrating out the spin variable $\omega$ from $E_0^{(2)}$ yields two expressions; one each for the interaction of particles with the same spin (SS) and opposite spin (OS):
#
# \begin{align}
# E_{\rm 0,\,SS}^{(2)} = \sum_{ij}\sum_{ab}\frac{(ia\mid jb)[(ia\mid jb) - (ib\mid ja)]}{\epsilon_i - \epsilon_a + \epsilon_j - \epsilon_b}, \quad
# E_{\rm 0,\,OS}^{(2)} = \sum_{ij}\sum_{ab}\frac{(ia\mid jb)(ia\mid jb)}{\epsilon_i - \epsilon_a + \epsilon_j - \epsilon_b},
# \end{align}
#
# where these spin-free expressions make use of integrals in chemists' notation over spatial orbitals. (Rearranged from [Szabo:1996], pp. 352, Eqn. 6.74)  Note that an exchange integral arises between particles of the same spin; this is because the motions of particles with identical spins are correlated due to the requirement that $\left|\Psi\right|^2$ remain invariant to the exchange of the spatial and spin coordinates of any pair of electrons.  Finally, the total MP2 correction energy $E_0^{(2)} = E_{\rm 0,\,SS}^{(2)} + E_{\rm 0,\,OS}^{(2)}$.
#
# ### Implementation of Conventional MP2
#
# Let's begin by importing Psi4, NumPy and TensorOperations, and setting memory and output file options:

# +
# ==> Import statements & Global Options <==
using PyCall: pyimport
psi4 = pyimport("psi4")
np   = pyimport("numpy") # used only to cast to Psi4 arrays
using TensorOperations: @tensor

psi4.set_memory(Int(2e9))
numpy_memory = 2
psi4.core.set_output_file("output.dat", false)
# -

# Next, we can define our molecule and Psi4 options.  Notice that we are using `scf_type pk` to indicate that we wish to use conventional, full 4-index ERIs, and that we have specified `mp2_type conv` so that the MP2 algorithm we check against also uses the conventional ERIs.

# +
# ==> Molecule & Psi4 Options Definitions <==
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")


psi4.set_options(Dict("basis"         => "6-31g",
                      "scf_type"      => "pk",
                      "mp2_type"      => "conv",
                      "e_convergence" => 1e-8,
                      "d_convergence" => 1e-8))
# -

# Since MP2 is a perturbation on the zeroth-order Hartree-Fock description of a molecular system, all of the relevant information (Fock matrix, orbitals, orbital energies) about the system can be computed using any Hartree-Fock program.  We could use the RHF program that we wrote in tutorial 3a, but we could just as easily use Psi4 to do our dirty work.  In the cell below, use Psi4 to compute the RHF energy and wavefunction, and store them using the `return_wfn=True` keyword argument to `psi4.energy()`:

# Get the SCF wavefunction & energies
scf_e, scf_wfn = psi4.energy("scf", return_wfn=true)

# In the expression for $E_0^{(2)}$, the two summations are over occupied and virtual indices, respectively.  Therefore, we'll need to get the number of occupied orbitals and the total number of orbitals.  Additionally, we must obtain the MO energy eigenvalues; again since the sums are over occupied and virtual orbitals, it is good to separate the occupied orbital energies from the virtual orbital energies.  From the SCF wavefunction you generated above, get the number of doubly occupied orbitals, number of molecular orbitals, and MO energies:

# +
# ==> Get orbital information & energy eigenvalues <==
# Number of Occupied orbitals & MOs
ndocc = scf_wfn.nalpha()
nmo = scf_wfn.nmo()

# Get orbital energies, cast into NumPy array, and separate occupied & virtual
eps = np.asarray(scf_wfn.epsilon_a())
e_ij = eps[1:ndocc]
e_ab = eps[ndocc+1:end];
# -

# Unlike the orbital information, Psi4 does not return the ERIs when it does a computation.  Fortunately, however, we can just build them again using the `psi4.core.MintsHelper()` class.  Recall that these integrals will be generated in the AO basis; before using them in the $E_0^{(2)}$ expression, we must transform them into the MO basis.  To do this, we first need to obtain the orbital coefficient matrix, **C**.  In the cell below, generate the ERIs for our molecule, get **C** from the SCF wavefunction, and obtain occupied- and virtual-orbital slices of **C** for future use. 

# +
# ==> ERIs <==
# Create instance of MintsHelper class
mints = psi4.core.MintsHelper(scf_wfn.basisset())

# Memory check for ERI tensor
I_size = nmo^4 * 8.e-9
println("\nSize of the ERI tensor will be $I_size GB.")
memory_footprint = I_size * 1.5
if I_size > numpy_memory
    psi4.core.clean()
    throw(OutOfMemoryError("Estimated memory utilization ($memory_footprint GB) exceeds " * 
                           "allotted memory limit of $numpy_memory GB."))
end

# Build ERI Tensor
I = np.asarray(mints.ao_eri())

# Get MO coefficients from SCF wavefunction
C = np.asarray(scf_wfn.Ca())
Cocc = C[:, 1:ndocc]
Cvirt = C[:, ndocc+1:end];
# -

# In order to transform the four-index integrals from the AO to the MO basis, we must perform the following contraction:
#
# $$(i\,a\mid j\,b) = C_{\mu i}C_{\nu a}(\mu\,\nu\mid|\,\lambda\,\sigma)C_{\lambda j}C_{\sigma b}$$
#
# Again, here we are using $i,\,j$ as occupied orbital indices and $a,\, b$ as virtual orbital indices.  We could carry out the above contraction all in one step using either `@tensor` or explicit loops:
#
# ~~~julia
# # Naive Algorithm for ERI Transformation
# @tensor I_mo[i,a,j,b] := Cocc[p,i] * Cvirt[q,a] * I[p,q,r,s] * Cocc[r,j] * Cvirt[s,b]
# ~~~
#
# Notice that the transformation from AO index to occupied (virtual) MO index requires only the occupied (virtual) block of the **C** matrix; this allows for computational savings in large basis sets, where the virtual space can be very large.  This algorithm, while efficient with `@tensor`, has horrendous scaling if the search of an optimal contraction fails.  We will enforce a better contraction. Examining the contraction more closely, we see that there are 8 unique indices, and thus the step above scales as ${\cal O}(N^8)$.  With this algorithm, a twofold increase of the number of MO's would result in $2^8 = 256\times$ expense to perform.  We can, however, refactor the above contraction such that
#
# $$(i\,a\mid j\,b) = \left[C_{\mu i}\left[C_{\nu a}\left[C_{\lambda j}\left[C_{\sigma b}(\mu\,\nu\mid|\,\lambda\,\sigma)\right]\right]\right]\right],$$
#
# where we have now written the transfomation as four ${\cal O}(N^5)$ steps instead of one ${\cal O}(N^8)$ step. This is a savings of $\frac{4}{n^3}$, and is responsible for the feasibility of the MP2 method for application to any but very small systems and/or basis sets.  We may carry out the above ${\cal O}(N^5)$ algorithm by carrying out one index transformation at a time, and storing the result in a temporary array.  In the cell below, transform the ERIs from the AO to MO basis, using our smarter algorithm:

# ==> Transform I -> I_mo @ O(N⁵) <==
I_mo = @tensor begin
   I_mo[i,q,r,s] := Cocc[p,i]     * I[p,q,r,s]
   I_mo[i,a,r,s] := Cvirt[q,a]    * I_mo[i,q,r,s]
   I_mo[i,a,j,s] :=                 I_mo[i,a,r,s] * Cocc[r,j]
   I_mo[i,a,j,b] :=                 I_mo[i,a,j,s] * Cvirt[s,b]
end
nothing

# We note here that we can use infrastructure in Psi4 to carry out the above integral transformation; this entails obtaining the occupied and virtual blocks of **C** Psi4-side, and then using the built-in `MintsHelper` function `MintsHelper.mo_eri()` to transform the integrals.  Just to check your work above, execute the next cell to see this tech in action:

# ==> Compare our Imo to MintsHelper <==
Co = scf_wfn.Ca_subset("AO","OCC")
Cv = scf_wfn.Ca_subset("AO","VIR")
MO = np.asarray(mints.mo_eri(Co, Cv, Co, Cv))
println("Do our transformed ERIs match Psi4's? ", np.allclose(I_mo, np.asarray(MO)))

# Now we have all the pieces needed to compute $E_0^{(2)}$.  This could be done by writing explicit loops over occupied and virtual indices Julia side, e.g.,

# +
# Compute SS & OS MP2 Correlation
mp2_corr = let mp2_ss_corr = 0.0, mp2_os_corr = 0.0
   nvirt = nmo - ndocc
   for i in 1:ndocc, a in 1:nvirt, j in 1:ndocc, b in 1:nvirt
       numerator = I_mo[i,a,j,b] * (I_mo[i, a, j, b] - I_mo[i, b, j, a])
       mp2_ss_corr += numerator / (e_ij[i] + e_ij[j] - e_ab[a] - e_ab[b])
       mp2_os_corr += I_mo[i,a,j,b]^2 / (e_ij[i] + e_ij[j] - e_ab[a] - e_ab[b])
   end
   mp2_ss_corr + mp2_os_corr
end

# Total MP2 Energy
MP2_E = scf_e + mp2_corr

# ==> Compare to Psi4 <==
psi4.compare_values(psi4.energy("mp2"), MP2_E, 6, "MP2 Energy")
# -

# In this method it is very clear what is going on and is easy to program. Julia has the distinct advantage loops are as fast as the same block written in a compiled language like C, C++, or Fortran. 
#
# However, we will provide an alternative formulation using tensor contractions. It should be clear how to contract the four-index integrals $(i\,a\mid j\,b)$ and $(i\,a\mid j\,b)$ with one another, but what about the energy eigenvalues $\epsilon$?  We can use a Julia trick called *broadcasting* to construct a four-index array of all possible energy denominators, which can then be contracted with the full I_mo arrays.  To do this, we'll use the function `reshape()`:
# ~~~julia
# # Prepare 4d energy denominator array
# e_denom  = reshape(e_ij,  1, 1, 1, :)      # Diagonal of 4d array are occupied orbital energies
# e_denom -= reshape(e_ab', 1, 1, :)         # all combinations of (e_ij - e_ab)
# e_denom += e_ij                            # all combinations of [(e_ij - e_ab) + e_ij]
# e_denom -= e_ab'                           # All combinations of full denominator
# e_denom  = premutedims(e_denom, (1,2,4,3)) # permute 3rd and 4th dims to have (nocc,nvirt,nocc,nvirt) shape
# e_denom  = inv.(e_denom)                   # Take reciprocal for contracting with numerator
# ~~~
# In the cell below, compute the energy denominator using `reshape()` and contract this array with the four-index ERIs to compute the same-spin and opposite-spin MP2 correction using `sum()`. Then, add these quantities to the SCF energy computed above to obtain the total MP2 energy.
#
# Hint: For the opposite-spin correlation, use `permutedims()` to obtain the correct ordering of the indices in the exchange integral.

# +
#using Einsum: @einsum
# ==> Compute MP2 Correlation & MP2 Energy <==
# Compute energy denominator array
e_denom = reshape(e_ij,1,1,1,:) .- reshape(e_ab',1,1,:) .+ (e_ij .- e_ab')
e_denom = permutedims(e_denom, (1,2,4,3)) # 3 ↔ 4
e_denom = inv.(e_denom)

# check
#using Test
#nvirt = nmo - ndocc
#for i in 1:ndocc, a in 1:nvirt, j in 1:ndocc, b in 1:nvirt
#    @test e_denom[i,a,j,b] ≈ 1 / (e_ij[i] + e_ij[j] - e_ab[a] - e_ab[b])
#end

# Compute SS & OS MP2 Correlation with sum()
bctd_mp2_os_corr = sum(I_mo .* I_mo .* e_denom)
I_mo_swap = permutedims(I_mo,(3,2,1,4)) # 1 ↔ 3
bctd_mp2_ss_corr = sum(I_mo .* (I_mo .- I_mo_swap) .* e_denom)

# Compare broadcasted and loop MP2
@assert bctd_mp2_os_corr + bctd_mp2_ss_corr ≈ mp2_corr

# Total MP2 Energy
MP2_E = scf_e + bctd_mp2_os_corr + bctd_mp2_ss_corr
# -

# ==> Compare to Psi4 <==
psi4.compare_values(psi4.energy("mp2"), MP2_E, 6, "MP2 Energy")

# ## References
#
# 1. Original paper: "Note on an Approximation Treatment for Many-Electron Systems"
# 	> [[Moller:1934:618](https://journals.aps.org/pr/abstract/10.1103/PhysRev.46.618)] C. Møller and M. S. Plesset, *Phys. Rev.* **46**, 618 (1934)
# 2. The Laplace-transformation in MP theory: "Minimax approximation for the decomposition of energy denominators in Laplace-transformed Møller–Plesset perturbation theories"
#     > [[Takasuka:2008:044112](http://aip.scitation.org/doi/10.1063/1.2958921)] A. Takatsuka, T. Siichiro, and W. Hackbusch, *J. Phys. Chem.*, **129**, 044112 (2008)
# 3. Equations taken from:
# 	> [[Szabo:1996](https://books.google.com/books?id=KQ3DAgAAQBAJ&printsec=frontcover&dq=szabo+%26+ostlund&hl=en&sa=X&ved=0ahUKEwiYhv6A8YjUAhXLSCYKHdH5AJ4Q6AEIJjAA#v=onepage&q=szabo%20%26%20ostlund&f=false)] A. Szabo and N. S. Ostlund, *Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory*. Courier Corporation, 1996.
# 4. Algorithms taken from:
# 	> [Crawford:prog] T. D. Crawford, "The Second-Order Møller–Plesset Perturbation Theory (MP2) Energy."  Accessed via the web at http://github.com/CrawfordGroup/ProgrammingProjects.
