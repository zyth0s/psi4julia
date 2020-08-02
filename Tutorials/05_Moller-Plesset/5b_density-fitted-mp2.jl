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
"""Tutorial: Describing the implementation of density-fitted MP2 from an RHF reference"""

__author__    = ["D. Menendez", "Dominic A. Sirianni"]
__credit__    = ["Dominic A. Sirianni", "Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2020, The Psi4Julia Developers"
__license__   = "BSD-3-Clause"
__date__      = "2020-07-30"
# -

# # Density Fitted MP2
#
# As we saw in tutorial (5a), the single most expensive step for a conventional MP2 program using full ERIs is the integral transformation from the atomic orbital (AO) to molecular orbital (MO) basis, scaling as ${\cal O}(N^5)$.  The scaling of this step may be reduced to ${\cal O}(N^4)$ if we employ density fitting, as the three-index density fitted tensors may be transformed individually into the MO basis before being recombined to form the full four-index tensors in the MO basis needed by the MP2 energy expression.  This tutorial will discuss the specific challenges encountered when applying density fitting to an MP2 program.
#
# ### Implementation
# The first part of our DF-MP2 program will look exactly the same as the conventional MP2 program that we wrote in (5a), with the exception that we must specify the `scf_type df` and omit the option `mp2_type conv` within the `psi4.set_options()` block, to ensure that we are employing density fitting in the Hartree-Fock reference.  Below, implement the following:
#
# - Import Psi4, NumPy, and TensorOperations, and set memory & output file
# - Define our molecule and Psi4 options
# - Compute the RHF reference wavefucntion and energy
# - Obtain the number of occupied and virtual MOs, and total number of MOs
# - Get the orbital energies and coefficient matrix; partition into occupied & virtual blocks

# +
# ==> Import statements & Global Options <==
using PyCall: pyimport
psi4 = pyimport("psi4")
np   = pyimport("numpy") # used only to cast to Psi4 arrays
using TensorOperations: @tensor

psi4.set_memory(Int(2e9))
numpy_memory = 2
psi4.core.set_output_file("output.dat", false)

# +
# ==> Options Definitions & SCF E, Wfn <==
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")


psi4.set_options(Dict("basis"         => "aug-cc-pvdz",
                      "scf_type"      => "df",
                      "e_convergence" => 1e-8,
                      "d_convergence" => 1e-8))

# Get the SCF wavefunction & energies
scf_e, scf_wfn = psi4.energy("scf", return_wfn=true)

# Number of Occupied orbitals & MOs
ndocc = scf_wfn.nalpha()
nmo = scf_wfn.nmo()
nvirt = nmo - ndocc

# Get orbital energies, cast into Julia array, and separate occupied & virtual
eps = np.asarray(scf_wfn.epsilon_a())
e_ij = eps[1:ndocc]
e_ab = eps[ndocc+1:end]

# Get MO coefficients from SCF wavefunction
C = np.asarray(scf_wfn.Ca())
Cocc = C[:, 1:ndocc]
Cvirt = C[:, ndocc+1:end];
# -

# From the conventional MP2 program, we know that the next step is to obtain the ERIs and transform them into the MO basis using the orbital coefficient matrix, **C**.  In order to do this using density-fitted integrals, we must first build and transform the DF-ERI's similar to that in the density-fitted HF chapter. However, we use an auxiliary basis set that better reproduces the valence electrons important for correlation compared to the JKFIT auxiliary basis of Hartree-Fock. We instead use the RIFIT auxiliary basis.

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

# +
# ==> Density Fitted ERIs <==
# Build auxiliary basis set
aux = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "RIFIT", "aug-cc-pVDZ")

# Build instance of Mints object
orb = scf_wfn.basisset()
mints = psi4.core.MintsHelper(orb)

# Build a zero basis
zero_bas = psi4.core.BasisSet.zero_ao_basis_set()

# Raw 3-index
Ppq = mints.ao_eri(zero_bas, aux, orb, orb)
Ppq = psi4view(Ppq)
Ppq = dropdims(Ppq, dims=1)

# Build and invert the Coulomb metric
metric = mints.ao_eri(zero_bas, aux, zero_bas, aux)
metric.power(-0.5, 1.e-14)
metric = psi4view(metric)
metric = dropdims(metric, dims=(1,3))

@tensor Qpq[Q,p,q] := metric[Q,P] * Ppq[P,p,q];
# -

# Now that we have our three-index integrals, we are able to transform them into the MO basis.  To do this, we can simply use `@tensor` to carry out the transformation in a single step:
# ~~~julia
# # Transform Qpq -> Qmo @ O(N^5)
# @tensor Qmo[Q,i,j] := C[p,i] * Qpq[Q,p,q] * C[q,j]
# ~~~
# This simple transformation appears to have $\mathcal{O}(N^5)$ scaling but is reduced with optimal contraction.  We borrow the idea from conventional MP2 to carry out the transformation in more than one step, saving the intermediates along the way.  Using this approach, we are able to transform the `Qpq` tensors into the MO basis in two successive ${\cal O}(N^4)$ steps. `@tensor` will do this for you. To see how it's done manually, in the cell below, we transform the `Qpq` tensors with this reduced scaling algorithm, and save the occupied-virtual slice of the full `Qmo` tensor:

# +
# ==> Transform Qpq -> Qmo @ O(N^4) <==
@tensor Qmo[Q,i,q] := C[p,i] * Qpq[Q,p,q]
@tensor Qmo[Q,i,j] :=          Qmo[Q,i,q] * C[q,j]

# Get Occupied-Virtual Block
Qmo = Qmo[:, 1:ndocc, ndocc+1:end];
# -

# We are now ready to compute the DF-MP2 correlation energy $E_0^{(2)}$.  One approach for doing this would clearly be to form the four-index OVOV $(ia\mid jb)$ ERI tensor directly [an ${\cal O}(N^5)$ contraction], and proceed exactly as we did for conventional MP2.  This would, however, result in needing to store this entire tensor in memory, which would be prohibitive for large systems/basis sets and would only result in minimal savings.  A more clever (and much less memory-intensive) algorithm can be found by considering the MP2 correlation energy expressions,
#
# \begin{equation}
# E_{\rm 0,\,SS}^{(2)} = \sum_{ij}\sum_{ab}\frac{(ia\mid jb)[(ia\mid jb) - (ib\mid ja)]}{\epsilon_i - \epsilon_a + \epsilon_j - \epsilon_b},\,{\rm and}
# \end{equation}
# \begin{equation}
# E_{\rm 0,\,OS}^{(2)} = \sum_{ij}\sum_{ab}\frac{(ia\mid jb)(ia\mid jb)}{\epsilon_i - \epsilon_a + \epsilon_j - \epsilon_b},
# \end{equation}
#
# for particular values of the occupied orbital indices $i$ and $j$:
#
# \begin{equation}
# E_{\rm 0,\,SS}^{(2)}(i, j) = \sum_{ab}\frac{I_{ab}[I_{ab} - I_{ba}]}{\epsilon_i + \epsilon_j - \boldsymbol{\epsilon}_{ab}}
# \end{equation}
# \begin{equation}
# E_{\rm 0,\,OS}^{(2)}(i, j) = \sum_{ab}\frac{I_{ab}I_{ab}}{\epsilon_i + \epsilon_j - \boldsymbol{\epsilon}_{ab}},
# \end{equation}
#
# for virtual-virtual blocks of the full ERI tensors $I_{ab}$ and a matrix $\boldsymbol{\epsilon}_{ab}$ containing all possible combinations of the virtual orbital energies $\epsilon_a$ and $\epsilon_b$.  These expressions are advantageous because they only call for two-index contractions between the virtual-virtual blocks of the OVOV ERI tensor, and the storage of only the VV-block of this tensor in memory.  Furthermore, the formation of the $I_{ab}$ tensor is also ameliorated, since only the auxiliary-virtual blocks of the three-index `Qmo` tensor must be contracted, which can be done on-the-fly as opposed to beforehand (requiring no storage in memory).  In practice, these expressions can be used within explicit loops over occupied indices $i$ and $j$; therefore the overall scaling of this step is still ${\cal O}(N^5)$ (formation of $I_{ab}$ is ${\cal O}(N^3)$ inside two loops), however the the drastically reduced memory requirements result in this method a significant win over conventional MP2.
#
# One potentially mysterious quantity in the frozen-index expressions given above is the virtual-virtual orbital eigenvalue tensor, **$\epsilon$**.  To build this array, we can again borrow an idea from our implementation of conventional MP2: reshaping and broadcasting.  In the cell below, use these techniques to build the VV $\boldsymbol{\epsilon}_{ab}$ tensor.
#
# Hint: In the frozen-index expressions above, $\boldsymbol{\epsilon}_{ab}$ is *subtracted* from the occupied orbital energies $\epsilon_i$ and $\epsilon_j$.  Therefore, the virtual orbital energies should be added together to have the correct sign!

# ==> Build VV Epsilon Tensor <==
e_vv = e_ab .+ e_ab' ;

# In addition to the memory savings incurred by generating VV-blocks of our ERI tensors on-the-fly, we can exploit the permutational symmetry of these tensors [Sherrill:ERI] to drastically reduce the number of loops (and therefore Qv,Qv contractions!) which are needed to compute the MP2 correlation energy.  To see the relevant symmetry, recall that a spin-free four index ERI over spatial orbitals (written in chemists' notation) is given by
#
# $$(i\,a\mid j\,b) = \int{\rm d}^3{\bf r}_1{\rm d}^3{\bf r}_2\phi_i^*({\bf x}_1)\phi_a({\bf x}_1)\frac{1}{r_{12}}\phi_j^*({\bf x}_2)\phi_b({\bf x}_2)$$
#
# For real orbitals, it is easy to see that $(i\,a\mid j\,b) = (j\,b\mid i\,a)$; therefore, it is unnecessary to iterate over all combinations of $i$ and $j$, since the value of the contractions containing either $(i\,a\mid j\,b)$ or $(j\,b\mid i\,a)$ will be identical.  Therefore, it suffices to iterate over all $i$ and only $j\geq i$.  Then, the "diagonal elements" ($i = j$) will contribute once to each of the same-spin and opposite-spin correlation energies, and the "off-diagonal" elements ($i\neq j$) will contribute twice to each correlation energy due to symmetry.  This corresponds to placing either a 1 or a 2 in the numerator of the energy denominator, i.e., 
#
# \begin{equation}
# E_{denom} = \frac{\alpha}{\epsilon_i + \epsilon_j - \boldsymbol{\epsilon}_{ab}};\;\;\;\alpha = \begin{cases}1;\; i=j\\2;\;i\neq j\end{cases},
# \end{equation}
#
# before contracting this tensor with $I_{ab}$ and $I_{ba}$ to compute the correlation energy.  In the cell below, compute the same-spin and opposite-spin DF-MP2 correlation energies using the frozen-index expressions 3 and 4 above, exploiting the permutational symmetry of the full $(ia\mid jb)$ ERIs.  Then, using the correlation energies, compute the total MP2 energy using the DF-RHF energy we computed above.

# +
function mp2_df()
   mp2_os_corr = 0.0
   mp2_ss_corr = 0.0
   for i in 1:ndocc
       # Get epsilon_i from e_ij
       e_i = e_ij[i]
       
       # Get 2d array Qa for i from Qov
       i_Qa = @view Qmo[:, i, :]
       
       for j in i:ndocc
           # Get epsilon_j from e_ij
           e_j = e_ij[j]
           
           # Get 2d array Qb for j from Qov
           j_Qb = @view Qmo[:, j, :]
           
           # Compute 2d ERI array for fixed i,j from Qa & Qb
           @tensor ij_Iab[a,b] := i_Qa[Q,a] * j_Qb[Q,b]

           # Compute energy denominator
           e_denom = inv.(e_i + e_j .- e_vv)
           if i !== j
                e_denom *= 2
           end

           # Compute SS & OS MP2 Correlation
           mp2_os_corr += sum( ij_Iab .*   ij_Iab             .* e_denom )
           mp2_ss_corr += sum( ij_Iab .*  (ij_Iab - ij_Iab')  .* e_denom )
       end
   end
   mp2_os_corr + mp2_ss_corr
end

# Compute MP2 correlation & total MP2 Energy
mp2_corr = mp2_df()
MP2_E = scf_e + mp2_corr
# -

# ==> Compare to Psi4 <==
psi4.compare_values(psi4.energy("mp2"), MP2_E, 8, "MP2 Energy")

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
# 5. ERI Permutational Symmetries
# 	> [Sherrill:ERI] C. David Sherrill, "Permutational Symmetries of One- and Two-Electron Integrals." Accessed via the web at http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.pdf.
