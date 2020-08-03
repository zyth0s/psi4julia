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
Tutorial: A reference implementation of configuration interactions singles.
"""

__authors__   = ["D. Menendez", "Boyi Zhang", "Adam S. Abbott"]
__credits__   = ["D. Menendez", "Boyi Zhang", "Adam S. Abbott", "Justin M. Turney"]

__copyright_amp__ = "(c) 2014-2020, The Psi4Julia Developers"
__license__   = "BSD-3-Clause"
__date__      = "2020-08-03"
# -

# # Configuration Interaction Singles (CIS) 

# ## I. Theoretical Overview

# In this tutorial, we will implement the configuration interaction singles method in the spin orbital notation. The groundwork for working in the spin orbital notation has been laid out in "Introduction to the Spin Orbital Formulation of Post-HF methods" [tutorial](../08_CEPA0_and_CCD/8a_Intro_to_spin_orbital_postHF.ipynb). It is highly recommended to work through that introduction before starting this tutorial. 

# ### Configuration Interaction (CI)
#
# The configuration interaction wavefunction is constructed as a linear combination of the reference determinants and all singly, doubly, ... n-tuple excited determinants where n is the number of electrons in a given system: 
#
# \begin{equation}
# \Psi_\mathrm{CI} = (1 + \hat{C_1} + \hat{C_2} + ...\hat{C_n)}\Phi
# \end{equation}
#
# Here, $\hat{C_n}$ is the n configuration excitation operator. 
#
# In Full CI, all possible excitations are included in the wavefunction expansion. In truncated CI methods, only a subset of excitations are included. 

# ## CIS
#
# In CIS, only single excitations from the occupied (indices i,j,k...) to the virtual (indices a,b,c...) orbitals are included. As a result, CIS gives transition energies to an excited state. 
#
# Assuming we are using canonical Hartree-Fock spin orbitals($\{\mathrm{\psi_p}\}$) with orbital energies $\{\epsilon_p\}$, we can build a shifted CIS Hamiltonian matrix:
#
# \begin{equation}
# \tilde{\textbf{H}} = \textbf{H} - E_0 \textbf{I} = [\langle \Phi_P | \hat{H_e} - E_0|\Phi_Q \rangle],\, 
# \Phi_P \in {\Phi_i^a}
# \end{equation}
#
# where $E_0$ is the ground state Hartree-Fock state energy given by $\langle \Phi | \hat{H_e}|\Phi \rangle$.
#
# The matrix elements of this shifted CIS Hamiltonian matrix can be evaluated using Slater's rules to give:
#
# \begin{equation}
# \langle \Phi_i^a | \hat{H_e} - E_0|\Phi_j^b \rangle = (\epsilon_a - \epsilon_i)\delta_{ij} \delta_{ab}
# + \langle aj || ib \rangle
# \end{equation}
#
# This then becomes a standard eigenvalue equation from which we can solve for the excitation energies and the wavefunction expansion coefficients:
#
# \begin{equation}
# \tilde{\textbf{H}} \textbf{c}_K = \Delta E_K\textbf{c}_K, \,\Delta E_K = E_K - E_0
# \end{equation}
#
#

# ## II. Implementation

# As with previous tutorials, let's begin by importing Psi4, NumPy, and TensorOperations and setting memory and output file options.

# +
# ==> Import Psi4, NumPy, & TensorOperations <==
using PyCall: pyimport
psi4 = pyimport("psi4")
np   = pyimport("numpy") # used only to cast to Psi4 arrays
using TensorOperations: @tensor

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

psi4.set_options(Dict("basis"         => "sto-3g",
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
    identity = [ 1.0 0.0; 0.0 1.0]
    I = np.kron(identity, I)
    np.kron(identity, permutedims(I, reverse(1:4)))
end
 
I = np.asarray(mints.ao_eri())
I_spinblock = spin_block_tei(I)
 
# Convert chemist's notation to physicist's notation, and antisymmetrize
# (pq | rs) ---> <pr | qs>
# <pr||qs> = <pr | qs> - <pr | sq>
gao = permutedims(I_spinblock, (1, 3, 2, 4)) - permutedims(I_spinblock, (1, 3, 4, 2));
# -

# We get the orbital energies from alpha and beta electrons and append them together. We spin-block the coefficients obtained from the reference wavefunction and convert them into NumPy arrays. There is a set corresponding to coefficients from alpha electrons and a set of coefficients from beta electrons. We then sort them according to the order of the orbital energies using argsort():

# +
# Get orbital energies, cast into NumPy array, and extend eigenvalues
eps_a = np.asarray(scf_wfn.epsilon_a())
eps_b = np.asarray(scf_wfn.epsilon_b())
eps = vcat(eps_a, eps_b)

# Get coefficients, block, and sort
Ca = np.asarray(scf_wfn.Ca())
Cb = np.asarray(scf_wfn.Cb())
C = [Ca zero(Ca); zero(Cb) Cb]; # direct sum

# Sort the columns of C according to the order of orbital energies
C = C[:, sortperm(eps)] 

# Sort orbital energies
sort!(eps);
# -

# We now transform the 2-electron integral from the AO basis into the MO basis using the coefficients:
#

# Transform gao, which is the spin-blocked 4d array of physicist's notation,
# antisymmetric two-electron integrals, into the MO basis using MO coefficients
gmo = @tensor begin
   gmo[P,Q,R,S] := gao[p,Q,R,S] * C[p,P]
   gmo[p,Q,R,S] := gmo[p,q,R,S] * C[q,Q]
   gmo[p,q,R,S] := gmo[p,q,r,S] * C[r,R]
   gmo[p,q,r,S] := gmo[p,q,r,s] * C[s,S]
end
nothing

# Now that we have our integrals, coefficents, and orbital energies set up in with spin orbitals, we can start our CIS procedure. We first start by initializing the shifted Hamiltonion matrix $\tilde{\textbf{H}}$ (`HCIS`). Let's think about the size of $\tilde{\textbf{H}}$. We need all possible single excitations from the occupied to virtual orbitals. This is given by the number of occupied orbitals times the number of virtual orbitals  (`nocc * nvirt`).
#
# The size of our matrix is thus `nocc * nvirt` by `nocc * nvirt`. 

# Initialize CIS matrix.
# The dimensions are the number of possible single excitations
HCIS = zeros(nocc * nvirt, nocc * nvirt);

# Next, we want to build all possible excitations from occupied to virtual orbitals. We create two for-loops that will loop over the number of occupied orbitals and number of virtual orbitals, respectively, and store the combination of occupied and virtual indices as a tuple `(i, a)`. We put all tuples in a list called `excitations`. 

# Build the possible excitations, collect indices into a list
excitations = []
for i in 1:nocc, a in nocc+1:nso
    push!(excitations,(i, a))
end

# Now we can evaluate the matrix elements of the shifted CIS Hamiltonian matrix using the equation given above. For each element, there are several layers of indexing that we have to consider. 
# First, there are the indices of the element itself, which gives the position of the element in the matrix. Indices `p` and `q` are used:
#
# `HCIS[p, q]`
#
# Second, there are two sets of excitations from occupied to virtual orbitals corresponding to the bra and ket of each matrix element. For these, we will take advantage of the `excitations` list that we build with the list of all possible excitations. We will use indices `i` and `a` to denote the excitation in the bra (`left_excitation`) and `j` and `b` to denote the excitation in the ket (`right_excitation`). 
#
# To manage these indices, we will use the `enumerate` function.
#
# Note that a Kronecker delta $\delta_{pq}$ can be represented as `p == q`. 
#

# Form matrix elements of shifted CIS Hamiltonian
for (p, left_excitation) in enumerate(excitations)
    i, a = left_excitation
    for (q, right_excitation) in enumerate(excitations)
        j, b = right_excitation
        HCIS[p, q] = (eps[a] - eps[i]) * (i == j) * (a == b) + gmo[a, j, i, b]
    end
end

# We now use the composed function `eigen ∘ Hermitian` (for hermitian matrices) to diagonalize the shifted CIS Hamiltonian. This will give us the excitation energies (`ECIS`). These eigenvalues correspond to the CIS total energies for various excited states. The columns of matrix `CCIS` give us the coefficients which describe the relative contribution of each singly excited determinant to the excitation energy. 
#

# +
# Diagonalize the shifted CIS Hamiltonian
using LinearAlgebra: eigen, Hermitian

ECIS, CCIS = (eigen ∘ Hermitian)(HCIS) ;
# -

# For a given excitation energy, each coefficent in the linear combination of excitations represents the amount that a particular excitation contributes to the overall excitation energy. The percentage contribution of each coefficient can be calculated by squaring the coefficent and multiplying by 100. 

# Percentage contributions for each state vector
percent_contrib = @. round(CCIS^2 * 100);

# In addition to excitation energies, we want to print the excitations that contribute 10% or more to the overall energy, as well as their percent contribution. 
#
# Note that `printfmt` allows us to print different sections to the same line without a line break.

# Print detailed information on significant excitations
using Formatting: printfmt
println("CIS:")
for state in eachindex(ECIS)
    # Print state, energy
    printfmt("State {1:3d} Energy (Eh) {2:10.7f} ", state, ECIS[state])
    for (idx, excitation) in enumerate(excitations)
        if percent_contrib[idx, state] > 10
            i, a = excitation
            # Print percentage contribution and the excitation
            printfmt("{1:4d}% {2:2d} -> {3:2d} ", percent_contrib[idx, state], i, a)
        end
    end
    printfmt("\n")
end

# ## References
# 1. Background paper:
#  >"Toward a systematic molecular orbital theory for excited states"
# [[Foresman:1992:96](http://pubs.acs.org/doi/abs/10.1021/j100180a030)] J. B. Foresman, M. Head-Gordon, J. A. Pople, M. J. Frisch, *J. Phys. Chem.* **96**, 135 (1992).
#
#
# 2. Algorithms from: 
# 	> [[CCQC:CIS](https://github.com/CCQC/summer-program/tree/master/7)] CCQC Summer Program, "CIS" accessed with https://github.com/CCQC/summer-program/tree/master/7.
#     
