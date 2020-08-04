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
"""Tutorial: Overlap, Kinetic, and Dipole Integrals"""

__authors__   = ["D. Menendez", "Adam S. Abbott", "Boyi Zhang", "Justin M. Turney"]
__credit__    = ["D. Menendez", "Adam S. Abbott", "Boyi Zhang", "Justin M. Turney"]

__copyright__ = "(c) 2014-2020, The Psi4Julia Developers"
__license__   = "BSD-3-Clause"
__date__      = "2020-08-03"
# -

# # Overlap, Kinetic, and Dipole Integrals
#
# In this tutorial we will compute the overlap and kinetic energy integrals encountered in Hartree-Fock. We will also compute the dipole integrals to obtain the molecular dipole moment. 
#
# ## Recurrence formula for one-electron integrals over Gaussian functions
# The direct calculation of every single integral in our arrays is not only a complicated procedure, but an expensive one too. Here, we will use the Obara-Saika recursion scheme for a simpler and more efficient implementation. For example, for the overlap integrals, given the value of $(s|s)$, one can recursively determine all other overlap integrals.
#
# ### Cartesian Gaussian functions
# Denote the origin of a 3-dimensional cartesian gaussian function by the coordinates $\mathbf{R} = (R_x, R_y, R_z)$.
# Let $\mathbf{r} = (x, y, z)$ be the coordinates of the electron, and $\alpha$ be the orbital exponent. We can now define an *unnormalized* Cartesian Gaussian function as:
#
# \begin{equation}
# \phi(\mathbf r; \alpha, \mathbf n, \mathbf R) = (x - R_x)^{n_x} (y - R_y)^{n_y} (z - R_z)^{n_z} \exp[-\alpha (\mathbf r - \mathbf R)^2]
# \end{equation}
#
# where $\alpha$ is the orbital exponent, and $\mathbf{n} = (n_x, n_y, n_z)$ is the angular momentum index vector. The sum $n_x + n_y + n_z = \lambda$ will hereafter be referred to as the **angular momentum**. We define a **shell** to be a set of functions (*components*) which share the same origin $\mathbf{R}$, and angular momentum $\lambda$.
#
# The shells with $\lambda$  equal to $0, 1, 2,...,$ are referred to as the $s, p, d, ...$ shell. Each shell has $(\lambda + 1) (\lambda + 2)/2$ components. The $s$ shell, with angular momentum $\lambda = 0$ has one component usually designated as $s$. The $p$ shell ($\lambda = 1$) has three components, designated as $p_x, p_y, p_z$. The $d$ shell ($\lambda = 2$) has six components, designated as $d_{xx}, d_{yy}, d_{zz}, d_{xy}, d_{xz}, d_{yz}$.
#
# In quantum chemistry, we typically represent a single component (an **atomic orbital**) by a linear combination of Gaussians ($c_1 \phi_1 + c_2 \phi_2 + c_3 \phi_3 ...$) hereon referred to as **primitives**. For example, the STO-3G basis set uses three primitives for each atomic orbital basis function. Each primitive is weighted by a coefficient $c$.
#
# Using our angular momentum index vector $\mathbf{n}$ we denote the single component of the $s$ shell to have angular momentum index $\mathbf{n} = \mathbf{0} = (0, 0, 0)$. Since the $p$ shell has three components, we may compactly express the angular momentum index vector as $\mathbf{1}_i$ where $i$ may be $x$, $y$, or $z$, and $\mathbf{1}_i = (\delta_{ix}, \delta_{iy}, \delta_{iz})$. For example, $p_x$ may be represented as $\mathbf{1}_x = (1, 0, 0)$. For the six components of the $d$ shell, we require a sum of two angular momentum index vectors $\mathbf{1}_i + \mathbf{1}_j$, where $(i,j = x,y,z)$. In this notation, the $d_{xy}$ component is $\mathbf{1}_x + \mathbf{1}_y = (1,0,0) + (0,1,0) = (1,1,0)$. To obtain higher order angular momentum components, we add the appropriate number of $\mathbf{1}_i$'s ($\mathbf{n}$'s).
#
#
# ### Two-center overlap integrals
# Two-center overlap integrals over unnormalized cartesian gaussian functions are of the form:
# \begin{equation}
# (\mathbf a|\mathbf b) = \int d\mathbf r\ \phi(\mathbf r; \alpha_a, \mathbf a, \mathbf A)\phi(\mathbf r; \alpha_b, \mathbf b, \mathbf B)
# \end{equation}
#
# Given $(\mathbf 0_A | \mathbf 0_B)$, we can use the Obara-Saika recursion relation to obtain overlap integrals between all basis functions. The overlap over $s$ functions is given by:
#
# \begin{equation}
# (\mathbf 0_A | \mathbf 0_B) = \left(\frac{\pi}{\alpha}\right)^{3/2} \exp[-\xi(\mathbf A-\mathbf B)^2]
# \end{equation}
#
# where $\alpha = \alpha_a + \alpha_b$ and $\zeta = \frac{\alpha_a\alpha_b}{\alpha}$.
# The recursion relations are given below. For a full derivation, see the appendix, or the
# [original paper](http://aip.scitation.org/doi/abs/10.1063/1.450106) by Obara and Saika. To increment the left side angular momentum:
#
# \begin{equation}
# (\mathbf a+\mathbf 1_i|\mathbf b) = (\mathbf{P - A})(\mathbf a|\mathbf b) + \frac{1}{2\alpha} N_i(\mathbf a)(\mathbf a-\mathbf 1_i|\mathbf b) + \frac{1}{2\alpha} N_i(\mathbf b)(\mathbf a|\mathbf b-\mathbf 1_i)
# \end{equation}
# and similarily, to increment the right side:
# \begin{equation}
# (\mathbf a|\mathbf b+\mathbf 1_i) = (\mathbf{P - B})(\mathbf a|\mathbf b) + \frac{1}{2\alpha} N_i(\mathbf a)(\mathbf a-\mathbf 1_i|\mathbf b) + \frac{1}{2\alpha} N_i(\mathbf b)(\mathbf a|\mathbf b-\mathbf 1_i)
# \end{equation}
#
#
# where \begin{equation}\mathbf{P} = \frac{\alpha_a \mathbf{A} + \alpha_b \mathbf{B}} {\alpha} \end{equation}
#
# and ${N}_i(\mathbf{a})$, ${N}_i(\mathbf{b})$ are just the angular momenta of $\mathbf{a}$ and $\mathbf{b}$.
#
# To fill in the first row, the seoncd term goes to zero, since we cannot have a negative angular momentum (cannot subtract $\mathbf 1_i$). To fill in the first column, the third term goes to zero for the same reason.
#
#

# # Implementation
# ### Write the recursion function
#
# The Obara-Saika recursion relationships depend on $\mathbf{P-A}$,  $\mathbf{P-B}$, $\alpha$, and the angular momentum values for $\mathbf a$ and $\mathbf b$, which we will denote as `PA`, `PB`, `alpha`, `AMa`, and `AMb`. 
# Let's write a function that takes these parameters and returns three matrices containing the x, y and z components of our unnormalized overlap integrals. These same components also can be used to construct our kinetic energy and dipole integrals later, as we will see.
#
# For now, we will set $(\mathbf 0_A | \mathbf 0_B)$ to $1.0$ for simplicity. Later, we will set the value as defined above. 

# +
using PyCall: pyimport
psi4 = pyimport("psi4")
np   = pyimport("numpy") # used only to cast to Psi4 arrays

function os_recursion(PA, PB, alpha, AMa, AMb)
    if length(PA) != 3 || length(PB) != 3
       error("PA and PB must be xyz coordinates.")
    end
   
    # Allocate space x, y, and z matrices
    # We add one because the equation for the kinetic energy
    # integrals require terms one beyond those in the overlap
    x = zeros(AMa + 1, AMb + 1)
    y = zeros(AMa + 1, AMb + 1)
    z = zeros(AMa + 1, AMb + 1)

    # Define 1/2alpha factor for convenience
    oo2a = 1 / (2alpha)

    # Set initial conditions (0a|0b) to 1.0 for each cartesian component
    x[1, 1] = y[1, 1] = z[1, 1] = 1.0

    
    # BEGIN RECURSION
    # Fill in the [0,1] position with PB
    if AMb > 0
        x[1, 2] = PB[1]
        y[1, 2] = PB[2]
        z[1, 2] = PB[3]
    end

    # Fill in the rest of row zero
    for b in 2:AMb
        x[1, b+1] = PB[1] * x[1, b] + b * oo2a * x[1, b - 1]
        y[1, b+1] = PB[2] * y[1, b] + b * oo2a * y[1, b - 1]
        z[1, b+1] = PB[3] * z[1, b] + b * oo2a * z[1, b - 1]
    end
    
    # Now, we have for each cartesian component
    # | 1.0  PB #  #|
    # |  0   0  0  0|
    # |  0   0  0  0| 
    # |  0   0  0  0|

    # Upward recursion in a for all b's
    # Fill in the [1,0] position with PA
    if AMa > 0                                                 
        x[2, 1] = PA[1]
        y[2, 1] = PA[2]
        z[2, 1] = PA[3]
        
    # Now, we have for each cartesian component
    # | 1.0  PB #  #|
    # |  PA  0  0  0|
    # |  0   0  0  0| 
    # |  0   0  0  0|

        # Fill in the rest of row one
        for b in 2:AMb
            x[2, b] = PA[1] * x[1, b] + b * oo2a * x[1, b - 1]
            y[2, b] = PA[2] * y[1, b] + b * oo2a * y[1, b - 1]
            z[2, b] = PA[3] * z[1, b] + b * oo2a * z[1, b - 1]
        end
            
        # Now, we have for each cartesian component
        # | 1.0  PB #  #|
        # |  PA  #  #  #|
        # |  0   0  0  0| 
        # |  0   0  0  0|

        # Fill in the rest of column 0
        for a in 2:AMa
            x[a + 1, 1] = PA[1] * x[a, 1] + a * oo2a * x[a - 1, 1]
            y[a + 1, 1] = PA[2] * y[a, 1] + a * oo2a * y[a - 1, 1]
            z[a + 1, 1] = PA[3] * z[a, 1] + a * oo2a * z[a - 1, 1]
            
        # Now, we have for each cartesian component
        # | 1.0  PB #  #|
        # |  PA  #  #  #|
        # |  #   0  0  0| 
        # |  #   0  0  0|
    
        # Fill in the rest of the a'th row
            for b in 2:AMb
                x[a + 1, b] = PA[1] * x[a, b] + a * oo2a * x[a - 1, b] + b * oo2a * x[a, b - 1]
                y[a + 1, b] = PA[2] * y[a, b] + a * oo2a * y[a - 1, b] + b * oo2a * y[a, b - 1]
                z[a + 1, b] = PA[3] * z[a, b] + a * oo2a * z[a - 1, b] + b * oo2a * z[a, b - 1]
            end
        end

        # Now, we have for each cartesian component
        # | 1.0  PB #  #|
        # |  PA  #  #  #|
        # |  #   #  #  #| 
        # |  #   #  #  #|
    end
        
    # Return the results
    (x, y, z)
end
# -

# # Overlap Integrals
#
# Now that we have our recursion set up, we are ready to compute the integrals, starting with the overlap integrals. First we need a molecule and a basis set. We do a quick Hartree-Fock computation to obtain a wavefunction for easy access to the basis set. Later, we will also use the wavefunction's density matrix to compute the dipole moment.
#
# By default, Psi4 uses spherical harmonic Gaussians, sometimes referred to as "pure angular momentum", whereas we will be using Cartesian Gaussians. We set `puream` to be off so that we can compare our integrals to Psi4's.

# +
psi4.core.set_output_file("output.dat", false)

mol = psi4.geometry("""
                        O
                        H 1 1.1
                        H 1 1.1 2 104
                        symmetry c1
                        """)

psi4.set_options(Dict("basis"         => "sto-3g",
                      "scf_type"      => "pk",
                      "mp2_type"      => "conv",
                      "puream"        => 0,
                      "e_convergence" => 1e-8,
                      "d_convergence" => 1e-8))

scf_e, scf_wfn = psi4.energy("scf", return_wfn=true)

basis = scf_wfn.basisset()
# -

# The factored form of the two-center overlap integrals is
# \begin{equation}
# (\mathbf a | \mathbf b) = \space I_x(n_{ax},n_{bx},n_{cx})\ I_y(n_{ay},n_{by},n_{cy})\ I_z(n_{az},n_{bz},n_{cz})
# \end{equation}
#
# where $I_x$ is the x component of the unnormalized overlap matrix.
#
# To obtain the overlap matrix, we must loop over all shells, and for each shell, loop over the primitive gaussians.
#
# ~~~julia
# # make space to store the overlap integral matrix
# S = zeros(basis.nao(),basis.nao())
#
# # loop over the shells, basis.nshell is the number of shells
# for i in 1:basis.nshell(), j in 1:basis.nshell()
#     # basis.shell is a shell (1s, 2s, 2p, etc.)
#     # for water, there are 5 shells: (H: 1s, H: 1s, O: 1s, 2s, 2p)
#     ishell = basis.shell(i-1) 
#     jshell = basis.shell(j-1)
#     # each shell has some number of primitives which make up each component of a shell
#     # sto-3g has 3 primitives for every component of every shell.
#     nprimi = ishell.nprimitive 
#     nprimj = jshell.nprimitive
# ~~~
#
# At this point we access the information for each primitive basis set $\mathbf{a}$ and $\mathbf{b}$ in $(\mathbf{a}|\mathbf{b})$. The `basis.shell()` (which we have set to `ishell` and `jshell`) is a `psi4.core.GaussianShell` object, which contains all the information we need about the primitives within a shell, such as the exponent `(shell().exp())`, the coefficient `(shell().coef())`, the angular momentum `(shell().am)` and the atom number it is centered over `(shell().ncenter)`, which we can use to get the coordinates of the primitive center ($\mathbf{A}$ or $\mathbf{B}$) (*whew!*). We store all of these parameters as variables in our for-loops for easy access. Continuing our sample script:
#
# ~~~julia
#     # loop over the primitives within a shell
#     for a in 1:nprimi, b in 1:nprimj
#         expa = ishell.exp(a-1) # primitive exponents
#         expb = jshell.exp(b-1)
#         coefa = ishell.coef(a-1)  # primitive coefficients
#         coefb = jshell.coef(b-1)
#         AMa = ishell.am  # angular momenta
#         AMb = jshell.am
#         # defining centers for each basis function:
#         # mol.x() returns the x coordinate of the atom given by ishell.ncenter
#         # we use this to define a coordinate vector for our centers
#         A = [mol.x(ishell.ncenter), mol.y(ishell.ncenter), mol.z(ishell.ncenter)]
#         B = [mol.x(jshell.ncenter), mol.y(jshell.ncenter), mol.z(jshell.ncenter)]
# ~~~
#
# We are now ready to define `PA` and `PB` which are arguments for our recursion function. 
#                 
#                 
# ~~~julia
#         alpha = expa + expb
#         zeta = (expa * expb) / alpha
#         P = (expa * A + expb * B) / alpha
#         PA = P - A
#         PB = P - B
#         AB = A - B
# ~~~
#
# We also define the value of the $(s|s)$ overlap `start`, originally set to 1.0 in the recursion function.
#
# \begin{equation} 
# s = (\mathbf 0_A | \mathbf 0_B) = \left(\frac{\pi}{\alpha}\right)^{3/2} \exp[-\zeta(\mathbf A-\mathbf B)^2] 
# \end{equation}
#
# ~~~julia
#        start = (π / alpha)^(3 / 2) * exp(-zeta * (AB[1]^2 + AB[2]^2 + AB[3]^2))
# ~~~
#
# Form the x-component, y-component, and z-component overlap matrices with our recursion function. Recall that we set $\mathbf{(0|0)} = 1.0$ and have ignored using the coefficients thus far. We will account for this later when we build each element of our overlap matrix.
#
# ~~~julia
#        # call the recursion
#        x, y, z = os_recursion(PA, PB, alpha, AMa+1, AMb+1)
# ~~~
#
#
# We are now ready to assign the elements of our overlap matrix. Currently, we are only looping over the shells, but not the individual components of each shell (e.g. we are not distinguishing between $p_x$, $p_y$, and $p_z$). Given the total angular momentum of **a** (`AMa`) and **b** (`AMb`), we loop over all possible allocations of this total angular momentum to each angular momentum index in $\mathbf{n} = (n_x, n_y, n_z)$ which we denote as `l`, `m`, and `n`.
#
# ~~~julia
#         # Basis function index where the shell begins
#         i_idx = ishell.function_index  
#         j_idx = jshell.function_index
#         
#         # We use counters to keep track of which component (e.g., p_x, p_y, p_z)
#         # within the shell we are on
#         counta = 1
#         
#         for p in 1:AMa
#             la = AMa - p                    # Let l take on all values, and p be the leftover a.m.
#             for q in 1:p
#                 ma = p - q                  # distribute all leftover a.m. to m and n
#                 na = q
#                 countb = 1
#                 for r in 1:AMb
#                     lb = AMb - r            # Let l take on all values, and r the leftover a.m.
#                     for s in 1:r
#                         mb = r - s          # distribute all leftover a.m. to m and n
#                         nb = s
#                         
#                         # set the value in the full overlap matrix
#                         S[i_idx + counta, j_idx + countb] += start    *
#                                                              coefa    *
#                                                              coefb    *
#                                                              x[la+1,lb+1] *
#                                                              y[ma+1,mb+1] *
#                                                              z[na+1,nb+1]
#
#                         countb += 1
#                     end
#                 end
#                 counta += 1
#             end
#         end
#     end
# end
# ~~~

# ## Computing the Full Overlap Integral Matrix
#
# Putting all the above code snippets together we form our full overlap matrix:

# +
# make space to store the overlap integral matrix
S = zeros(basis.nao(),basis.nao())

# loop over the shells, basis.nshell is the number of shells
for i in 1:basis.nshell(), j in 1:basis.nshell()
    # basis.shell is a shell (1s, 2s, 2p, etc.)
    # for water, there are 5 shells: (H: 1s, H: 1s, O: 1s, 2s, 2p)
    ishell = basis.shell(i-1) 
    jshell = basis.shell(j-1)
    # each shell has some number of primitives which make up each component of a shell
    # sto-3g has 3 primitives for every component of every shell.
    nprimi = ishell.nprimitive 
    nprimj = jshell.nprimitive
    # loop over the primitives within a shell
    for a in 1:nprimi, b in 1:nprimj
        expa = ishell.exp(a-1) # exponents
        expb = jshell.exp(b-1)
        coefa = ishell.coef(a-1)  # coefficients
        coefb = jshell.coef(b-1)
        AMa = ishell.am  # angular momenta
        AMb = jshell.am
        # defining centers for each basis function 
        # mol.x() returns the x coordinate of the atom given by ishell.ncenter
        # we use this to define a coordinate vector for our centers
        A = [mol.x(ishell.ncenter), mol.y(ishell.ncenter), mol.z(ishell.ncenter)]
        B = [mol.x(jshell.ncenter), mol.y(jshell.ncenter), mol.z(jshell.ncenter)]

        alpha = expa + expb
        zeta = (expa * expb) / alpha
        P = (expa * A + expb * B) / alpha
        PA = P - A
        PB = P - B
        AB = A - B
        start = (π / alpha)^(3 / 2) * exp(-zeta * (AB[1]^2 + AB[2]^2 + AB[3]^2))
       
        # call the recursion
        x, y, z = os_recursion(PA, PB, alpha, AMa+1, AMb+1)

        
        # Basis function index where the shell begins
        i_idx = ishell.function_index  
        j_idx = jshell.function_index
        
        # We use counters to keep track of which component (e.g., p_x, p_y, p_z)
        # within the shell we are on
        counta = 1
        
        for p in 1:AMa
            la = AMa - p                    # Let l take on all values, and p be the leftover a.m.
            for q in 1:p
                ma = p - q                  # distribute all leftover a.m. to m and n
                na = q
                countb = 1
                for r in 1:AMb
                    lb = AMb - r            # Let l take on all values, and r the leftover a.m.
                    for s in 1:r
                        mb = r - s          # distribute all leftover a.m. to m and n
                        nb = s
                        
                        # set the value in the full overlap matrix
                        S[i_idx + counta, j_idx + countb] += start    *
                                                             coefa    *
                                                             coefb    *
                                                             x[la+1,lb+1] *
                                                             y[ma+1,mb+1] *
                                                             z[na+1,nb+1]

                        countb += 1
                    end
                end
                counta += 1
            end
        end
    end
end
# -

# Check our overlap against Psi4:

mints = psi4.core.MintsHelper(basis)
Spsi4 = np.asarray(mints.ao_overlap())
@assert np.allclose(S, Spsi4, 6)

#
# # Kinetic energy integrals 
# Before proceeding to an analysis of the kinetic energy integral, it will prove convenient to establish a short-hand notation for integrals related to the overlap integral:
# \begin{equation}
# (0 \mid 0) = \int G(\alpha_a, \mathbf A, l_a, m_a, n_a) G(\alpha_b, \mathbf B, l_b, m_b, n_b)\ d\tau
# \end{equation}
# The symbol $(+1 \mid 0)_{\mathrm{x}}$ will denote an integral of the form given by the above equation, except that the quantum number $l_a$ has been incremented by 1. Similarly, $(0 \mid +1)_{\mathrm{x}}$ increments $l_b$ by 1. Quantum numbers $m$ and $n$ are incremented in the same way with subscripts $y$ and $z$.
#
# The kinetic energy integral is defined as
#
# \begin{equation}
# T = - \frac{1}{2}  \int G(\alpha_a, \mathbf A, l_a, m_a, n_a) \left(\frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} + \frac{\partial^2}{\partial z^2}\right) G(\alpha_b, \mathbf B, l_b, m_b, n_b)\ d\tau
# \end{equation}
#
# The equation for this integral is "unsymmetric" because the quantum numbers of the Gaussian-type function centered on $\mathbf B$ are altered by the Laplacian while those of the Gaussian-type function centered on $\mathbf A$ are not. The symmetric $x$, $y$, and $z$ components are given by:
#
#
# \begin{equation}
# T_x = \frac{1}{2}\Big\{l_a l_b (-1|-1)_{\mathrm{x}} + 4\alpha_a \alpha_b (+1|+1)_{\mathrm{x}} - 2\alpha_a l_b (+1|-1)_{\mathrm{x}} - 2\alpha_b l_a (-1|+1)_{\mathrm{x}}\Big\}
# \end{equation}
#
# \begin{equation}
# T_y = \frac{1}{2}\Big\{l_a l_b (-1|-1)_{\mathrm{y}} + 4\alpha_a \alpha_b (+1|+1)_{\mathrm{y}} - 2\alpha_a l_b (+1|-1)_{\mathrm{y}} - 2\alpha_b l_a (-1|+1)_{\mathrm{y}}\Big\}
# \end{equation}
#
# \begin{equation}
# T_z = \frac{1}{2}\Big\{l_a l_b (-1|-1)_{\mathrm{z}} + 4\alpha_a \alpha_b (+1|+1)_{\mathrm{z}} - 2\alpha_a l_b (+1|-1)_{\mathrm{z}} - 2\alpha_b l_a (-1|+1)_{\mathrm{z}}\Big\}
# \end{equation}
#
# and the full kinetic energy integral is:
#
# \begin{equation}
# T =  (T_x + T_y + T_z)
# \end{equation}
#
#
# Within our same for-loop structure as for the overlap integrals, we can compute the kinetic energy integrals with:
#
# ~~~julia
# Tx = (1 / 2) * (la * lb * x[lam, lbm] + 4 * expa * expb * x[la + 2, lb + 2] -
#        2 * expa * lb * x[la + 2, lbm] - 2 * expb * la * x[lam, lb + 2])   *
#        y[ma+1, mb+1] * z[na+1, nb+1]
#
# Ty = (1 / 2) * (ma * mb * y[mam, mbm] + 4 * expa * expb * y[ma + 2, mb + 2] -
#        2 * expa * mb * y[ma + 2, mbm] - 2 * expb * ma * y[mam, mb + 2])   *
#        x[la+1, lb+1] * z[na+1, nb+1]
#
# Tz = (1 / 2) * (na * nb * z[nam, nbm] + 4 * expa * expb * z[na + 2, nb + 2] -
#        2 * expa * nb * z[na + 2, nbm] - 2 * expb * na * z[nam, nb + 2])   *
#        x[la+1, lb+1] * y[ma+1, mb+1]
#
# # incorporate the value of (0|0) and coefficients
#
# T[i_idx + counta, j_idx + countb] += start * coefa * coefb * (Tx + Ty + Tz)
# ~~~
#
# # Dipole moment integrals
# We discuss these non-energy integrals here because they are frequently used and closely related to overlap and kinetic energy integrals. The dipole moment is defined with respect to a point in space $\mathbf C$, which is almost always taken to be the center of mass. Fortunately for our purposes, Psi4 automatically moves the molecule to be centered on the center of mass and thus $\mathbf C$ is $\mathbf 0$. The dipole moment integral is written for the $x$ direction as
#
# \begin{equation}
# d_{\mathrm{x}} =  \int G(\alpha_a, \mathbf A, l_a, m_a, n_a) G(\alpha_b, \mathbf B, l_b, m_b, n_b) \mathrm{x_\mathbf{C}}\ d\tau
# \end{equation}
#
# and similarly for the operators $\mathrm{y_\mathbf{C}}$ and $\mathrm{z_\mathbf{C}}$. Here, $x_\mathbf{C}$ is
#
# \begin{equation}
# \mathrm{x_\mathbf{C}} = \mathrm{x} - \mathrm{C_x} 
# \end{equation}
#
#
# A convenient procedure is to redefine $\mathrm{x_\mathbf{C}}$ in terms of  $\mathrm{x_\mathbf{A}}$ or $\mathrm{x_\mathbf{B}}$ (we will use $\mathrm{x_\mathbf{A}}$):
#
# \begin{equation}
# \mathrm{x_\mathbf{C}}= (\mathrm{x - A_x}) + (\mathrm{A_x - C_x}) = \mathrm{x_\mathbf{A}} + (\mathrm{A_x - C_x})
# \end{equation}
#
# and we find
#
# \begin{equation}
# d_\mathrm{x} = (+1|0)_{\mathrm{x}} + (\mathrm{A_x - C_x}) (0|0)
# \end{equation}
#
# and the expressions are similar for $d_{\mathrm{y}}$ and $d_{\mathrm{z}}$. Putting this into code:
#
#
# ~~~julia
# dx = (x[la + 2, lb+1] + A[1] * x[la+1, lb+1]) * y[ma+1, mb+1] * z[na+1, nb+1]
# dy = (y[ma + 2, mb+1] + A[2] * y[ma+1, mb+1]) * x[la+1, lb+1] * z[na+1, nb+1]
# dz = (z[na + 2, nb+1] + A[3] * z[na+1, nb+1]) * x[la+1, lb+1] * y[ma+1, mb+1]
#
# # incorporate the (0|0) value and the coefficients:
#
# Dx[i_idx + counta, j_idx + countb] += start * coefa * coefb * dx
# Dy[i_idx + counta, j_idx + countb] += start * coefa * coefb * dy
# Dz[i_idx + counta, j_idx + countb] += start * coefa * coefb * dz
# ~~~
#
#

# # Full Implementation
#
# Putting everything together, we get the overlap, kinetic, and dipole integrals:

# +
# make space to store the overlap, kinetic, and dipole integral matrices
S = zeros(basis.nao(),basis.nao())
T = zeros(basis.nao(),basis.nao())
Dx = zeros(basis.nao(),basis.nao())
Dy = zeros(basis.nao(),basis.nao())
Dz = zeros(basis.nao(),basis.nao())

# loop over the shells, basis.nshell is the number of shells
for i in 1:basis.nshell(), j in 1:basis.nshell()
    # basis.shell is a shell (1s, 2s, 2p, etc.)
    # for water, there are 5 shells: (H: 1s, H: 1s, O: 1s, 2s, 2p)
    ishell = basis.shell(i-1) 
    jshell = basis.shell(j-1)
    # each shell has some number of primitives which make up each component of a shell
    # sto-3g has 3 primitives for every component of every shell.
    nprimi = ishell.nprimitive 
    nprimj = jshell.nprimitive
    # loop over the primitives within a shell
    for a in 1:nprimi, b in 1:nprimj
        expa = ishell.exp(a-1) # exponents
        expb = jshell.exp(b-1)
        coefa = ishell.coef(a-1)  # coefficients
        coefb = jshell.coef(b-1)
        AMa = ishell.am  # angular momenta
        AMb = jshell.am
        # defining centers for each basis function 
        # mol.x() returns the x coordinate of the atom given by ishell.ncenter
        # we use this to define a coordinate vector for our centers
        A = [mol.x(ishell.ncenter), mol.y(ishell.ncenter), mol.z(ishell.ncenter)]
        B = [mol.x(jshell.ncenter), mol.y(jshell.ncenter), mol.z(jshell.ncenter)]
        alpha = expa + expb
        zeta = (expa * expb) / alpha
        P = (expa * A + expb * B) / alpha
        PA = P - A
        PB = P - B
        AB = A - B
        start = (π / alpha)^(3 / 2) * exp(-zeta * (AB[1]^2 + AB[2]^2 + AB[3]^2))
        # call the recursion
        x, y, z = os_recursion(PA, PB, alpha, AMa+1, AMb+1)

        
        # Basis function index where the shell begins
        i_idx = ishell.function_index  
        j_idx = jshell.function_index
        
        # We use counters to keep track of which component (e.g., p_x, p_y, p_z)
        # within the shell we are on
        counta = 1
        
        for p in 1:AMa
            la = AMa - p                    # Let l take on all values, and p be the leftover a.m.
            for q in 1:p
                ma = p - q                  # distribute all leftover a.m. to m and n
                na = q
                countb = 1
                for r in 1:AMb
                    lb = AMb - r            # Let l take on all values, and r the leftover a.m.
                    for s in 1:r
                        mb = r - s          # distribute all leftover a.m. to m and n
                        nb = s
                        
                        S[i_idx + counta, j_idx + countb] += start    *
                                                             coefa    *
                                                             coefb    *
                                                             x[la+1,lb+1] *
                                                             y[ma+1,mb+1] *
                                                             z[na+1,nb+1] 
                                            
                        lam = la < 1 ? lastindex(x,1) : la
                        lbm = lb < 1 ? lastindex(x,1) : lb
                        nam = na < 1 ? lastindex(x,1) : na
                        nbm = nb < 1 ? lastindex(x,1) : nb
                        mam = ma < 1 ? lastindex(x,1) : ma
                        mbm = mb < 1 ? lastindex(x,1) : mb
                        Tx = (1 / 2) * (la * lb * x[lam, lbm] + 4 * expa * expb * x[la + 2, lb + 2] -
                               2 * expa * lb * x[la + 2, lbm] - 2 * expb * la * x[lam, lb + 2])   *
                               y[ma+1, mb+1] * z[na+1, nb+1]

                        Ty = (1 / 2) * (ma * mb * y[mam, mbm] + 4 * expa * expb * y[ma + 2, mb + 2] -
                               2 * expa * mb * y[ma + 2, mbm] - 2 * expb * ma * y[mam, mb + 2])   *
                               x[la+1, lb+1] * z[na+1, nb+1]

                        Tz = (1 / 2) * (na * nb * z[nam, nbm] + 4 * expa * expb * z[na + 2, nb + 2] -
                               2 * expa * nb * z[na + 2, nbm] - 2 * expb * na * z[nam, nb + 2])   *
                               x[la+1, lb+1] * y[ma+1, mb+1]

                        T[i_idx + counta, j_idx + countb] += start * coefa * coefb * (Tx + Ty + Tz)


                        dx = (x[la + 2, lb+1] + A[1] * x[la+1, lb+1]) * y[ma+1, mb+1] * z[na+1, nb+1]
                        dy = (y[ma + 2, mb+1] + A[2] * y[ma+1, mb+1]) * x[la+1, lb+1] * z[na+1, nb+1]
                        dz = (z[na + 2, nb+1] + A[3] * z[na+1, nb+1]) * x[la+1, lb+1] * y[ma+1, mb+1]

                        Dx[i_idx + counta, j_idx + countb] += start * coefa * coefb * dx
                        Dy[i_idx + counta, j_idx + countb] += start * coefa * coefb * dy
                        Dz[i_idx + counta, j_idx + countb] += start * coefa * coefb * dz
                        
                        countb += 1
                    end
                end
                counta += 1
            end
        end
    end
end
# -

# Check the kinetic energy integrals:

Tpsi4 = np.asarray(mints.ao_kinetic())
@assert np.allclose(T, Tpsi4, 6)

# # Computing the Dipole Moment

# The x-component of the dipole moment is given by
#
# \begin{align}
# \mu_x =& - \mu_{\mathrm{elec}} + \mu_{\mathrm{nuclear}}\\
# \mu_x =& - \sum_{\mu \nu}^{\mathrm{AO}} D_{\mu \nu} d_{\mu \nu}^{x} + \sum_A^N Z_A X_A
# \end{align}
#
# You can obtain the nuclear contribution to the dipole $\mu_{\mathrm{nuclear}}$ from Psi4 using
#
#     psi4.core.molecule.nuclear_dipole()
#
# The total dipole moment is given by
#
# \begin{equation}
# \mu = \sqrt{\mu_x^2 + \mu_y^2 + \mu_z^2}
# \end{equation}

# +
#Code to compute dipole moment

# Get density matrices
Da = np.asarray(scf_wfn.Da())
Db = np.asarray(scf_wfn.Db())
D = Da + Db

# Get nuclear dipole
nuc_dipole = mol.nuclear_dipole()

# Compute dipole moment components
mux = -sum(D .* Dx) + get(nuc_dipole,1)
muy = -sum(D .* Dy) + get(nuc_dipole,2)
muz = -sum(D .* Dz) + get(nuc_dipole,3)

# Compute the dipole moment
mu = sqrt(mux^2 + muy^2 + muz^2)

# Get Psi4's dipole moment components
mux_psi4 = psi4.core.variable("SCF DIPOLE X")
muy_psi4 = psi4.core.variable("SCF DIPOLE Y")
muz_psi4 = psi4.core.variable("SCF DIPOLE Z")

# Compute Psi4 dipole moment
mu_psi4 = sqrt(mux_psi4^2 + muy_psi4^2 + muz_psi4^2)

# Psi4 prints in Debye, ours is in a.u.
# 2.54174623 Debye in 1 a.u.
mu_psi4 *= 1/2.54174623

# Compare Psi4 dipole moment to ours

@assert np.allclose(mu, mu_psi4, 4)

println("Psi4 calculated molecular dipole magnitude: ", mu_psi4)
println("Here calculated molecular dipole magnitude: ", mu)
# -

# # References
#
# 1. Recursion scheme:
#     > [[Obara:1986](http://aip.scitation.org/doi/abs/10.1063/1.450106)] S. Obara and A. Saika, *J. Chem. Phys* **84**, 3963 (1986)
#
# 2. T. Helgaker, P. Jorgensen, and J. Olsen, *Molecular Electronic Structure Theory*, John Wiley & Sons Inc, 2000.
