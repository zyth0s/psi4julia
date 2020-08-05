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

# # DFT: The LDA kernel
# ## I. Theory
#
# Previously we described the DFT Fock matrix as
# $$F^{DFT}_{\mu\nu} = H_{\mu\nu} + 2J[D]_{\mu\nu} - \zeta K[D]_{\mu\nu} + V^{\rm{xc}}_{\mu\nu}$$
# upon examination it is revealed that the only quantity that we cannot yet compute is $V^{\rm{xc}}$. 
#
# Here we will explore the local density approximation (LDA) functionals where $V^{\rm{xc}} = f[\rho(\hat{r})]$. For these functionals the only required bit of information is the density at the grid point. As we discussed the grid last chapter we will now focus exactly on how to obtain the density on the grid.
#
# Before we begin we should first recall that the Fock matrix is the derivative of the energy with respect to atomic orbitals. Therefore, the $V^{\rm{xc}}$ matrix is not the XC energy, but the derivate of that energy, which can expressed as $\frac{\partial e_{\rm{xc}}}{\partial\rho}$. 

# +
using PyCall: pyimport
psi4 = pyimport("psi4")
np   = pyimport("numpy") # used only to cast to Julia Arrays

mol = psi4.geometry("""
He
symmetry c1
""")
psi4.set_options(Dict("BASIS" => "cc-pvdz",
                      "DFT_SPHERICAL_POINTS" => 6,
                      "DFT_RADIAL_POINTS" => 5))

psi4.core.set_output_file("output.dat", false)

svwn_w, wfn = psi4.energy("SVWN", return_wfn=true)
Vpot = wfn.V_potential()
# -

# ## 2. Density on a Grid
# The density on the grid can be expressed as
# $$\rho(\hat{r}) = \sum\limits_{\mu\nu} D_{\mu\nu}\;\phi_\mu(\hat{r})\phi_\nu(\hat{r})$$
#
# Recall that we compute DFT quanties on a grid, so $\hat{r}$ will run over a grid instead of all space. Using this we can build collocation matrices that map between atomic orbital and grid space $$\phi_\mu(\hat{r}) \rightarrow \phi_\mu^p$$
# where our $p$ index will be the index of individual grid points. Our full expression becomes:
#
# $$\rho_p = \phi_\mu^p D_{\mu\nu} \phi_\nu^p$$
#
# To compute these quantities let us first remember that the DFT grid is blocked loosely over atoms. It should now be apparent to why we do this, consider the $\phi_\mu^p$ objects. The total size of this object would be `nbf` $\times$ `npoints`. To put this in perspective a moderate size molecule could have 1e4 basis functions and 1e8 grid points, so about 8 terabytes of data! As this object is very sparse it is much more convenient to store the grid and compute $\phi\mu^p$ matrices on the fly. 
#
# We then need object to compute $\phi_\mu^p$. 

# +
# Grab a "points function" to compute the Phi matrices
points_func = Vpot.properties()[1]

# Grab a block and obtain its local mapping
block = Vpot.get_block(1)
npoints = block.npoints()
lpos = np.array(block.functions_local_to_global()) .+ 1
println("Local basis function mapping")
display(lpos)

# Copmute phi, note the number of points and function per phi changes.
phi = np.array(points_func.basis_values()["PHI"])[1:npoints, 1:size(lpos)[1]]
println("\nPhi Matrix")
display(phi)
# -

# ## 3. Evaluating the kernel
#
# After building the density on the grid we can then compute the exchange-correlation $f_{xc}$ at every gridpoint. This then need to be reintegrated back to atomic orbital space which can be accomplished like so:
#
# $$V^{\rm{xc}}_{pq}[D_{pq}] = \phi_\mu^a\;\phi_\nu^a\;\; w^a\;f^a_{\rm{xc}}{(\phi_\mu^p D_{\mu\nu} \phi_\nu^p)}$$
#
# Where $w^a$ is our combined Truetler and Lebedev weight at every point.
#
# Unlike SCF theory where the SCF energy can be computed as the sum of the Fock and Density matrices the energy for XC kernels must be computed in grid space. Fortunately, the energy is simply defined as:
#
# $$e_{\rm{xc}} = w^a f^a_{\rm{xc}}$$
#
# We can now put all the pieces together to compute $e_{\rm{xc}}$ and $\frac{\partial E_{\rm{xc}}}{\partial\rho}= V^{\rm{xc}}$.

# +
D = np.array(wfn.Da())

V = zero(D)

rho = []
points_func = Vpot.properties()[1]
superfunc = Vpot.functional()

xc_e, V = let xc_e = 0.0, rho = rho
   # Loop over the blocks
   for b in 1:Vpot.nblocks()
       
       # Obtain block information
       block = Vpot.get_block(b-1)
       points_func.compute_points(block)
       npoints = block.npoints()
       lpos = np.array(block.functions_local_to_global()) .+ 1
       
       # Obtain the grid weight
       w = np.array(block.w())

       # Compute ϕᴾμ!
       phi = np.array(points_func.basis_values()["PHI"])[1:npoints, 1:size(lpos)[1]]
       
       # Build a local slice of D
       lD = D[lpos,lpos]
       
       # Copmute ρ
       # ρᴾ = ϕᴾμ Dμν ϕᴾν
       rho = 2vec(sum( (phi * lD) .* phi, dims=2))

       inp = Dict()
       inp["RHO_A"] = psi4.core.Vector.from_array(rho)
       
       # Compute the kernel
       ret = superfunc.compute_functional(inp, -1)
       
       # Compute the XC energy
       vk = np.array(ret["V"])[1:npoints]
       xc_e = w .* vk
           
       # Compute the XC derivative.
       v_rho_a = np.array(ret["V_RHO_A"])[1:npoints]
       # Vab = ϕᴾ_b Vᴾ Wᴾ ϕᴾ_a
       Vtmp = phi' * (v_rho_a .* w .* phi)

       # Add the temporary back to the larger array by indexing, ensure it is symmetric
       V[lpos, lpos] += 0.5(Vtmp + Vtmp')
   end
   xc_e, V
end


println("XC Energy: ")
display(xc_e)
println("V matrix:")
display(V)

println("\nMatches Psi4 V: ", np.allclose(V, wfn.Va()))
# -

# Refs:
# - Johnson, B. G.; Fisch M. J.; *J. Chem. Phys.*, **1994**, *100*, 7429
