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

# ## DFT: GGA & Meta GGA Kernels

# ### Theory
#
#
# In density functional theory, we are interested in the energy expression:
#
# $$
# E_{xc} = D_{\mu \nu}^{T}(T_{\mu \nu}  + V_{\mu \nu}) + \frac{1}{2} D_{\mu \nu }^{T} D_{\lambda \sigma}^T (\mu \nu|\lambda \sigma) + E_{xc}[\rho_{\alpha}({\vec{r})}, \rho_{\beta}({\vec{r})}]
# $$
#
#
#
#
#
# Although the exchange correlation energy $E_{xc}$ is a functional of the density alone, the dependence on  $\rho_{\sigma}(\vec{r})$ is highly non-local. Because of this, small variations of the densities may cause large variations of the exchange correlation potential $v_{xc}$. Additionally, $v_{xc}$ at a given point $\vec{r}_i$ may be sensitive to changes at very distant points of $\vec{r}_j$. 
#
#
#
# In order to overcome this, both semilocal and nonlocal ingredients must be added to the energy density. As expressed on the grid, these include:
#
# The gradient of the density:
# $$
# g = |\nabla \rho(\vec(r)) |
# $$
# The laplacian of the density:
# $$
# l = \nabla^2 \rho (\vec{r})
# $$
# And the non-interacting kinetic energy density:
# $$
# \tau = \frac{1}{2} \sum_k^{occ.} | \nabla \psi_k (\vec{r}) |^2
# $$
#
# Different functional approximations are defined by what ingredients are required to be created. Here, we concentrate on GGAs and meta-GGAs:
#
#
# $$
# E_{xc}^{GGA}[\rho] = \int f(\rho, g) \cdot d\vec{r}
# $$
# $$
# E_{xc}^{MGGA}[\rho] = \int f(\rho, g, l, \tau) \cdot d\vec{r}
# $$
#
# Where the integrands are known as the kernel, or the exchange-correlation energy density. It is clear here that, the more sophisticated density functional approximations, the more components are added to the exchange-correlation energy and potential. 
#
#
# In practice we need to build a Kohn-Sham matrix:
#
# $$
# F_{\mu \nu}^{\alpha} = H_{\mu \nu} + J_{\mu \nu} + V_{\mu \nu}^{xc},
# $$
#
# where the last therm is the exchange-correlation contribution $V_{\mu \nu}^{xc}$ that is defined as the functional derivative of the energy with respect to the density. 
#
#
# $$
# V^{xc} = \frac{\partial E_{xc}}{\partial D_{ab}} 
# $$
#
# Once we have the $V^{xc}$ we can build the Kohn-Sham matrix and solve self consistently. 
#

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

gga_energy, gga_wfn = psi4.energy("PBE", return_wfn=true)
meta_energy, meta_wfn = psi4.energy("TPSS", return_wfn=true)

Vpot = gga_wfn.V_potential()
Vpot_meta = meta_wfn.V_potential()
# -

# ## Building the GGA kernel
#
#
# From the LDA tutorial, we have seen how to obtain the density from the basis functions:
#
# $$
# \rho_{\sigma}(\vec{r}) = D_{\mu \nu}^{\sigma} \phi_{\mu}(\vec{r}) \phi_{\nu}(\vec{r}) 
# $$
#
#
#
#
# The GGA depends on the gradient of the density. Using a basis set it is calculated as:
#
#
# $$
# \nabla_{\sigma} \rho{} (\vec{r}) = 2  D_{\mu \nu}^{\sigma}  \phi_{\mu}(\vec{r}) \nabla \phi_{\nu}(\vec{r})
# $$ 
#
# So that we can produce $\gamma$:
#
#
# $$
# \gamma_{\alpha \alpha}(\vec{r}) = \nabla \rho_{\alpha}(\vec{r}) \cdot \nabla \rho_{\alpha}(\vec{r})
# $$
#
#
# We then need to get the energy from the kernel by doing a numerical integration:
# $$e_{\rm{xc}} = w^a f^a_{\rm{xc}}$$
#
# Where the $w_{\alpha}$ correspond to the combined Truetler and Lebedev weights at each point needed for the numerical quadrature.
#
# Finally, the potential on the grid will be given by the derivative of the kernel with respect to gamma:
#
# $$
# V^{\gamma} = 2 \frac{\partial f}{\partial \gamma_{\alpha\alpha}} \nabla \rho_{\alpha} + \frac{\partial f}{\partial \gamma_{\alpha\beta}} \nabla \rho_{\beta}
# $$
#
#
# And it can be added and then it needs to get reintegrated back to atomic orbital space:
#
# $$
# V_{ab}^{\gamma} = \int_{\mathbb{R}^3} V^{\gamma} \nabla (\phi_{\mu} \phi_{\nu}) d\vec{r}
# $$
#
# The next calculation assumes that the density matrix $D$ is symmetric. This means that $ \nabla \phi(\vec{r}) D_{\mu \nu} \phi(\vec{r})= \phi(\vec{r}) D_{\mu \nu} \nabla \phi(\vec{r})$.
# One then ought to be careful with systems where this condition is not met, for example CPHF. 
#
# Specifically,  here write this matrix contribution as:
#
# $$
# V_{ab}^{\gamma} = 4 \cdot \nabla \phi_{\mu}(\vec{r}) \cdot V_{\alpha \alpha}^{\gamma} \cdot \nabla n_{p}(\vec{r}) \cdot w_{\alpha} \cdot \phi_{\nu} (\vec{r})
# $$
#
#

# +
#GGA Kernel

D = np.array(gga_wfn.Da())

V = zero(D)
xc_e = 0.0

rho = []
points_func = Vpot.properties()[1]
superfunc = Vpot.functional()

xc_e, V = let xc_e = xc_e, rho=rho

   # Loop over the blocks
   for b in 1:Vpot.nblocks()
       
       # Obtain block information
       block = Vpot.get_block(b-1)
       points_func.compute_points(block)
       npoints = block.npoints()
       lpos = np.array(block.functions_local_to_global()) .+ 1
       
       # Obtain the grid weight
       w = np.array(block.w())

       # Compute phi!
       phi = np.array(points_func.basis_values()["PHI"])[1:npoints, 1:size(lpos,1)]
       
       phi_x = np.array(points_func.basis_values()["PHI_X"])[1:npoints, 1:size(lpos,1)]
       phi_y = np.array(points_func.basis_values()["PHI_Y"])[1:npoints, 1:size(lpos,1)]
       phi_z = np.array(points_func.basis_values()["PHI_Z"])[1:npoints, 1:size(lpos,1)]
       
       # Build a local slice of D
       lD = D[lpos, lpos]
       
       # Compute rho
       # ρ[p] = ϕ[pμ] lD[μν] ϕ[pν]
       rho = 2vec(sum( (phi * lD) .* phi, dims=2))
       
       # 2.0 for Px D P + P D Px, 2.0 for non-spin Density
       # ∇ϕₓ[p] = 4 ϕ[pμ] lD[μν] ∇ₓϕ[pν] ∀ x,y,z
       rho_x = 4vec(sum( (phi * lD) .* phi_x, dims=2))
       rho_y = 4vec(sum( (phi * lD) .* phi_y, dims=2))
       rho_z = 4vec(sum( (phi * lD) .* phi_z, dims=2))
       gamma = @. rho_x^2 + rho_y^2 + rho_z^2
       
       inp = Dict()
       inp["RHO_A"] = psi4.core.Vector.from_array(rho)
       inp["GAMMA_AA"] = psi4.core.Vector.from_array(gamma)
       
       # Compute the kernel
       ret = superfunc.compute_functional(inp, -1)
       
       # Compute the XC energy
       vk = np.array(ret["V"])[1:npoints]
       xc_e += sum(w .* vk)
           
       # Compute the XC derivative.
       v_rho_a = np.array(ret["V_RHO_A"])[1:npoints]
       # Vtmp[ab] = 0.5 ϕ[pb] Vρ_a[p] w[p] ϕ[pa]
       Vtmp = 0.5phi' * (v_rho_a .* w .* phi)

       #Comute gamma and its associated potential
       v_gamma_aa = np.array(ret["V_GAMMA_AA"])[1:npoints]
       # Vtmp[ab] += 2 ∇ₓϕ[pb] Vγ_aa[p] ∇ₓρ[p] w[p] ϕ[pa] ∀ x,y,z
       Vtmp += 2phi_x' * (v_gamma_aa .* rho_x .* w .* phi)
       Vtmp += 2phi_y' * (v_gamma_aa .* rho_y .* w .* phi)
       Vtmp += 2phi_z' * (v_gamma_aa .* rho_z .* w .* phi)


       # Add the temporary back to the larger array by indexing, ensure it is symmetric
       V[lpos, lpos] += (Vtmp + Vtmp')
   end
   xc_e, V
end


println("XC Energy ", xc_e)
println("V matrix:")
display(V)

println("\nMatches Psi4 V: ", np.allclose(V, gga_wfn.Va()))

# -

# ## Building the meta-GGA kernel
#
#
# Just like we did with GGA, meta-GGA requires an extra component to be added. In this case is the kinetic energy density. 
#
# $$
# \tau_{\sigma} (\vec{r}) = D_{\mu \nu}^{\sigma} \nabla \phi_{\mu}(\vec{r}) \nabla \phi_{\nu}(\vec{r})
# $$
#
# We calculate the $E_{xc}$ again with $ w^a f^a_{\rm{xc}}$.
#
# And finally, the $\tau$ potential contribution can be calculated as:
#
# $$
# V^{\tau} = \frac{\partial f}{\partial \tau}
# $$
#
# Which is expressed as a matrix in the basis sest as:
#
# $$
# V_{\mu \nu}^{\tau} = \int_{\mathbb{R}^3} V^{\tau} \nabla \phi_{\mu} \nabla \phi_{\nu} d\vec{r}
# $$
#
#
# In the code we calculate this contribution like so:
#
# $$
# V_{\mu \nu}^{\tau} = \frac{1}{2}  \cdot \nabla \phi_{\mu} \cdot V^{\tau}_{a} \cdot w_{a} \cdot \nabla \phi_{\nu} 
# $$
#
#
#

# +
#meta-GGA Kernel

D = np.array(meta_wfn.Da())

V = zero(D)
xc_e = 0.0


rho = []
points_func = Vpot_meta.properties()[1]
superfunc = Vpot_meta.functional()

xc_e, V = let xc_e = xc_e, rho=rho

   # Loop over the blocks
   for b in 1:Vpot.nblocks()
       
       # Obtain block information
       block = Vpot.get_block(b-1)
       points_func.compute_points(block)
       npoints = block.npoints()
       lpos = np.array(block.functions_local_to_global()) .+ 1
       
       tau = np.zeros(npoints)
       
       # Obtain the grid weight
       w = np.array(block.w())

       # Compute phi!
       phi = np.array(points_func.basis_values()["PHI"])[1:npoints, 1:size(lpos,1)]
       
       phi_x = np.array(points_func.basis_values()["PHI_X"])[1:npoints, 1:size(lpos,1)]
       phi_y = np.array(points_func.basis_values()["PHI_Y"])[1:npoints, 1:size(lpos,1)]
       phi_z = np.array(points_func.basis_values()["PHI_Z"])[1:npoints, 1:size(lpos,1)]
       
       # Build a local slice of D
       lD = D[lpos, lpos]
       
       # Compute rho
       # ρ[p] = 2 ϕ[pm] lD[mn] ϕ[pn]
       rho = 2vec(sum( (phi * lD) .* phi, dims=2))
       
       # 2.0 for Px D P + P D Px, 2.0 for non-spin Density
       # ∇ₓρ[p] = 4 ϕ[pm] lD[mn] ∇ₓϕ[pn]
       rho_x = 4vec(sum( (phi * lD) .* phi_x, dims=2))
       rho_y = 4vec(sum( (phi * lD) .* phi_y, dims=2))
       rho_z = 4vec(sum( (phi * lD) .* phi_z, dims=2))
       gamma = @. rho_x^2 + rho_y^2 + rho_z^2
       
       #Compute Tau
       # τ[p] = ∇ₓϕ[pm] lD[mn] ∇ₓϕ[pn]
       tau  = vec(sum( (phi_x * lD) .* phi_x, dims=2))
       tau += vec(sum( (phi_y * lD) .* phi_y, dims=2))
       tau += vec(sum( (phi_z * lD) .* phi_z, dims=2))
       
       inp = Dict()
       inp["RHO_A"] = psi4.core.Vector.from_array(rho)
       inp["GAMMA_AA"] = psi4.core.Vector.from_array(gamma)
       inp["TAU_A"]= psi4.core.Vector.from_array(tau)
       
       # Compute the kernel
       ret = superfunc.compute_functional(inp, -1)
       
       # Compute the XC energy
       vk = np.array(ret["V"])[1:npoints]
       xc_e += sum(w .* vk)
           
       # Compute the XC derivative.
       v_rho_a = np.array(ret["V_RHO_A"])[1:npoints]
       # Vtmp[ab] = 0.5 ϕ[pb] Vρ_a[p] w[p] ϕ[pa]
       Vtmp = 0.5phi' * (v_rho_a .* w .* phi)

       
       #Compute gamma and its potential matrix
       v_gamma_aa = np.array(ret["V_GAMMA_AA"])[1:npoints]
       # Vtmp[ab] = 2∇ₓϕ[pb] Vγ_aa[p] ∇ₓϕ[p] w[p] ϕ[pa] ∀ x,y,z
       Vtmp += 2phi_x' * (v_gamma_aa .* rho_x .* w .* phi)
       Vtmp += 2phi_y' * (v_gamma_aa .* rho_y .* w .* phi)
       Vtmp += 2phi_z' * (v_gamma_aa .* rho_z .* w .* phi)
       
       #Compute Vτ
       v_tau_a = np.array(ret["V_TAU_A"])[1:npoints]
       # Vtmp[ab] += 0.5 ∇ₓϕ[pb] Vτ_a[p] w[p] ∇ₓϕ[pa] ∀ x,y,z
       Vtmp += 0.5phi_x' * (v_tau_a .* w .* phi_x)
       Vtmp += 0.5phi_y' * (v_tau_a .* w .* phi_y)
       Vtmp += 0.5phi_z' * (v_tau_a .* w .* phi_z)

       # Add the temporary back to the larger array by indexing, ensure it is symmetric
       V[lpos, lpos] += (Vtmp + Vtmp')
   end
   xc_e, V
end


println("XC Energy ", xc_e)
println("V matrix:")
display(V)

println("\nMatches Psi4 V: ", np.allclose(V, meta_wfn.Va()))

# -

# #### To put all the approximations into perspective, let us look at every component of the meta-GGA exchange-correlation potential that we just created. 
#
# $$
# V_{\mu \nu}^{xc, \alpha} = \int_{\mathbb{R}^3} \bigg( \frac{\partial f}{\partial \rho_{\alpha}} \bigg)  \phi_{\mu} \phi_{\nu} d\vec{r}
# $$
#
#
#
#
#

# $$
# +\int_{\mathbb{R}^3} \bigg(2 \frac{\partial f}{\partial \gamma_{\alpha\alpha}} \nabla \rho_{\alpha} + \frac{\partial f}{\partial \gamma_{\alpha\beta}} \bigg)  \nabla \rho_{\beta} \nabla (\phi_{\mu} \phi_{\nu}) d\vec{r}
# $$
#

# $$
# \int_{\mathbb{R}^3} \bigg( \frac{\partial f}{\partial \tau} \bigg)  \nabla \phi_{\mu} \nabla \phi_{\nu} d\vec{r}
# $$
#     
# Here every line represent each of the rungs in the systematic methodology of density functional aproximations, the first line corresponds to LDA, addition of the second line corresponds to GGA and addition to the third line corresponds to meta-GGA.

# ## References
#
# 1. Original papers:
# 	> [[Hohenberg:1964:136](https://journals.aps.org/pr/abstract/10.1103/PhysRev.136.B864)] P. Hohenberg and W. Kohn, Phys. Rev. 136, B864-B871, **1964**.
#     
#     > [[Kohn:1965:A1133](https://journals.aps.org/pr/abstract/10.1103/PhysRev.140.A1133)] W. Kohn and L.J. Sham, Phys. Rev. 140, A1133-A1138, **1965**.
# 2. Analytic derivatives and algorithm:
#     > [[Johnson:1994:100](https://aip.scitation.org/doi/abs/10.1063/1.466887)] Johnson, B. G.; Fisch M. J.; *J. Chem. Phys.*, **1994**, *100*, 7429
# 4. Additional information:
# 	> [[Staroverov:2012](https://onlinelibrary.wiley.com/doi/abs/10.1002/9781118431740#page=156)] Staroverov, Viktor N. "Density-functional approximations for exchange and correlation." A Matter of Density, **2012**: 125-156.
#     
#     > [[Parr:1989](https://link.springer.com/chapter/10.1007/978-94-009-9027-2_2)] R.G. Parr and W. Yang, Density Functional Theory of Atoms and Molecules Oxford University Press, USA, 1989 ISBN:0195357736, 9780195357738
