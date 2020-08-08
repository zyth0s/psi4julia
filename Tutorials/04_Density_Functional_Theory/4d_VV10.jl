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

# # VV10 Non-local correlation kernel
# One of the largest deficiencies of semilocal functionals is the lack of long-range correlation effects. The most notable effect is the lack of dispersion in the interactions between molecules. VV10 was expressly created to bridge the gap between the expensive true non-local correlation and a computational tractable form. We will begin by writing the overall expression:
#
# $$E_c^{\rm{nl}} = \frac{1}{2}\int \int d{\bf r}d{\bf r'}\rho({\bf r})\Phi({\bf r},{\bf r'})\rho({\bf r'}),$$
#
# where the two densities are tied together through the $\Phi$ operator.
#
# For VV10 we have:
# $$
# \begin{align}
# \Phi &= -\frac{3}{2gg'(g + g')},\\
#  g &= \omega_0({\rm r}) R^2 + \kappa({\rm r)},\\
#  g' &= \omega_0({\rm r}) R^2 + \kappa({\rm r')},
# \end{align}
# $$
#
# where $\omega_{0}$:
#
# $$
# \begin{align}
# \omega_{0}(r) &= \sqrt{\omega_{g}^2(r) + \frac{\omega_p^2(r)}{3}} \\
# \omega_g^2(r) &= C \left | \frac{\nabla \rho({\bf r})}{\rho({\bf r})} \right |^4 \\
# \omega_p^2(r) &= 4 \pi \rho({\bf r}),
# \end{align}
# $$
#
# and finally:
#
# $$\kappa({\bf r}) = b * \frac{3 \pi}{2} \left [ \frac{\rho({\bf r})}{9\pi} \right ]^\frac{1}{6}.$$
#
# While there are several expressions, this is actually relatively easy to compute. First let us examine how the VV10 energy is reintegrated:
#
# $$E_c^{\rm{VV10}} = \int d{\bf r} \rho{\bf r} \left [ \beta + \frac{1}{2}\int d{\bf r'} \rho{\bf r'} \Phi({\bf r},{\bf r'}) \right].$$
#
#

# +
using PyCall: pyimport
psi4 = pyimport("psi4")
np   = pyimport("numpy") # used only to cast to Psi4 arrays
using LinearAlgebra: dot
import Formatting: printfmt
include("ks_helper.jl")

mol = psi4.geometry("""
He 0 0 -5
He 0 0  5
symmetry c1
""")
options = (Dict("BASIS" => "aug-cc-pvdz",
                "DFT_SPHERICAL_POINTS" => 110,
                "DFT_RADIAL_POINTS" => 20))
# -

# ## VV10 coefficients
# First let us build set and build a few static coefficients:

coef_C = 0.0093
coef_B = 5.9
β = 1.0 / 32.0 * (3.0 / (coef_B^2.0))^(3.0 / 4.0)


# ## VV10 kernel
# First let us construct a function that computes $\omega_0$, and $\kappa$ quantities. We simplify the following piece contained in $\omega_g$:
# $$\left |\frac{\nabla \rho({\bf r})}{\rho({\bf r})} \right|^4$$
#
# by recalling that 
#
# $$\gamma({\bf r}) = \nabla\rho({\bf r})\cdot\nabla\rho({\bf r}),$$
#
# therefore, we can simplify the above to:
#
# $$\left |\frac{\nabla \rho({\bf r})}{\rho({\bf r})} \right |^4 = \left | \frac{\gamma({\bf r})}{\rho({\bf r})\cdot \rho({\bf r})} \right | ^2 $$

function compute_vv10_kernel(ρ, γ)
    κ_pref = coef_B * (1.5 * π) / ((9.0 * π)^(1.0 / 6.0))
    
    # Compute R quantities
    Wp = (4.0 / 3.0) * π * ρ
    Wg = @. coef_C * ((γ / (ρ * ρ))^2.0)
    W0 = @. sqrt(Wg + Wp)
    
    κ = @. ρ^(1.0 / 6.0) * κ_pref

    W0, κ
end


# ## VV10 energy and gradient evaluation
#
# The next block of code computes the VV10 energy and its gradient. In the very end we plug this function in
# the Khon-Sham solver `ks_solver` that is in a separate file `ks_helper.jl`. We conveniently separated the solver to focus in the non-local kernel.

# +
function compute_vv10(D, Vpot)

    nbf = D.shape[1]
    Varr = zeros(nbf, nbf)
    
    total_e = 0.0
    tD = 2np.array(D)
    
    points_func = Vpot.properties()[1]
    superfunc = Vpot.functional()

    xc_e = 0.0
    vv10_e = 0.0
    
    # First loop over the outer set of blocks
    for l_block in 1:Vpot.nblocks()
        
        # Obtain general grid information
        l_grid = Vpot.get_block(l_block-1)
        l_w = np.array(l_grid.w())
        l_x = np.array(l_grid.x())
        l_y = np.array(l_grid.y())
        l_z = np.array(l_grid.z())
        l_npoints = size(l_w,1)

        points_func.compute_points(l_grid)

        # Compute the functional itself
        ret = superfunc.compute_functional(points_func.point_values(), -1)
        
        xc_e += dot(l_w, np.array(ret["V"])[1:l_npoints])
        v_ρ = np.array(ret["V_RHO_A"])[1:l_npoints]
        v_γ = np.array(ret["V_GAMMA_AA"])[1:l_npoints]
        
        # Begin VV10 information
        l_ρ = np.array(points_func.point_values()["RHO_A"])[1:l_npoints]
        l_γ = np.array(points_func.point_values()["GAMMA_AA"])[1:l_npoints]
        
        l_W0, l_κ = compute_vv10_kernel(l_ρ, l_γ)
        
        phi_kernel = zero(l_ρ)
        phi_U = zero(l_ρ)
        phi_W = zero(l_ρ)
        
        # Loop over the inner set of blocks
        for r_block in 1:Vpot.nblocks()
            
            # Repeat as for the left blocks
            r_grid = Vpot.get_block(r_block-1)
            r_w = np.array(r_grid.w())
            r_x = np.array(r_grid.x())
            r_y = np.array(r_grid.y())
            r_z = np.array(r_grid.z())
            r_npoints = size(r_w,1)

            points_func.compute_points(r_grid)

            r_ρ = np.array(points_func.point_values()["RHO_A"])[1:r_npoints]
            r_γ = np.array(points_func.point_values()["GAMMA_AA"])[1:r_npoints]
        
            r_W0, r_κ = compute_vv10_kernel(r_ρ, r_γ)
            
            newaxis = [CartesianIndex()]

            # Build the distance matrix
            R2  = (l_x[:, newaxis] .- r_x').^2
            R2 += (l_y[:, newaxis] .- r_y').^2
            R2 += (l_z[:, newaxis] .- r_z').^2
            
            # Build g
            g  = @. l_W0[:, newaxis] * R2 + l_κ[:, newaxis]
            gp = @. r_W0' * R2 + r_κ'
        
            F_kernel = @. -1.5(r_w * r_ρ)' / (g * gp * (g + gp))
            F_U = @. F_kernel * ((1.0 / g) + (1.0 / (g + gp)))
            F_W = F_U .* R2

            phi_kernel += sum(F_kernel, dims=2)
            phi_U += -sum(F_U, dims=2)
            phi_W += -sum(F_W, dims=2)
        end
            
        # Compute those derivatives
        κ_dn = l_κ ./ (6l_ρ)
        w0_dγ = @. coef_C * l_γ / (l_W0 * l_ρ^4.0)
        w0_dρ = @. 2.0 / l_W0 * (π/3.0 - coef_C * l_γ^2.0 / (l_ρ^5.0))

        # Sum up the energy
        vv10_e += sum(@. l_w * l_ρ * (β + 0.5phi_kernel))

        # Perturb the derivative quantities
        v_ρ += @. β + phi_kernel + l_ρ * (κ_dn * phi_U + w0_dρ * phi_W)
        v_ρ *= 0.5
        
        v_γ += l_ρ .* w0_dγ .* phi_W

        # Recompute to l_grid
        lpos = np.array(l_grid.functions_local_to_global()) .+ 1
        points_func.compute_points(l_grid)
        nfunctions = size(lpos,1)
        
        # Integrate the LDA and GGA quantities
        phi = np.array(points_func.basis_values()["PHI"])[1:l_npoints, 1:nfunctions]
        phi_x = np.array(points_func.basis_values()["PHI_X"])[1:l_npoints, 1:nfunctions]
        phi_y = np.array(points_func.basis_values()["PHI_Y"])[1:l_npoints, 1:nfunctions]
        phi_z = np.array(points_func.basis_values()["PHI_Z"])[1:l_npoints, 1:nfunctions]
        
        # LDA
        # Vtmp[ab] = ϕ[pb] Vρ[p] lw[p] ϕ[pa]
        Vtmp = phi' * (v_ρ .* l_w .* phi)

        # GGA
        l_ρ_x = np.array(points_func.point_values()["RHO_AX"])[1:l_npoints]
        l_ρ_y = np.array(points_func.point_values()["RHO_AY"])[1:l_npoints]
        l_ρ_z = np.array(points_func.point_values()["RHO_AZ"])[1:l_npoints]
        
        tmp_grid = 2l_w .* v_γ
        # Vtmp[ab] += ϕ[pb] tmp_grid[p] ∇ₓρ[p] ∇ₓϕ[pa] ∀ x,y,z
        Vtmp += phi' * (tmp_grid .* l_ρ_x .* phi_x)
        Vtmp += phi' * (tmp_grid .* l_ρ_y .* phi_y)
        Vtmp += phi' * (tmp_grid .* l_ρ_z .* phi_z)
        
        # Sum back to the correct place
        Varr[lpos, lpos] += Vtmp + Vtmp'
    end
        
    printfmt("   VV10 NL energy: {:16.8f}\n", vv10_e)
        
    xc_e += vv10_e
    return xc_e, Varr
end

ks_solver("VV10", mol, options, compute_vv10)       
# -

# Refs:
#  - Vydrov O. A.; Van Voorhis T., *J. Chem. Phys.*, **2010**, *133*, 244103
