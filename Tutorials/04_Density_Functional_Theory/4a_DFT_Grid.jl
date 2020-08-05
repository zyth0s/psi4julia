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

# # Density Functional Theory: Grid
# ## I. Theoretical Overview
# This tutorial will discuss the basics of DFT and discuss the grid used to evaluate DFT quantities.
# As with HF, DFT aims to solve the generalized eigenvalue problem:
#
# $$\sum_{\nu} F_{\mu\nu}C_{\nu i} = \epsilon_i\sum_{\nu}S_{\mu\nu}C_{\nu i}$$
# $${\bf FC} = {\bf SC\epsilon},$$
#
# While with HF the Fock matrix is constructed as:
#
# $$F^{HF}_{\mu\nu} = H_{\mu\nu} + 2J[D]_{\mu\nu} - K[D]_{\mu\nu}$$
#
# $$D_{\mu\nu} = C_{\mu i} C_{\nu i},$$
#
# with DFT we generalize this construction slightly to:
# $$F^{DFT}_{\mu\nu} = H_{\mu\nu} + 2J[D]_{\mu\nu} - \zeta K[D]_{\mu\nu} + V^{\rm{xc}}_{\mu\nu}.$$
#
# $\zeta$ is an adjustable parameter where we can vary the amount of exact (HF) exchange and $V$ is the DFT potenital which typically attempts to add dynamical correlation in the self-consistent field methodolgy.
#
#

# ## 2. Examining the Grid
# We will discuss the evaluation and manipulation of the grid.

using PyCall: pyimport
psi4 = pyimport("psi4")
np   = pyimport("numpy") # used only to cast to Psi4 arrays
import PyPlot: plt, matplotlib
matplotlib.use(backend="MacOSX")

build_superfunctional = nothing
if VersionNumber(psi4.__version__) >= v"1.3a1"
    build_superfunctional = psi4.driver.dft.build_superfunctional
else
    build_superfunctional = psi4.driver.dft_funcs.build_superfunctional
end

# Set computation options and molecule, any single atom will do.

mol = psi4.geometry("He")
psi4.set_options(Dict("BASIS" => "cc-pvdz",
                      "DFT_SPHERICAL_POINTS" => 50,
                      "DFT_RADIAL_POINTS" => 12))

basis = psi4.core.BasisSet.build(mol, "ORBITAL", "CC-PVDZ")
sup = build_superfunctional("PBE", true)[1]
Vpot = psi4.core.VBase.build(basis, sup, "RV")
Vpot.initialize()

x, y, z, w = Vpot.get_np_xyzw()
R = @. sqrt(x^2 + y^2 + z^2);

fig, ax = plt.subplots()
ax.scatter(x, y, c=w)
#ax.set_xscale("log")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
mask = R .> 8
p = ax.scatter(x[mask], y[mask], z[mask], c=w[mask], marker="o")
plt.colorbar(p)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# +
mol = psi4.geometry("""
 O
 H 1 1.1
 H 1 1.1 2 104
""")
mol.update_geometry()
psi4.set_options(Dict("BASIS" => "cc-pvdz",
                      "DFT_SPHERICAL_POINTS" => 26,
                      "DFT_RADIAL_POINTS" => 12))

basis = psi4.core.BasisSet.build(mol, "ORBITAL", "CC-PVDZ")
sup = build_superfunctional("PBE", true)[1]
Vpot = psi4.core.VBase.build(basis, sup, "RV")
Vpot.initialize()
x, y, z, w = Vpot.get_np_xyzw()
R = @. sqrt(x^2 + y^2 + z^2)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
mask = R .> 0
p = ax.scatter(x[mask], y[mask], z[mask], c=w[mask], marker="o")
plt.colorbar(p)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
# -

# ## Refs:
# - Koch, W. and Holthausen, M.C., **2001**, A Chemist’s Guide to Density Functional Theory, 2nd, Wiley-VCH, Weinheim.
# - Kohn, W. and Sham, L. *J, Phys. Rev.*, **1965**, *140*, A1133- A1138
# - Becke, A.D., *J. Chem. Phys.*, **1988**, *88*, 2547
# - Treutler, O. and Ahlrichs, R., *J. Chem. Phys.*, **1995**, *102*, 346
# - Gill, P.M.W., Johnson, B.G., and Pople, J.A., *Chem. Phys. Lett.*, **1993,209 (5), pp. 506, 16 July 1993.
