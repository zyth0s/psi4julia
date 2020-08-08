#! A simple Psi4 input script to compute a SCF reference using Psi4's libJK

psi4 = pyimport("psi4")
np   = pyimport("numpy") # used only to cast to Psi4 arrays

build_superfunctional = nothing
if VersionNumber(psi4.__version__) >= v"1.3a1"
    build_superfunctional = psi4.driver.dft.build_superfunctional
else
    build_superfunctional = psi4.driver.dft_funcs.build_superfunctional
end

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
   permutedims(unsafe_wrap(Array{Float64,ndims}, Ptr{Float64}(ptr), shape), reverse(1:ndims))
end

# Diagonalize routine
function build_orbitals(diag, A, ndocc)
    Fp = psi4.core.triplet(A, diag, A, true, false, true)

    #A_view = psi4view(A)
    #nbf = size(A_view,1)
    nbf = A.shape[1]
    Cp = psi4.core.Matrix(nbf, nbf)
    eigvecs = psi4.core.Vector(nbf)
    Fp.diagonalize(Cp, eigvecs, psi4.core.DiagonalizeOrder.Ascending)

    C = psi4.core.doublet(A, Cp, false, false)
    C_jl = np.asarray(C)

    #Cocc = psi4.core.Matrix(nbf, ndocc)
    #Cocc_view = psi4view(Cocc) # returns C ordering
    #Cocc_view .= C_jl[:,1:ndocc] # or reverse because C ordering
    #Cocc.np[:] = C.np[:, 1:ndocc] # It is not going to work
    Cocc_jl  = zeros(nbf,ndocc)
    Cocc_jl .= C_jl[:,1:ndocc] # or reverse because C ordering
    Cocc = psi4.core.Matrix.from_array(Cocc_jl)

    D = psi4.core.doublet(Cocc, Cocc, false, true)

    C, Cocc, D, eigvecs
end

function ks_solver(alias, mol, options, V_builder::Function,
                   jk_type="DF", output="output.dat", restricted=true)

    # Build our molecule
    mol = mol.clone()
    mol.reset_point_group("c1")
    mol.fix_orientation(true)
    mol.fix_com(true)
    mol.update_geometry()

    # Set options
    psi4.set_output_file(output)

    psi4.core.prepare_options_for_module("SCF")
    psi4.set_options(options)
    psi4.core.set_global_option("SCF_TYPE", jk_type)

    maxiter = 20
    E_conv = psi4.core.get_option("SCF", "E_CONVERGENCE") 
    D_conv = psi4.core.get_option("SCF", "D_CONVERGENCE")
    
    # Integral generation from Psi4's MintsHelper
    wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option("BASIS"))
    mints = psi4.core.MintsHelper(wfn.basisset())
    S = mints.ao_overlap()

    # Build the V Potential
    sup = build_superfunctional(alias, restricted)[1]
    sup.set_deriv(2)
    sup.allocate()
    
    vname = "RV"
    if !restricted
        vname = "UV"
    end
    Vpot = psi4.core.VBase.build(wfn.basisset(), sup, vname)
    Vpot.initialize()
    
    # Get nbf and ndocc for closed shell molecules
    nbf = wfn.nso()
    ndocc = wfn.nalpha()
    if wfn.nalpha() != wfn.nbeta()
        error("Only valid for RHF wavefunctions!")
    end
    
    println("\nNumber of occupied orbitals: ", ndocc)
    println("Number of basis functions:   ", nbf)
    
    # Build H_core
    V = mints.ao_potential()
    T = mints.ao_kinetic()
    H = T.clone()
    H.add(V)
    
    # Orthogonalizer A = S^(-1/2)
    A = mints.ao_overlap()
    A.power(-0.5, 1.e-14)
    
    # Build core orbitals
    C, Cocc, D, eigs = build_orbitals(H, A, ndocc)
    
    # Setup data for DIIS
    #t = @elapsed begin
       E = 0.0
       Enuc = mol.nuclear_repulsion_energy()
       Eold = 0.0
       
       # Initialize the JK object
       jk = psi4.core.JK.build(wfn.basisset())
       jk.set_memory(Int(1.25e8))  # 1GB
       jk.initialize()
       jk.print_header()
       
       diis_obj = psi4.p4util.solvers.DIIS(max_vec=3, removal_policy="largest")
    #end
    
    #println("\nTotal time taken for setup: $t seconds")
    
    println("\nStarting SCF iterations:")
   
    println("\n    Iter            Energy             XC E         Delta E        D RMS\n")
    SCF_E = 0.0
    #t = @elapsed begin

       for SCF_ITER in 1:maxiter
       
           # Compute JK
           jk.C_left_add(Cocc)
           jk.compute()
           jk.C_clear()
       
           # Build Fock matrix
           F = H.clone()
           F.axpy(2.0, jk.J()[1])
           F.axpy(-Vpot.functional().x_alpha(), jk.K()[1])

           # Build V
           ks_e = 0.0

           Vpot.set_D([D])
           Vpot.properties()[1].set_pointers(D)
           V = V_builder(D, Vpot)
           if isnothing(V)
               ks_e = 0.0
           else
               ks_e, V = V
               V = psi4.core.Matrix.from_array(V)
           end
       
           F.axpy(1.0, V)

           # DIIS error build and update
           diis_e = psi4.core.triplet(F, D, S, false, false, false)
           diis_e.subtract(psi4.core.triplet(S, D, F, false, false, false))
           diis_e = psi4.core.triplet(A, diis_e, A, false, false, false)
       
           diis_obj.add(F, diis_e)
       
           dRMS = diis_e.rms()

           # SCF energy and update
           SCF_E  = 2H.vector_dot(D)
           SCF_E += 2jk.J()[1].vector_dot(D)
           SCF_E -= Vpot.functional().x_alpha() * jk.K()[1].vector_dot(D)
           SCF_E += ks_e
           SCF_E += Enuc
       
           printfmt("SCF Iter{1:3d}: {2:18.14f}   {3:11.7f}   {4:1.5E}   {5:1.5E}\n",
                          SCF_ITER, SCF_E, ks_e, (SCF_E - Eold), dRMS)
           if abs(SCF_E - Eold) < E_conv && dRMS < D_conv
               break
           end
       
           Eold = SCF_E
       
           # DIIS extrapolate
           F = diis_obj.extrapolate()
       
           # Diagonalize Fock matrix
           C, Cocc, D, eigs = build_orbitals(F, A, ndocc)
       
           if SCF_ITER == maxiter
               error("Maximum number of SCF cycles exceeded.")
           end
       end
    #end
    
    #println("\nTotal time for SCF iterations: $t seconds.")
    
    printfmt("\nFinal SCF energy: {:.8f} hartree \n", SCF_E)

    data = Dict()
    data["Da"] = D
    data["Ca"] = C
    data["eigenvalues"] = eigs
    SCF_E, data
end
