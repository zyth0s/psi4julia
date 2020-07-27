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

# # Tensor Manipulation: Psi4 and NumPy manipulation routines
# Contracting tensors together forms the core of the Psi4Julia project. First let us consider the popluar [Einstein Summation Notation](https://en.wikipedia.org/wiki/Einstein_notation) which allows for very succinct descriptions of a given tensor contraction.
#
# For example, let us consider a [inner (dot) product](https://en.wikipedia.org/wiki/Dot_product):
# $$c = \sum_{ij} A_{ij} * B_{ij}$$
#
# With the Einstein convention, all indices that are repeated are considered summed over, and the explicit summation symbol is dropped:
# $$c = A_{ij} * B_{ij}$$
#
# This can be extended to [matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication):
# \begin{align}
# \rm{Conventional}\;\;\;  C_{ik} &= \sum_{j} A_{ij} * B_{jk} \\
# \rm{Einstein}\;\;\;  C &= A_{ij} * B_{jk} \\
# \end{align}
#
# Where the $C$ matrix has *implied* indices of $C_{ik}$ as the only repeated index is $j$.
#
# However, there are many cases where this notation fails. Thus we often use the generalized Einstein convention. To demonstrate let us examine a [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)):
# $$C_{ij} = \sum_{ij} A_{ij} * B_{ij}$$
#
#
# This operation is nearly identical to the dot product above, and is not able to be written in pure Einstein convention. The generalized convention allows for the use of indices on the left hand side of the equation:
# $$C_{ij} = A_{ij} * B_{ij}$$
#
# Usually it should be apparent within the context the exact meaning of a given expression.
#
# Finally we also make use of Matrix notation:
# \begin{align}
# {\rm Matrix}\;\;\;  \bf{D} &= \bf{A B C} \\
# {\rm Einstein}\;\;\;  D_{il} &= A_{ij} B_{jk} C_{kl}
# \end{align}
#
# Note that this notation is signified by the use of bold characters to denote matrices and consecutive matrices next to each other imply a chain of matrix multiplications! 

# ## Einsum
#
# To perform most operations we turn to tensor packages (here we use [NumPy's einsum function](https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html) but there are similar Julia packages, Einsum.jl, Tullio.jl, ..., that might be explored in future revisions). Those allow Einsten convention as an input. In addition to being much easier to read, manipulate, and change, it is usually more efficient than our loop implementation.
#
# To begin let us consider the construction of the following tensor (which you may recognize):
# $$G_{pq} = 2.0 * I_{pqrs} D_{rs} - 1.0 * I_{prqs} D_{rs}$$ 
#
# First let us import our normal suite of modules:

using PyCall: pyimport
psi4 = pyimport("psi4")
np   = pyimport("numpy")

# We can then use conventional Julia loops and einsum to perform the same task. Keep size relatively small as these 4-index tensors grow very quickly in size.

using BenchmarkTools: @btime

# With `@btime` we measure execution time several times to have more reliable timings than with `@time` (single execution).

# <font color="red">**WARNING: We are using Julia's global variables, and those are known to be less efficient than local variables. It is better to wrap code inside function. For large computations we do a single execution so the timings will also include compilation time. Succesive runs are faster.**</font>

# +
dims = 20

@assert dims <= 30 "Size must be smaller than 30."
D = rand(dims, dims)
I = rand(dims, dims, dims, dims)

# Build the Fock matrix using loops, while keeping track of time
println("Time for loop G build:")
Gloop = @btime begin
   Gloop = np.zeros((dims, dims))
   @inbounds for ind in CartesianIndices(I)
       p, q, r, s = Tuple(ind)
       Gloop[p, q] += 2I[p, q, r, s] * D[r, s]
       Gloop[p, q] -=  I[p, r, q, s] * D[r, s]
   end
   Gloop
end

# Build the Fock matrix using einsum, while keeping track of time
println("Time for einsum G build:")
G = @btime begin
   J = np.einsum("pqrs,rs", I, D, optimize=true)
   K = np.einsum("prqs,rs", I, D, optimize=true)
   G = 2J - K
end

# Make sure the correct answer is obtained
println("Loop and einsum builds of the Fock matrix match?    ", np.allclose(G, Gloop))
println()
# Print out relative times for explicit loop vs einsum Fock builds
#println("G builds with einsum are $(g_loop_time/einsum_time) times faster than Julia loops!")
# -

# As you can see, the einsum function can be considerably faster than the plain Julia loops.

# ## Matrix multiplication chain/train
#
# Now let us turn our attention to a more canonical matrix multiplication example such as:
# $$D_{il} = A_{ij} B_{jk} C_{kl}$$
#
# We could perform this operation using einsum; however, matrix multiplication is an extremely common operation in all branches of linear algebra. Thus, these functions have been optimized to be more efficient than the `einsum` function. The matrix product will explicitly compute the following operation:
# $$C_{ij} = A_{ij} * B_{ij}$$
#
# This is Julia's matrix multiplication method `*`.

# +
dims = 200
A = rand(dims, dims)
B = rand(dims, dims)
C = rand(dims, dims)

# First compute the pair product
tmp_dot = A * B
tmp_einsum = np.einsum("ij,jk->ik", A, B, optimize=true)
println("Pair product allclose? ", np.allclose(tmp_dot, tmp_einsum))
# -

# Now that we have proved exactly what the dot product does, let us consider the full chain and do a timing comparison:

# +
D_dot = A * B * C
D_einsum = np.einsum("ij,jk,kl->il", A, B, C, optimize=true)
println("Chain multiplication allclose? ", np.allclose(D_dot, D_einsum))

println()
println("* time:")
@btime A * B * C

println("np.einsum time")
# no optimization here for illustrative purposes!
@btime np.einsum("ij,jk,kl->il", A, B, C);
# -

# On most machines the `*` times are roughly ~2,000 times faster. The reason is twofold:
#  - The `*` routines typically call [Basic Linear Algebra Subprograms (BLAS)](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms). The BLAS routines are highly optimized and threaded versions of the code.
#  - The `np.einsum` code will not factorize the operation by default; Thus, the overall cost is ${\cal O}(N^4)$ (as there are four indices) rather than the factored $(\bf{A B}) \bf{C}$ which runs ${\cal O}(N^3)$.
#  
# The first issue is difficult to overcome; however, the second issue can be resolved by the following:

println("np.einsum factorized time:")
# no optimization here for illustrative purposes!
@btime np.einsum("ik,kl->il", np.einsum("ij,jk->ik", A, B), C);

# On most machines the factorized `einsum` expression is only ~10 times slower than `*`. While a massive improvement, this is a clear demonstration the BLAS usage is usually recommended. Thankfully, in Julia its syntax is very clear. The Psi4Julia project tends to lean toward usage of tensor pacakges but if Julia's built-in matrix multiplication is faster we would use it.
#
# Starting in NumPy 1.12, the [einsum function](https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html) has a `optimize` flag which will automatically factorize the einsum code for you using a greedy algorithm, leading to considerable speedups at almost no cost:

println("np.einsum optimized time")
@btime np.einsum("ij,jk,kl->il", A, B, C, optimize=true);

# In this example, using `optimize=true` for automatic factorization is "only" 50% slower than `*`. Furthermore, it is ~8 times faster than factorizing the expression by hand, which represents a very good trade-off between speed and readability. When unsure, `optimize=true` is strongly recommended. The real value of tensor packages will become tangible for more complicated expressions.

# ## Complicated tensor manipulations
# Let us consider a popular index transformation example:
# $$M_{pqrs} = C_{pi} C_{qj} I_{ijkl} C_{rk} C_{sl}$$
#
# Here, a naive `einsum` call would scale like $\mathcal{O}(N^8)$ which translates to an extremely costly computation for all but the smallest $N$.

# +
# Grab orbitals
dims = 15
@assert dims <= 15 || "Size must be smaller than 15."
    
C = rand(dims, dims)
I = rand(dims, dims, dims, dims)

# Numpy's einsum N^8 transformation.
print("\nStarting np.einsum N^8 transformation...")
# no optimization here for illustrative purposes!
n8_time = @elapsed MO_n8 = np.einsum("pI,qJ,pqrs,rK,sL->IJKL", C, C, I, C, C)
print("complete in $n8_time s\n")

# Numpy's einsum N^5 transformation.
print("\nStarting np.einsum N^5 transformation with einsum ... ")
n5_time = @elapsed begin
   # no optimization here for illustrative purposes!
   MO_n5 = np.einsum("pA,pqrs->Aqrs", C, I)
   MO_n5 = np.einsum("qB,Aqrs->ABrs", C, MO_n5)
   MO_n5 = np.einsum("rC,ABrs->ABCs", C, MO_n5)
   MO_n5 = np.einsum("sD,ABCs->ABCD", C, MO_n5)
end
print("complete in $n5_time s \n")
println("N^5 is $(n8_time/n5_time) faster than N^8 algorithm!")
println("Allclose? ", np.allclose(MO_n8, MO_n5))

# Numpy's einsum optimized transformation.
print("\nNow np.einsum optimized transformation... ")
n8_time_opt = @elapsed MO_n8 = np.einsum("pI,qJ,pqrs,rK,sL->IJKL", C, C, I, C, C, optimize=true)
print("complete in $n8_time_opt s \n")

# Julia's GEMM N^5 transformation.
# Try to figure this one out!
print("\nStarting Julia's N^5 transformation with * ... ")
dgemm_time = @elapsed begin
   MO = C' * reshape(I, dims, :)
   MO = reshape(MO, :, dims) * C
   MO = permutedims(reshape(MO, dims, dims, dims, dims), (2, 1, 4, 3))

   MO = C' * reshape(MO, dims, :)
   MO = reshape(MO, :, dims) * C
   MO = permutedims(reshape(MO, dims, dims, dims, dims),(2, 1, 4, 3))
end
print("complete in $dgemm_time s \n")
println("Allclose? ", np.allclose(MO_n8, MO))
println("N^5 is $(n8_time/dgemm_time) faster than N^8 algorithm!")

# There are still several possibilities to explore:
# @inbounds, @simd, LinearAlgebra.LAPACK calls, Einsum.jl, Tullio.jl, ...
# -


