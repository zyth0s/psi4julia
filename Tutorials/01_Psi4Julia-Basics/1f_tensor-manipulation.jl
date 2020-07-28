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

# ## Tensor Operations
#
# To perform most operations we turn to tensor packages (here we use [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl)). Those allow Einstein convention as an input. In addition to being much easier to read, manipulate, and change, it has (usually) optimal performance.
# First let us import our normal suite of modules:

using PyCall: pyimport
psi4 = pyimport("psi4")
np   = pyimport("numpy")
using TensorOperations: @tensor

# We can then use conventional Julia loops or `@tensor` to perform the same task. 

using BenchmarkTools: @btime, @belapsed

# With `@btime`/`@belapsed` we average time over several executions to have more reliable timings than `@time`/`@elapsed` (single execution).

# <font color="red">**WARNING: We are using Julia's global variables, and those are known to be less efficient than local variables. It is better to wrap code inside function.**</font>

# To begin let us consider the construction of the following tensor (which you may recognize):
# $$G_{pq} = 2.0 * I_{pqrs} D_{rs} - 1.0 * I_{prqs} D_{rs}$$ 
#
# Keep size relatively small as these 4-index tensors grow very quickly in size.

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
println("Time for @tensor G build:")
G = @btime @tensor G[p,q] := 2I[p,q,r,s] * D[r,s] - I[p,r,q,s] * D[r,s]

# Make sure the correct answer is obtained
println("Loop and einsum builds of the Fock matrix match?    ", np.allclose(G, Gloop))
println()
# Print out relative times for explicit loop vs einsum Fock builds
#println("G builds with einsum are $(g_loop_time/einsum_time) times faster than Julia loops!")
# -

# As you can see, the `@tensor` macro can be considerably faster than plain Julia loops.

# ## Matrix multiplication chain/train
#
# Now let us turn our attention to a more canonical matrix multiplication example such as:
# $$D_{il} = A_{ij} B_{jk} C_{kl}$$
#
# Matrix multiplication is an extremely common operation in all branches of linear algebra. Thus, these functions have been optimized to be extremely efficient. `@tensor` uses it. The matrix product will explicitly compute the following operation:
# $$C_{ij} = A_{ij} * B_{ij}$$
#
#
# This is Julia's matrix multiplication method `*` for matrices.

# +
dims = 200
A = rand(dims, dims)
B = rand(dims, dims)
C = rand(dims, dims)

# First compute the pair product
tmp_dot = A * B
@tensor tmp_tensor[i,k] := A[i,j] * B[j,k]
println("Pair product allclose? ", np.allclose(tmp_dot, tmp_tensor))
# -

# Now that we have proved exactly what `*` product does, let us consider the full chain and do a timing comparison:

D_dot = A * B * C
@tensor D_tensor[i,l] := A[i,j] * B[j,k] * C[k,l]
println("Chain multiplication allclose? ", np.allclose(D_dot, D_tensor))

# +
println()
println("* time:")
@btime A * B * C

println()
println("@tensor time:")
@btime @tensor D_tensor[i,l] := A[i,j] * B[j,k] * C[k,l];
# -

# Both have similar timings, and both call [Basic Linear Algebra Subprograms (BLAS)](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms). The BLAS routines are highly optimized and threaded versions of the code.
#  - The `@tensor` code will factorize the operation by default; Thus, the overall cost is not ${\cal O}(N^4)$ (as there are four indices) rather it is the factored $(\bf{A B}) \bf{C}$ which runs ${\cal O}(N^3)$.
#  
# Therefore you do not need to factorize the expression yourself (sometimes you might need):

println("@tensor factorized time:")
@btime @tensor begin
   tmp[i,k]  := A[i,j] * B[j,k]
   tmp2[i,l] := tmp[i,k] * C[k,l]
end
nothing

# On most machines the three have similar timings. The BLAS usage is usually recommended. Thankfully, in Julia its syntax is very clear. The Psi4Julia project tends to lean toward usage of tensor packages but if Julia's built-in matrix multiplication is significantly cleaner/faster we would use it. The real value of tensor packages will become tangible for more complicated expressions.

# ## Complicated tensor manipulations
# Let us consider a popular index transformation example:
# $$M_{pqrs} = C_{pi} C_{qj} I_{ijkl} C_{rk} C_{sl}$$
#
# Here, a naive loop implementation would scale like $\mathcal{O}(N^8)$ which translates to an extremely costly computation for all but the smallest $N$. A smarter implementation (factorizing the whole expression) would scale
# like $\mathcal{O}(N^5)$.

# <font color="red">**WARNING: First execution is slow because of compilation time. Successive are more honest to the running time.**</font>

# +
# Grab orbitals
dims = 15
@assert dims <= 15 || "Size must be smaller than 15."
    
C = rand(dims, dims)
I = rand(dims, dims, dims, dims)

# @tensor full transformation.
print("\nStarting @tensor full transformation...")
n8_time = @elapsed @tensor MO_n8[I,J,K,L] := C[p,I] * C[q,J] * I[p,q,r,s] * C[r,K] * C[s,L]
print("complete in $n8_time s\n")

# @tensor factorized N^5 transformation.
print("\nStarting @tensor factorized N^5 transformation with einsum ... ")
n5_time = @elapsed @tensor begin
   MO_n5[A,q,r,s] := C[p,A] * I[p,q,r,s]
   MO_n5[A,B,r,s] := C[q,B] * MO_n5[A,q,r,s]
   MO_n5[A,B,C,s] := C[r,C] * MO_n5[A,B,r,s]
   MO_n5[A,B,C,D] := C[s,D] * MO_n5[A,B,C,s]
end
print("complete in $n5_time s \n")
println("  @tensor factorized is $(n8_time/n5_time) faster than full @tensor algorithm!")
println("  Allclose? ", np.allclose(MO_n8, MO_n5))

# Julia's GEMM N^5 transformation.
# Try to figure this one out!
print("\nStarting Julia's factorized transformation with * ... ")
dgemm_time = @elapsed begin
   MO = C' * reshape(I, dims, :)
   MO = reshape(MO, :, dims) * C
   MO = permutedims(reshape(MO, dims, dims, dims, dims), (2, 1, 4, 3))

   MO = C' * reshape(MO, dims, :)
   MO = reshape(MO, :, dims) * C
   MO = permutedims(reshape(MO, dims, dims, dims, dims),(2, 1, 4, 3))
end
print("complete in $dgemm_time s \n")
println("  * factorized is $(n8_time/dgemm_time) faster than full @tensor algorithm!")
println("  Allclose? ", np.allclose(MO_n8, MO))

# There are still several possibilities to explore:
# @inbounds, @simd, LinearAlgebra.LAPACK calls, Einsum.jl, Tullio.jl, ...
# -
# None of the above algorithms is $\mathcal{O}(N^8)$. `@tensor` factorizes the expression to achieve better performance. There is a small edge in doing the factorization manually. Factorized algorithms have similar timings, although it is clear that with `@tensor` is easier than with Julia's built-in `*`. To use the usual matrix multiplication with tensors we have to reshape and permute their dimensions, subtracting appeal to the simple `*` syntax.

