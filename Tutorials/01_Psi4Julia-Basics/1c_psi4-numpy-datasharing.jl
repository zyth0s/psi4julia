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

# # Psi4 $\leftrightarrow$ Julia Data Sharing
#
# The heart of the Psi4Julia project its the ability to easily share and manipulate quantities in Julia. While Psi4 offers the ability to manipulate most objects and perform tensor operations at the Python layer, it is often much easier to use Julia. Fortunately, Psi4 offers seemless integration with Julia. More details on the underlying functions can be found in the Psi4 [documentation](http://psicode.org/psi4manual/master/numpy.html).
#
# As before, let us start off with importing Psi4 and NumPy while also creating a random `5 x 5` Julia array:

# +
using PyCall: pyimport
psi4 = pyimport("psi4")
np   = pyimport("numpy")

# Random number array
array = rand(5, 5)
# -

# Converting this to a Psi4 Matrix, which is an instance of the [`psi4.core.Matrix`](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.Matrix 
# "Go to API") class, and back again is as simple as:

# +
psi4_matrix = psi4.core.Matrix.from_array(array)
new_array = np.array(psi4_matrix)

println("Allclose new_array, array: ", np.allclose(new_array, array))
# -

# ## Views
# Because both of these objects have the same in-memory data layout, the conversion is accomplished using the NumPy 
# [array_interface](https://docs.scipy.org/doc/numpy/reference/arrays.interface.html). This also opens the opportunity 
# to manipulate the Psi4 Matrix and Vector classes directly in memory.  To do this, we wrap the PyObject with a Julia Matrix object:

# +
matrix = psi4.core.Matrix(3, 3)
print("Zero Psi4 Matrix:")
display(np.array(matrix))

#matrix.np[:] .= 1 # does not work as in Python, it is read-only
#matrix.set(1)     # works but it uses psi.core, not julia
function psi4view(psi4matrix)
   # Limitations: you have to check the type manually, it may not be a matrix
   # Use numpy __array_interface__
   # ["data"][1] is the pointer to the first element of the array
   # ["data"][2] is true when it is read-only
   # ["strides"] is nothing if has C ordering (default)
   # ["shape"]   has the shape in (C ordering by default) [needs reversing]
   # ["typestr"] has the element type
   array_interface = psi4matrix.__array_interface__
   array_interface["data"][2] == false   || @warn "Not writable"
   array_interface["strides"] == nothing || @warn "Different ordering than C"
   ptr   = array_interface["data"][1]
   shape = reverse(array_interface["shape"])
   unsafe_wrap(Matrix{Float64}, Ptr{Float64}(ptr), shape)
end
jlmatrix = psi4view(matrix) # This is a Julia matrix using psi4 memory
jlmatrix .= 1
print("Matrix updated to ones:")
display(np.asarray(matrix))
# -

# Theoretically, `PyArray(matrix)` (from PyCall) should return a Julia array. But here it does not work.
# The `psi4view` function created here effectively returns a Julia matrix that uses the memory of the Psi4 object. This view can then be manipulated as a conventional NumPy array and the underlying Psi4 Matrix data will be modified.
#
# <font color='red'>**Warning!** The following operation operation is incorrect and can potenitally lead to confusion:</font>

display(psi4.core.Matrix(3, 3).np)

# While the above operation works about ~90% of the time, occasionally you will notice extremely large and small values. This is due to the fact that when you create the Psi4 Matrix and grab its view, the Psi4 Matrix is no longer bound to anything, and Python will attempt to "garbage collect" or remove the object. This sometimes happens *before* Python prints out the object so the NumPy view is pointing to a random piece of data in memory. However, this has not been observed in Julia. A safe way to do this would be:

# +
mat = psi4.core.Matrix(3, 3)
display(mat.np)

# or
display(np.asarray(psi4.core.Matrix(3, 3)))
# -

# Similar to the `.np` attribute, one can use `np.asarray` to create a Julia copy of a Psi4 object. `np.array` which will copy the data too.

# +
mat = psi4.core.Matrix(3, 3)
mat_view = np.asarray(mat)

mat_view .= rand(mat.shape...)
display(mat.np)

# +
mat = psi4.core.Matrix(3, 3)
mat_view = psi4view(mat)

mat_view .= rand(mat.shape...)
display(mat.np)
# -

# Keep in mind that you must *update* this view using the `.=` syntax and not replace it (`=`). The following example should demonstrate the difference:

# +
mat_view = zeros(3, 3)

# Mat is not updated as we replaced the mat_view with a new Julia matrix.
display(mat.np)
# -

# ## Vector class
# Like the Psi4 Matrix class, the [`psi4.core.Vector`](http://psicode.org/psi4manual/master/psi4api.html#psi4.core.Vector "Go to API")
# class has similar accessors:

arr = rand(5)
vec = psi4.core.Vector.from_array(arr)
display(vec.np)
