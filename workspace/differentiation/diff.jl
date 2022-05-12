"""
Data structure

eg1: x^2 + y^2 + z^2
eg2: x*y + y*z + x*z

Scope:
- polynomial
- homogenous
- hyperbolic -> how to check?

ref: https://discourse.julialang.org/t/how-to-do-partial-derivatives/19869/3
ref: http://www.cecm.sfu.ca/~mmonagan/talks/eccad2013.pdf
"""

# =====================================================#
# Function representation to value
# =====================================================#
function func_val(p::Matrix{T}, x::Vector{T}) where {T <: Real}
    res = 0 # result
    (row, col) = size(p)
    for j = 1:col
        addend = p[1,j]
        for i = 2:row
            addend *= x[i-1]^p[i,j]
        end
        res += addend
    end
    return res
end

# =====================================================#
# Gradient function
# =====================================================#
# TODO: conventional function documentation in Julia
function gradient(point, f)
    """
    Function: outputs the gradient value of polynomial f.
    Currently only works with monomials(?) (only works with x^a + y^b, not x^a*y^b + x^c*y^d)
        f       : polynomial function represented as a matrix
        point   : in this example, it's 3D, representing (x,y,z)
    """
    res = []
    _, y = size(f)
    for i in 1:y
        append!(res, f[1,i]*f[i+1,i]*point[i]^(f[i+1,i] - 1))
    end;
    return res
end;