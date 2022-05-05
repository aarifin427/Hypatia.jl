"""
Data structure

eg1: x^2 + y^2 + z^2
eg2: x*y + y*z + x*z

Scope:
- polynomial
- homogenous
- hyperbolic -> how to check?
"""

# ===========================================================#
"""
This is the naive way or using a package (that does everything for you)

Using automatic diff package: "ForwardDiff"
ref: https://discourse.julialang.org/t/how-to-do-partial-derivatives/19869/3
"""
# ===========================================================#
using ForwardDiff

# eg1: x^2 + y^2 + z^2
f(x) = x[1]^2 + x[2]^2 + x[3]^2

g = x -> ForwardDiff.gradient(f, x);
g([1,2,3])

# ===========================================================#
"""
ref: http://www.cecm.sfu.ca/~mmonagan/talks/eccad2013.pdf
"""
# ===========================================================#

p = [     # data structure representing a polynomial
    1 1 1 # the coefficients of every term
    2 0 0 # power of x in every term
    0 2 0 # power of y in every term
    0 0 2 # power of z in every term
]

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

@assert g([1,2,3]) == gradient([1,2,3], p)