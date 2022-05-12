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