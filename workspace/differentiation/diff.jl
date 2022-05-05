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

using ForwardDiff
using Random

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

# =====================================================#
# TEST
# =====================================================#
# TODO: Conventional unittest in Julia
pwc = 50 # power cap
n = 100  # number of tests
passes = 0
for k = 1:n
    local a, b, c = rand(1:pwc), rand(1:pwc), rand(1:pwc)
    local p = [
        rand(1:pwc) rand(1:pwc) rand(1:pwc)
        rand(1:pwc) 0           0
        0           rand(1:pwc) 0
        0           0           rand(1:pwc)
    ]
    local f(x) = p[1,1]*x[1]^p[2,1] + p[1,2]*x[2]^p[3,2] + p[1,3]*x[3]^p[4,3]
    local g = x -> ForwardDiff.gradient(f, x);
    local point = [rand(1:10), rand(1:10), rand(1:10)]
    
    global passes += g(point) == gradient(point, p)
end;
print(passes/n*100, "%")