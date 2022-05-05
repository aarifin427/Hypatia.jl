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