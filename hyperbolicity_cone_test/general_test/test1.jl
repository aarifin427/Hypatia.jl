"""
Problem definition:
min     c'*x
s.t.    A*x - b = 0
        h - G*x ∈ int(K)
where K is the cone.

Simplified:
min     c'*x
s.t.    A*x - b = 0
        x ∈ int(K)
where h = 0, G = -identity
"""

using ForwardDiff
using Hypatia
using Hypatia.Cones
using Hypatia.Models
import Hypatia.Solvers
using LinearAlgebra

# Solution type
T = Float64;

# Dimension of solution
n = 3;

# 1 linear constraint for A*x - b = 0
A = 1.0*[0.1 1 1]
b = [0.5]

# Objective function coefficients
c = [
    0.6485574918878491;
    0.9051391876123972;
    0.5169347627896711;
]

G = Diagonal(-one(T) * I, n)
h = zeros(T, n)

# Hyperbolic polynomial definition with respect to vector e
p(x) = x[1]*x[2]*x[3] + x[1]*x[2]^2
e = 1.0*[1,1,1]

grad = x -> - 1/p(x) * ForwardDiff.gradient(x->p(x),x)
dpx = x -> ForwardDiff.gradient(x->p(x),x)
hess = x -> (-ForwardDiff.hessian(x -> p(x), x) * p(x) + dpx(x)*dpx(x)')/(p(x)^2)

""" 
Define hyperbolicity cone in terms of: 
    n           = cone dimension
    p           = hyperbolic polynomial
    grad        = gradient of polynomial's barrier function
    hess        = hessian of polynomial's barrier function
    e           = directional vector that polynomial p is hyperbolic to
    d           = [opt] degree of polynomial

Specifying argument d will use numerical approach in feasibility oracle.
"""
cone_test = Cones.Hyperbolicity{T}(n, p, grad, hess, e, d=3)

"""
Define the problem/model in native Hypatia interface in terms of the 
optimisation parameters: c, A, b, G, h and a list of cones. 

Note: the cones must be passed as a list with type Cones.Cone{T}
"""
model = Models.Model{T}(c, A, b, G, h, Cones.Cone{T}[cone_test])
solver = Solvers.Solver{T}(verbose = true);
Solvers.load(solver, model)
Solvers.solve(solver)
Solvers.get_status(solver)