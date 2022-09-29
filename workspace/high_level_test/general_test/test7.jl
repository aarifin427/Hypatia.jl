"""
PASS (optimality)

4D, 4 addends, 2 constraints
"""

using ForwardDiff
using Hypatia
using Hypatia.Cones
using Hypatia.Models
import Hypatia.Solvers
using LinearAlgebra

T = Float64;
n = 4;

A = [
    6.27898  1.38615   1.85652  -1.17268;
    7.99791  0.433622  6.81715  5.79162;
]
A .*= 10
b = vec(sum(A, dims = 2))
c = [
    0.6485574918878491;
    0.9051391876123972;
    0.5169347627896711;
    0.8591212789374901
]

G = Diagonal(-one(T) * I, n)
h = zeros(T, n)

p(x) = x[1]^2 - x[2]^2 - x[3]^2 - x[4]^2
e = 1.0*[1,0,0,0]

grad = x -> - 1/p(x) * ForwardDiff.gradient(x->p(x),x)
dpx = x -> ForwardDiff.gradient(x->p(x),x)
hess = x -> (-ForwardDiff.hessian(x -> p(x), x) * p(x) + dpx(x)*dpx(x)')/(p(x)^2)

cone_test = Cones.Hyperbolicity{T}(n, p, grad, hess, e, d=2)
model = Models.Model{T}(c, A, b, G, h, Cones.Cone{T}[cone_test])

solver = Solvers.Solver{T}(verbose = true);
Solvers.load(solver, model)
Solvers.solve(solver)
Solvers.get_status(solver)