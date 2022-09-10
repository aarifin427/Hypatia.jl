"""
PASS

4D, 1 addend, init point is feasible (within constraints of Ax=b)

p(x) = x[1]*x[2]*x[3]*x[4]
init_point = 1/sqrt(6)*[1,1,1,1]
"""

using ForwardDiff
using Hypatia
using Hypatia.Cones
using Hypatia.Models
import Hypatia.Solvers
using LinearAlgebra

T = Float64;
n = 4;

A = 1.0*[0.1 1 1 0.1]
b = [0.5]
c = [
    0.6485574918878491;
    0.9051391876123972;
    0.5169347627896711;
    0.8591212789374901
]

G = Diagonal(-one(T) * I, n)
h = zeros(T, n)

# Nonnegative Cone setup
cone_control = Cones.Cone{T}[Cones.Nonnegative{T}(n)]
model = Models.Model{T}(c, A, b, G, h, cone_control)

solver = Solvers.Solver{T}(verbose = false);
Solvers.load(solver, model)
Solvers.solve(solver)
ans_control = Solvers.get_x(solver)

# hyperbolicity cone in terms of the specified p(x)
p(x) = x[1]*x[2]*x[3]*x[4]
init_point = 1/sqrt(6)*[1,1,1,1]

grad = x -> - 1/p(x) * ForwardDiff.gradient(x->p(x),x)
dpx = x -> ForwardDiff.gradient(x->p(x),x)
hess = x -> (-ForwardDiff.hessian(x -> p(x), x) * p(x) + dpx(x)*dpx(x)')/(p(x)^2)

cone_test = Cones.Cone{T}[Cones.Conesample{T}(n, p, grad, hess, init_point)]
model = Models.Model{T}(c, A, b, G, h, cone_test)

solver = Solvers.Solver{T}(verbose = false);
Solvers.load(solver, model)
Solvers.solve(solver)
ans_test = Solvers.get_x(solver)