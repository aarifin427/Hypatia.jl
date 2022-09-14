"""
PASS

4D, 1 addend, init point is infeasible (not within constraints of Ax = b)

NOTE: same as "nonneg/test2.jl", testing if changing initial point's multiplier fixes "mu" problems
"""

using ForwardDiff
using Hypatia
using Hypatia.Cones
using Hypatia.Models
import Hypatia.Solvers
using LinearAlgebra

T = Float64;
n = 4;

A = 1.0*[0 1 1 0]
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
init_point = 1.0*[1,1,1,1]

cone_test = Cones.Cone{T}[Cones.Conesample{T}(n, p, grad, hess, init_point)]
model = Models.Model{T}(c, A, b, G, h, cone_test)

solver = Solvers.Solver{T}(verbose = false);
Solvers.load(solver, model)
Solvers.solve(solver)
ans_test = Solvers.get_x(solver)