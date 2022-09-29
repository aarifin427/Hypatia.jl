"""
PASS

4D, 1 addend, more constraints

NOTE: 3 constraints tested
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
    6.27898  1.38615   1.85652  8.17268;
    7.99791  0.433622  6.81715  5.79162;
    2.42863  5.26224   8.98049  7.61971
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

# Nonnegative Cone setup
cone_control = Cones.Cone{T}[Cones.Nonnegative{T}(n)]
model = Models.Model{T}(c, A, b, G, h, cone_control)

solver = Solvers.Solver{T}(verbose = true);
Solvers.load(solver, model)
Solvers.solve(solver)
ans_control = Solvers.get_x(solver)

# # hyperbolicity cone in terms of the specified p(x)
p(x) = x[1]*x[2]*x[3]*x[4]
init_point = 1/sqrt(6)*[1,1,1,1]

grad = x -> - 1/p(x) * ForwardDiff.gradient(x->p(x),x)
dpx = x -> ForwardDiff.gradient(x->p(x),x)
hess = x -> (-ForwardDiff.hessian(x -> p(x), x) * p(x) + dpx(x)*dpx(x)')/(p(x)^2)

cone_test = Cones.Cone{T}[Cones.Hyperbolicity{T}(n, p, grad, hess, init_point, d=4)]
model = Models.Model{T}(c, A, b, G, h, cone_test)

solver = Solvers.Solver{T}(verbose = false);
Solvers.load(solver, model)
Solvers.solve(solver)
ans_test = Solvers.get_x(solver)

println(norm(ans_test-ans_control))