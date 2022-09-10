using SparseArrays
using Hypatia
using Hypatia.Cones
using Hypatia.Models
import Hypatia.Solvers
using LinearAlgebra
using ForwardDiff

# Vamos polynomial representation
f1(x) = x[1]^2 * x[2]^2 + 4*(x[1]+x[2]+x[3]+x[4])*(x[1]*x[2]*x[3] + x[1]*x[2]*x[4] + x[1]*x[3]*x[4] + x[2]*x[3]*x[4])
p1 = [
    1 4 4 4 4 4 4 16 4 4 4 4 4 4
    2 2 2 2 1 1 1 1  1 1 1 0 0 0
    2 1 1 0 2 2 1 1  1 0 0 2 1 1
    0 1 0 1 1 0 2 1  0 2 1 1 2 1
    0 0 1 1 0 1 0 1  2 1 2 1 1 2
];
p1 = 1.0*p1;
init_point = [1,1,0,0]
# init_point = [2.24*10^-5, 1.1*10^-5, 0.3, 9.5*10^-6]

"""
Formulation:
min c'*x
s.t.
    b - A*x = 0
    h - G*x in K

where K is the hyperbolicity cone
b = 1 (scalar)
A = [0 1 1 0] (whatever you want it to be so that the solution isn't infeasible or just 0's)
h = 0
G = -I
c = weights

Simplified formulation:
min c'*x
s.t.
    1 - (x2 + x3) = 0
    x in K
"""

# (1) For pure lib only
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

grad = x -> ForwardDiff.gradient(x -> -log(f1(x)), x)
hess = x -> ForwardDiff.hessian(x -> -log(f1(x)), x)
cones = Cones.Cone{T}[Cones.Conesample{T}(n, f1, grad, hess, init_point)]

model = Models.Model{T}(c, A, b, G, h, cones)

solver = Solvers.Solver{T}(verbose = true);
Solvers.load(solver, model);
Solvers.solve(solver);
Solvers.get_status(solver)
Solvers.get_primal_obj(solver)
Solvers.get_dual_obj(solver)
Solvers.get_x(solver)

# # (2) For pure lib only
# T = Float64;
# n = 4;

# A = [3.10373  0.683718  8.51255  9.66848]
# A .*= 100
# b = [sum(A)]
# c = [
#     0.6485574918878491;
#     -0.9051391876123972;
#     -0.5169347627896711;
#     0.8591212789374901
# ]

# G = Diagonal(-one(T) * I, n)
# h = zeros(T, n)

# grad = x -> ForwardDiff.gradient(x -> -log(f1(x)), x)
# hess = x -> ForwardDiff.hessian(x -> -log(f1(x)), x)
# cones = Cones.Cone{T}[Cones.Conesample{T}(n, f1, grad, hess, init_point)]

# model = Models.Model{T}(c, A, b, G, h, cones)
# solver = Solvers.Solver{T}(verbose = true, tol_slow=32);
# Solvers.load(solver, model);
# Solvers.solve(solver);
# Solvers.get_status(solver)
# Solvers.get_primal_obj(solver)
# Solvers.get_dual_obj(solver)
# Solvers.get_x(solver)

# # With mat rep
# T = Float64;
# n = 4;

# A = 1.0*[0 1 1 0]
# b = [3.0]
# c = 1.0*[3; -4; -2; 1]

# G = Diagonal(-one(T) * I, n)
# h = zeros(T, n)

# cones = Cones.Cone{T}[Cones.Conesample{T}(n, p1, f1, init_point)]

# model = Models.Model{T}(c, A, b, G, h, cones)

# solver = Solvers.Solver{T}(verbose = true);
# Solvers.load(solver, model);
# Solvers.solve(solver);
# Solvers.get_status(solver)
# Solvers.get_primal_obj(solver)
# Solvers.get_dual_obj(solver)
# Solvers.get_x(solver)