"""
PASS

4D, vamos polynomial, 3 constraints
"""

using ForwardDiff
using Hypatia
using Hypatia.Cones
using Hypatia.Models
import Hypatia.Solvers
using LinearAlgebra
include("main_spd.jl")

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

A1 = [
    0 0 0 0 0 0 0
    0 1 2 0 1 0 0
    0 2 6 0 4 0 0
    0 0 0 4 0 0 0
    0 1 4 0 4 0 0
    0 0 0 0 0 0 0
    0 0 0 0 0 0 0
];

A2 = [
    1 1 2 1 0 2 1
    1 4 4 4 0 4 4
    2 4 9 4 0 8 4
    1 4 4 4 0 4 4
    0 0 0 0 0 0 0
    2 4 8 4 0 8 4
    1 4 4 4 0 4 4
];

A3 = [
    3 3 10 0 3 4 0
    3 4 12 0 4 4 0
    10 12 41 0 12 16 0
    0 0 0 0 0 0 0
    3 4 12 0 4 4 0
    4 4 16 0 4 8 0
    0 0 0 0 0 0 0
];

A4 = [
    2 2 2 0 0 0 0
    2 7 3 0 0 0 4
    2 3 4 0 0 0 0
    0 0 0 0 0 0 0
    0 0 0 0 0 0 0
    0 0 0 0 0 0 0
    0 4 0 0 0 0 4
];

# SPD
A_mat = [A1, A2, A3, A4]
status, X_sol, obj_value = jump_SDP(A_mat, A, b, c, n);
ans_control = []
for i in eachindex(X_sol)
    push!(ans_control, value(X_sol[i]))
end

# Hyperbolic
p(x) = x[1]^2 * x[2]^2 + 4*(x[1]+x[2]+x[3]+x[4])*(x[1]*x[2]*x[3] + x[1]*x[2]*x[4] + x[1]*x[3]*x[4] + x[2]*x[3]*x[4])
init_point = 1.0*[1,1,0,0]

grad = x -> - 1/p(x) * ForwardDiff.gradient(x->p(x),x)
dpx = x -> ForwardDiff.gradient(x->p(x),x)
hess = x -> (-ForwardDiff.hessian(x -> p(x), x) * p(x) + dpx(x)*dpx(x)')/(p(x)^2)

cone_test = Cones.Cone{T}[Cones.Hyperbolicity{T}(n, p, grad, hess, init_point, d=4)]
model = Models.Model{T}(c, A, b, G, h, cone_test)

solver = Solvers.Solver{T}(verbose = true);
Solvers.load(solver, model)
Solvers.solve(solver)
Solvers.get_status(solver)
ans_test = Solvers.get_x(solver)

println(norm(ans_control - ans_test))
