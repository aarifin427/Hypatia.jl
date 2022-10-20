using ForwardDiff
using Hypatia
using Hypatia.Cones
using Hypatia.Models
import Hypatia.Solvers
using LinearAlgebra
using JuMP
using SCS
include("main_spd.jl")

T = Float64;
n = 3;

A = 1.0*[0.1 1 1]
b = [0.5]
c = [
    0.6485574918878491;
    0.9051391876123972;
    0.5169347627896711;
]

G = Diagonal(-one(T) * I, n)
h = zeros(T, n)

d = 3
A1 = I(d)
A2 = [  
    19 32 24 ;
    32 54 41 ;
    24 41 38  
]
A3 = [        
    18 11 25 ;
    11 9 21 ; 
    25 21 57  
]
# SDP
A_mat = [A1, A2, A3]
status, X_sol, obj_value = jump_SDP(A_mat, A, b, c, n);
ans_control = []
for i in eachindex(X_sol)
    push!(ans_control, value(X_sol[i]))
end

# Hyperbolic
p(x) = det(I(d)*x[1] + A2*x[2] + A3*x[3])
init_point = 1.0*[1,0,0]

grad = x -> - 1/p(x) * ForwardDiff.gradient(x->p(x),x)
dpx = x -> ForwardDiff.gradient(x->p(x),x)
hess = x -> (-ForwardDiff.hessian(x -> p(x), x) * p(x) + dpx(x)*dpx(x)')/(p(x)^2)

# d = the first dimension of square matrix
cone_test = Cones.Cone{T}[Cones.Hyperbolicity{T}(n, p, grad, hess, init_point)]
model = Models.Model{T}(c, A, b, G, h, cone_test)

solver = Solvers.Solver{T}(verbose = true);
Solvers.load(solver, model)
Solvers.solve(solver)
ans_test = Solvers.get_x(solver)