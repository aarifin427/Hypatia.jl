"""
PASS (optimality)

3D, 1st derivative of hyperbolic polynomials

"""

using ForwardDiff
using Hypatia
using Hypatia.Cones
using Hypatia.Models
import Hypatia.Solvers
using LinearAlgebra
using Symbolics

T = Float64;
n = 3;

A = 1.0*[1 1 1]
b = [3.0]
c = [
    0.6485574918878491;
    0.9051391876123972;
    0.5169347627896711;
]

G = Diagonal(-one(T) * I, n)
h = zeros(T, n)

hp(x) = x[1]*x[2]*x[3]
e = 1.0*[1,1,1]

"""
Feasibility oracle currently uses symbolic solver, thus 
this function must also use a symbolic differentiation too.

ForwardDiff's derivative method does not produce symbolic anonymous function.
"""
function p(x)
    Symbolics.@variables t1
    return substitute(Symbolics.derivative(hp(x + t1*e), t1), (Dict(t1 => 0)))
end

"""
Gradient and hessian oracles can be anonymous (symbolic or not)
"""
p_aux(x) = ForwardDiff.derivative(t -> hp(x + t*e), 0)
grad = x -> - 1/p_aux(x) * ForwardDiff.gradient(x->p_aux(x),x)
dpx = x -> ForwardDiff.gradient(x->p_aux(x),x)
hess = x -> (-ForwardDiff.hessian(x -> p_aux(x), x) * p_aux(x) + dpx(x)*dpx(x)')/(p_aux(x)^2)

cone_test = Cones.Conesample{T}(n, p, grad, hess, e)
model = Models.Model{T}(c, A, b, G, h, Cones.Cone{T}[cone_test])

solver = Solvers.Solver{T}(verbose=true);
Solvers.load(solver, model)
Solvers.solve(solver)
Solvers.get_status(solver)