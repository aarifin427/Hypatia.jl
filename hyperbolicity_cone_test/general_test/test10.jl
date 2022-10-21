using ForwardDiff
using Hypatia
using Hypatia.Cones
using Hypatia.Models
import Hypatia.Solvers
using LinearAlgebra
using SymPy

T = Float64;
n = 3;

A = [
    6.27898  1.38615   1.85652;
    7.99791  0.433622  6.81715;
]
A .*= 10
b = vec(sum(A, dims = 2))
c = [
    0.6485574918878491;
    0.9051391876123972;
    0.5169347627896711;
]

G = Diagonal(-one(T) * I, n)
h = zeros(T, n)

hp(x) = x[1]*x[2]*x[3]
e = 1.0*[1,1,1]

@vars t
p1 = x -> diff(hp(x + t*e), t).subs(t, 0)
p(x) = ForwardDiff.derivative(t -> hp(x + t*e), 0)

grad = x -> - 1/p(x) * ForwardDiff.gradient(x->p(x),x)
dpx = x -> ForwardDiff.gradient(x->p(x),x)
hess = x -> (-ForwardDiff.hessian(x -> p(x), x) * p(x) + dpx(x)*dpx(x)')/(p(x)^2)

# d=2 since degree decreases by 1 after differentiation
cone_test = Cones.Cone{T}[Cones.Hyperbolicity{T}(n, p1, grad, hess, e, d=2)]
model = Models.Model{T}(c, A, b, G, h, cone_test)

solver = Solvers.Solver{T}(verbose = true);
Solvers.load(solver, model)
Solvers.solve(solver)
Solvers.get_status(solver)