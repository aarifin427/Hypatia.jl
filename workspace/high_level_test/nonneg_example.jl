"""
Source:
(1) https://github.com/chriscoey/Hypatia.jl/blob/master/examples/linearopt/native.jl
(2) https://github.com/chriscoey/Hypatia.jl/blob/master/examples/linearopt/native_test.jl
(3) https://chriscoey.github.io/Hypatia.jl/dev/solving/
"""

using SparseArrays
using Hypatia
using Hypatia.Cones
using Hypatia.Models
import Hypatia.Solvers
using LinearAlgebra
using ForwardDiff

"""
Example:
p(x) = x1*x2*x3
e = (1,1,1)
{(x1, x2, x3): x1>=0 , x2>=0, x3>=0}
"""
T = Float64;
n = 4;

# A = [
#     3.10373  0.683718  8.51255  9.66848;
#     7.44653  4.25423   2.48018  9.67681
# ]
# A .*= 10
# b = vec(sum(A, dims = 2))

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

cones = Cones.Cone{T}[Cones.Nonnegative{T}(n)]

model = Models.Model{T}(c, A, b, G, h, cones)

solver = Solvers.Solver{T}(verbose = true);
Solvers.load(solver, model);
Solvers.solve(solver);
Solvers.get_status(solver)
Solvers.get_primal_obj(solver)
Solvers.get_dual_obj(solver)
ans = Solvers.get_x(solver)

####
p(x) = x[1]*x[2] + x[1]*x[3] + x[1]*x[4] + x[2]*x[3] + x[2]*x[4] + x[3]*x[4]
# p(x) = x[1]*x[2]*x[3]*x[4]
init_point = 1/sqrt(6)*[1,1,1,1]

grad = x -> ForwardDiff.gradient(x -> -log(p(x)), x)
hess = x -> ForwardDiff.hessian(x -> -log(p(x)), x)

cone0 = Cones.Cone{T}[Cones.Conesample{T}(n, p, grad, hess, init_point)]

model = Models.Model{T}(c, A, b, G, h, cone0)

solver = Solvers.Solver{T}(verbose = true);
Solvers.load(solver, model);
Solvers.solve(solver);
Solvers.get_status(solver)
Solvers.get_primal_obj(solver)
Solvers.get_dual_obj(solver)
a = Solvers.get_x(solver)

println(norm(a - ans))


println("This is Nonnegative answers: ")
println(ans)

println()
println("This is Conesample answers: ")
println(a)