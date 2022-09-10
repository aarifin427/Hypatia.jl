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

(m, n, nz_frac) = (2, 4, 1.0)
@assert 0 < nz_frac <= 1

T = Float64;

# generate random data
# A = (nz_frac >= 1) ? rand(T, m, n) : sprand(FloatT64, m, n, nz_frac)
A = [
    3.10373  0.683718  8.51255  9.66848;
    7.44653  4.25423   2.48018  9.67681
]
A .*= 10
b = vec(sum(A, dims = 2))
# c = rand(T, n)
c = [
    0.6485574918878491;
    0.9051391876123972;
    0.5169347627896711;
    0.8591212789374901
]

G = Diagonal(-one(T) * I, n) # TODO uniformscaling
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
cone0 = Cones.Cone{T}[Cones.Conesample{T}(n)]

model = Models.Model{T}(c, A, b, G, h, cone0)

solver = Solvers.Solver{T}(verbose = true);
Solvers.load(solver, model);
Solvers.solve(solver);
Solvers.get_status(solver)
Solvers.get_primal_obj(solver)
Solvers.get_dual_obj(solver)
a = Solvers.get_x(solver)

println(norm(a - ans))