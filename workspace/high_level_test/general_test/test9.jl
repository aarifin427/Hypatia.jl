
using ForwardDiff
using Hypatia
using Hypatia.Cones
using Hypatia.Models
import Hypatia.Solvers
using LinearAlgebra
using PolynomialRoots
using Roots
include("graph_hyperbolic.jl")

T = Float64;
n = 8;

A = [
    0.302846  0.415007   0.036932  0.487096  0.00754302  0.382551  0.284391   0.707595
    0.029501  0.443038   0.548903  0.092135  0.397634    0.518824  0.0312923  0.357963
    0.698182  0.337618   0.23081   0.605408  0.854847    0.777123  0.0728251  0.0348335
    0.815636  0.0530244  0.455034  0.412178  0.827097    0.140736  0.53434    0.536223
]
A .*= 10
b = vec(sum(A, dims = 2))
c = [
    0.38408576929026117;
    0.8308981967744447;
    0.9166491510620962;
    0.3151943865574789;
    0.9929631611060344;
    0.40282991488422304;
    0.8076716064355823;
    0.20674598984176895
]

G = Diagonal(-one(T) * I, n)
h = zeros(T, n)

"""
Example: Q3 (3D hypercube), all weights are 1
polynomial is 8D (2^3 = 8)
"""
weighted_edge_set = [
    [1,2,1],
    [2,3,1],
    [3,4,1],
    [1,4,1],
    [5,6,1],
    [6,7,1],
    [7,8,1],
    [5,8,1],
    [1,5,1],
    [2,6,1],
    [3,7,1],
    [4,8,1]
]

"""
Example: Q_3 (3D hypercube), all weights are 1
polynomial is in 8D (n = 2^3 = 8)
highest power d = 4 ∀ graphs with r = 2
"""
d = 4
p(x) = get_p(n, weighted_edge_set, x)
e = 1.0*[1,1,1,1,1,1,1,1]

grad = x -> - 1/p(x) * ForwardDiff.gradient(x->p(x),x)
dpx = x -> ForwardDiff.gradient(x->p(x),x)
hess = x -> (-ForwardDiff.hessian(x -> p(x), x) * p(x) + dpx(x)*dpx(x)')/(p(x)^2)

cone_test = Cones.Hyperbolicity{T}(n, p, grad, hess, e, d = 4)
model = Models.Model{T}(c, A, b, G, h, Cones.Cone{T}[cone_test])

solver = Solvers.Solver{T}(verbose = true);
Solvers.load(solver, model)
Solvers.solve(solver)
Solvers.get_status(solver)

# function polyroot(d::Int, f::Any)
#     lambda_set = [] # should be d+1 elements
#     for i = 0:d
#         push!(lambda_set, exp(2*pi*im*i/(d+1)))
#     end    

#     f_lambda_set = ComplexF64[]
#     for i = 0:d
#         push!(f_lambda_set, f(lambda_set[i+1]))
#     end

#     mat = zeros(ComplexF64, d+1,d+1)
#     for i = 1:d + 1
#         for j = 1:d + 1
#             mat[i,j] = lambda_set[i]^(j-1)
#         end
#     end

#     coeffs = mat^-1*f_lambda_set

#     return coeffs
# end

# f(λ) = p(λ*e-e)
# coeffs = polyroot(4,f)
# a0 = Complex{Float64}.(PolynomialRoots.roots(big.(coeffs)))
# for i in eachindex(a0)
#     a0[i] = find_zero(f, real(a0[i]))
# end
# a0
