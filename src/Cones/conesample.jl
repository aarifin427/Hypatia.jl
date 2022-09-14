"""
$(TYPEDEF)

Cloned from the src/Cones/nonnegative.jl file

ConeSample cone of dimension `dim`.

    $(FUNCTIONNAME){T}(dim::Int)
"""
# The struct or abstract type definition 
mutable struct Conesample{T <: Real} <: Cone{T}
    # use_dual_barrier::Bool
    dim::Int
    nu::T

    init::Vector{T}
    f::Any  # p(x) function abstract
    barrier_grad_f::Any # grad(-log(p(x)))
    barrier_hess_f::Any # hess(-log(p(x)))

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T} # this stores gradient of barrier function = - nabla_p(x)/p(x) at point x
    dder3::Vector{T}
    vec1::Vector{T}
    vec2::Vector{T}
    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_fact_updated::Bool # what dis?
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_mat::Symmetric{T, Matrix{T}} # what dis?
    hess_fact::Factorization{T} # what dis?

    # p_function is matrix rep of a function
    function Conesample{T}(dim::Int, f::Any, barrier_grad_f::Any, barrier_hess_f::Any, init::AbstractVector) where {T <: Real}
        @assert dim >= 1
        cone = new{T}()
        cone.dim = dim
        cone.f = f
        cone.barrier_grad_f = barrier_grad_f
        cone.barrier_hess_f = barrier_hess_f
        cone.init = convert(T,1)*init
        cone.nu = -cone.init'*cone.barrier_grad_f(cone.init)
        return cone
    end
end

use_dual_barrier(::Conesample) = false

reset_data(cone::Conesample) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.hess_fact_updated = false)

use_sqrt_hess_oracles(::Int, cone::Conesample) = false

get_nu(cone::Conesample) = cone.nu

function set_initial_point!(arr::AbstractVector, cone::Conesample)
    arr .= cone.init
    return arr;
end

# TODO
function update_feas(cone::Conesample{T}) where T
    @assert !cone.feas_updated

    λ = symbols("λ")
    e = cone.init
    x = cone.point
    a = solve(cone.f(λ*e-x))

    cone.is_feas = true
    for i in eachindex(a)
        if abs(imag(a[i])) > eps(T) || abs(a[i]) < eps(T)
            # not real, infeasible (?)
            cone.is_feas = false
            cone.feas_updated = true
            return cone.is_feas
        end
    end
    cone.feas_updated = true
    return cone.is_feas
end

# =====================================================#
# Gradient barrier function
# =====================================================#
function update_grad(cone::Conesample)
    # outputs gradient of barrier function -log(p(x)) where x = cone.point
    @assert cone.is_feas
    cone.grad = cone.barrier_grad_f(cone.point)
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::Conesample{T}) where T
    cone.grad_updated || update_grad(cone)
    # output hessian of barrier function -log(p(x)) where x = cone.point
    cone.hess = Symmetric(cone.barrier_hess_f(cone.point))
    cone.hess_updated = true
    return cone.hess
end

use_dder3(cone::Conesample) = false

# function dder3(cone::Conesample, dir::AbstractVector) 
#     dr3 = x -> ForwardDiff.gradient(x -> dir'*cone.barrier_hess_f(x)*dir, x)
#     cone.dder3 = dr3(cone.point)
#     return cone.dder3
# end