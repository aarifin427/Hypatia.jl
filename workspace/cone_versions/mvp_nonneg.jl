"""
$(TYPEDEF)
Cloned from the src/Cones/nonnegative.jl file
ConeSample cone of dimension `dim`.
    $(FUNCTIONNAME){T}(dim::Int)
"""
mutable struct Conesample{T <: Real} <: Cone{T}
    dim::Int

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T}
    dder3::Vector{T}
    vec1::Vector{T}
    vec2::Vector{T}
    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    is_feas::Bool

    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    # These 3 are added
    hess_fact_updated::Bool
    hess_fact_mat::Symmetric{T, Matrix{T}}
    hess_fact::Factorization{T}

    function Conesample{T}(dim::Int) where {T <: Real}
        @assert dim >= 1
        cone = new{T}()
        cone.dim = dim
        return cone
    end
end
# used during setup_loaded
use_dual_barrier(::Conesample) = false
use_dder3(cone::Conesample) = false

reset_data(cone::Conesample) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.hess_fact_updated = false)

get_nu(cone::Conesample) = cone.dim

set_initial_point!(arr::AbstractVector, cone::Conesample) = (arr .= 1)

function update_feas(cone::Conesample{T}) where T
    @assert !cone.feas_updated
    # Feasibility check: check if all elements of the cone's point is > 0
    cone.is_feas = all(>(eps(T)), cone.point)
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::Conesample)
    @assert cone.is_feas
    @. cone.grad = -inv(cone.point)
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::Conesample{T}) where T
    cone.grad_updated || update_grad(cone)
    if !isdefined(cone, :hess)
        cone.hess = Symmetric(zeros(T, cone.dim, cone.dim))
    end
    x = @. abs2(cone.grad)
    for i=1:cone.dim
        cone.hess[i,i] = x[i]
    end
    cone.hess_updated = true
    return cone.hess
end