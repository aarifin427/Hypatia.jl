"""
$(TYPEDEF)

Cloned from the src/Cones/nonnegative.jl file

ConeSample cone of dimension `dim`.

    $(FUNCTIONNAME){T}(dim::Int)
"""

# The struct or abstract type definition 
"""
Contains:
- Dimensions
- Point (???)
- Dual_point (???)
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
    hess::Diagonal{T, Vector{T}}
    inv_hess::Diagonal{T, Vector{T}}

    # TODO: What is this function for? Context of defining function in structs?
    function Conesample{T}(dim::Int) where {T <: Real}
        @assert dim >= 1
        cone = new{T}()
        cone.dim = dim
        return cone
    end
end

use_dual_barrier(::Conesample) = false

reset_data(cone::Conesample) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = false)

use_sqrt_hess_oracles(::Int, cone::Conesample) = true

get_nu(cone::Conesample) = cone.dim

set_initial_point!(arr::AbstractVector, cone::Conesample) = (arr .= 1)

function update_feas(cone::Conesample{T}) where T
    @assert !cone.feas_updated

    # Feasibility check: check if all elements of the cone's point is > 0
    cone.is_feas = all(>(eps(T)), cone.point)

    cone.feas_updated = true
    return cone.is_feas
end

is_dual_feas(cone::Conesample{T}) where T = all(>(eps(T)), cone.dual_point)

function update_grad(cone::Conesample)
    @assert cone.is_feas
    @. cone.grad = -inv(cone.point)
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::Conesample{T}) where T
    cone.grad_updated || update_grad(cone)
    if !isdefined(cone, :hess)
        cone.hess = Diagonal(zeros(T, cone.dim))
    end

    @. cone.hess.diag = abs2(cone.grad)
    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess(cone::Conesample{T}) where T
    @assert cone.is_feas
    if !isdefined(cone, :inv_hess)
        cone.inv_hess = Diagonal(zeros(T, cone.dim))
    end

    @. cone.inv_hess.diag = abs2(cone.point)
    cone.inv_hess_updated = true
    return cone.inv_hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::Conesample,
    )
    @assert cone.is_feas
    @. prod = arr / cone.point / cone.point
    return prod
end

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::Conesample,
    )
    @assert cone.is_feas
    @. prod = arr * cone.point * cone.point
    return prod
end

function sqrt_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::Conesample,
    )
    @assert cone.is_feas
    @. prod = arr / cone.point
    return prod
end

function inv_sqrt_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::Conesample,
    )
    @assert cone.is_feas
    @. prod = arr * cone.point
    return prod
end

function dder3(cone::Conesample, dir::AbstractVector)
    @. cone.dder3 = abs2(dir / cone.point) / cone.point
    return cone.dder3
end

hess_nz_count(cone::Conesample) = cone.dim
hess_nz_count_tril(cone::Conesample) = cone.dim
inv_hess_nz_count(cone::Conesample) = cone.dim
inv_hess_nz_count_tril(cone::Conesample) = cone.dim
hess_nz_idxs_col(cone::Conesample, j::Int) = [j]
hess_nz_idxs_col_tril(cone::Conesample, j::Int) = [j]
inv_hess_nz_idxs_col(cone::Conesample, j::Int) = [j]
inv_hess_nz_idxs_col_tril(cone::Conesample, j::Int) = [j]

# nonnegative is not primitive, so sum and max proximity measures differ
function get_proxsqr(
    cone::Conesample{T},
    irtmu::T,
    use_max_prox::Bool,
    ) where {T <: Real}
    aggfun = (use_max_prox ? maximum : sum)
    return aggfun(abs2(si * zi * irtmu - 1) for (si, zi) in
        zip(cone.point, cone.dual_point))
end
