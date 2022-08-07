"""
$(TYPEDEF)

Cloned from the src/Cones/nonnegative.jl file

ConeSample cone of dimension `dim`.

    $(FUNCTIONNAME){T}(dim::Int)
"""

# The struct or abstract type definition 
mutable struct Conesample{T <: Real} <: Cone{T}
    dim::Int

    init::Union{Bool, Vector{T}}
    p_function::Matrix{T}
    dp_function::Matrix{Matrix{T}}
    ddp_function::Matrix{Matrix{T}}

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T}
#     dder3::Vector{T}
    vec1::Vector{T}
    vec2::Vector{T}
    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    # p_function is matrix rep of a function
    function Conesample{T}(dim::Int, p_function::Matrix{T}, init::Union{Bool, Vector{T}}=false) where {T <: Real}
        @assert dim >= 1
        cone = new{T}()
        cone.dim = dim
        cone.init = init
        cone.p_function = p_function
        cone.dp_function = compute_grad_rep(cone.p_function)
        cone.ddp_function = compute_hess_rep(cone.dp_function)
        return cone
    end
end

use_dual_barrier(::Conesample) = false

use_sqrt_hess_oracles(::Int, cone::Conesample) = false

use_dder3(::Cone)::Bool = false

reset_data(cone::Conesample) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = false)

use_sqrt_hess_oracles(::Int, cone::Conesample) = false

get_nu(cone::Conesample) = cone.dim


# used in the following stack:

function set_initial_point!(arr::AbstractVector, cone::Conesample)
    if cone.init == false
        arr .= 1
        return arr
    else
        # TODO: assert that cone.init has same size or dimensions as arr input
        return cone.init
    end
end

function update_feas(cone::Conesample{T}) where T
    @assert !cone.feas_updated

    # Feasibility check: check if all elements of the cone's point is > 0
    cone.is_feas = all(>(eps(T)), cone.point)

    cone.feas_updated = true
    return cone.is_feas
end

is_dual_feas(cone::Conesample{T}) where T = all(>(eps(T)), cone.dual_point)

# =====================================================#
# Function representation to value
# =====================================================#
function func_eval(p::Matrix{T}, x::Vector{T}) where {T <: Real}
    res = 0 # result
    (row, col) = size(p)
    for j = 1:col
        addend = p[1,j]
        for i = 2:row
            addend *= x[i-1]^p[i,j]
        end
        res += addend
    end
    return res
end

# =====================================================#
# Matrix representation to value
# =====================================================#
function mat_eval(amat::Matrix{Matrix{T}}, x::Vector{T}) where {T <: Real}
    row, col = size(amat)
    if col == 1
        res = Vector{T}(undef, row)
    else
        res = Matrix{T}(undef, row, col)
    end
    for i = 1:row
        for j = 1:col
            res[i,j] = func_eval(amat[i,j], x)
        end
    end
    return res
end

# =====================================================#
# Partially differentiate
# =====================================================#
function partial_diff(p::Matrix{T}, idx::Int) where {T <: Real}
    """
    Get partial derivative of p(x) in terms of x_idx or the idx-th variable.
    """
    row, col = size(p)
    res = Matrix{T}(undef, row, 1)
    for j = 1:col
        if p[idx+1, j] == 0
            continue
        end
        vec = p[:, j]
        vec[1,1] *= vec[idx+1, 1]
        vec[idx+1, 1] -= 1
        res = hcat(res, vec)
    end
    return res[:, 2:end]
end

# =====================================================#
# Get grad matrix function rep
# =====================================================#
function compute_grad_rep(p::Matrix{T}) where {T <: Real}
    row, _ = size(p)
    dp = Matrix{Matrix{T}}(undef, row-1, 1)
    for i = 1:row-1
        dp[i,1] = partial_diff(p, i)
    end
    return dp
end

# =====================================================#
# Get jacobian matrix function rep
# =====================================================#
function compute_hess_rep(dp::Matrix{Matrix{T}}) where {T <: Real}
    row, _ = size(dp)
    ddp = Matrix{Matrix{T}}(undef, row, row)
    for i = 1:row
        for j = 1:row
            ddp[i,j] = partial_diff(dp[i,1], j)
        end
    end
    return ddp
end

# =====================================================#
# Gradient barrier function
# =====================================================#
function update_grad(cone::Conesample)
    @assert cone.is_feas
    @. cone.grad = -mat_eval(cone.dp_function, cone.point)/func_eval(cone.p_function, cone.point)
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::Conesample{T}) where T
    cone.grad_updated || update_grad(cone)
    @. cone.hess.diag = mat_eval(cone.ddp_function, cone.point)
    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::Conesample,
    )
    @assert cone.is_feas
    hess = cone.hess
    grad = cone.grad
    px = func_eval(cone.p_function, cone.point)
    k1, k2 = -1/px, 1/(px^2) * (grad'arr)
    @. prod = k1*(hess * arr) + k2*grad
    return prod
end

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

# used during setup_loaded
use_dual_barrier(::Conesample) = false

reset_data(cone::Conesample) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = false)

use_sqrt_hess_oracles(::Int, cone::Conesample) = true

get_nu(cone::Conesample) = cone.dim


# used in the following stack:

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
"""