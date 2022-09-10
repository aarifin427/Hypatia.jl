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

    init::Vector{T}
    p_function::Matrix{T}
    dp_function::Matrix{Matrix{T}} # this is nabla_p(x) as matrix rep
    ddp_function::Matrix{Matrix{T}} # this is nabla^2_p(x) as matrix rep
    f::Any

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
    function Conesample{T}(dim::Int, p_function::Matrix{T}, f::Any, init::AbstractVector) where {T <: Real}
        @assert dim >= 1
        cone = new{T}()
        # cone.use_dual_barrier = false
        cone.dim = dim
        cone.init = convert(T,1)*init
        cone.p_function = p_function
        cone.dp_function = compute_grad_rep(cone.p_function)
        cone.ddp_function = compute_hess_rep(cone.dp_function)
        cone.f = f
        return cone
    end
end

use_dual_barrier(::Conesample) = false

reset_data(cone::Conesample) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.hess_fact_updated = false)

use_sqrt_hess_oracles(::Int, cone::Conesample) = false

get_nu(cone::Conesample) = cone.dim

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
        if !isreal(a[i]) || a[i] < eps(T)
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
    # outputs gradient of barrier function -log(p(x)) where x = cone.point
    @assert cone.is_feas
    cone.grad = -mat_eval(cone.dp_function, cone.point)/func_eval(cone.p_function, cone.point)
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::Conesample{T}) where T
    # output hessian of barrier function -log(p(x)) where x = cone.point
    px = func_eval(cone.p_function, cone.point) # evaluates p(x) where x = cone.point
    dpx = mat_eval(cone.dp_function, cone.point)
    d2px = mat_eval(cone.ddp_function, cone.point) # evaluates nabla^2_p(x) or the hess of p(x) at x = cone.point
    cone.hess = Symmetric((-px * d2px + dpx * dpx')/(px^2))
    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::Conesample,
    )
    # hess product H(x).arr, where H(x) is the hessian of barrier function, x = cone.point
    @assert cone.is_feas
    cone.hess_updated || update_hess(cone) # TODO
    # hess = cone.hess
    # grad = cone.grad
    # px = func_eval(cone.p_function, cone.point)
    # k1, k2 = -1/px, 1/(px^2) * (grad'arr)
    # prod = k1*(hess * arr) + k2*grad
    px = func_eval(cone.p_function, cone.point)
    dpx = mat_eval(cone.dp_function, cone.point)
    d2px = mat_eval(cone.ddp_function, cone.point)
    k1, k2 = -1/px, 1/(px^2)
    prod = k1*(d2px * arr) + k2*(dpx * dpx' * arr)
    return prod
end

# =====================================================#
# Dder3 function
# =====================================================#
function dder3(
    cone::Conesample, 
    dir::AbstractVector
    )
    """
    This function evaluates v'*hessF(x)*v where hessF(x) is the hessian
        of the barrier function or hessF(x) = hess(-log(p(x)))
    
    v'*hessF(x)*v is expressed as a(x)/b(x) where:
        a(x) = -p(x) * v' * hess(p(x)) * v + (v * grad(p(x))')^2
        b(x) = (p(x))^2
    
    break up the expressions:
        a(x) = term1(x) + term2(x), where
            term1(x) = -px * vd2pxv, where
                px = p(x)
                vd2pxv = v' * hess(p(x)) * v
            term2(x) = (vdpx)^2, where
                vdpx = v * grad(p(x))'
        dax = grad(a(x)) = -vd2pxv * dpx - px * dvd2pxv + 2*vdpx*dvdpx, where
            dvd2pxv = grad(v' * hess(p(x)) * v) in terms of x
        dbx = 2*px*dpx, where
            dpx = grad(p(x))
    """
    # @assert cone.hess_updated
    px = func_eval(cone.p_function, cone.point) # evaluates p(x) where x = cone.point
    dpx = mat_eval(cone.dp_function, cone.point)
    d2px = mat_eval(cone.ddp_function, cone.point) # evaluates nabla^2_p(x) or the hess of p(x) at x = cone.point
    n = cone.dim
    v = dir

    # b(x=cone.point)
    bx_val = px^2
    # a(x=cone.point)
    ax_val = -px*(v' * d2px * v) + (v' * dpx)^2

    T = typeof(cone.point[1])
    ################################################
    # Construct grad of a(x), first term
    ################################################
    multiplier = v * v'

    # compute v'*(∇2p(x)*v) at x = cone.point
    # this is equivalent to element by element mult of (v * v').(∇2p(x)) where x = cone.point where multiplier = v*v'
    vd2pxv = 0
    for i = 1:n
        for j = 1:n
            vd2pxv += multiplier[i,j]*d2px[i,j]
        end
    end

    # compute d/dx(v'*(∇2p(x)*v)) at x = cone.point
    dvd2pxv = zeros(T, 4)
    for k = 1:n
        sum = 0
        # element by element multiplication of (v * v').Y, where Y = d/dx_k(∇2p(x))
        for i = 1:n
            for j = 1:n
                sum += multiplier[i,j]*func_eval(partial_diff(cone.ddp_function[i,j], k), cone.point)
            end
        end
        dvd2pxv[k] = sum
    end

    ################################################
    # Construct grad of a(x), second term
    ################################################
    # compute v'*∇p(x) at x = cone.point
    # this is equivalent to element by element multiplication of v.∇p(x)
    vdpx = 0
    for i = 1:n
        vdpx += v[i]*dpx[i]
    end

    # compute d/dx_i(v'*∇p(x)) at x = cone.point
    dvdpx = zeros(T, 4)
    for i = 1:n
        for j = 1:n
            dvdpx[i] += v[j] * d2px[i,j]
        end
    end

    ################################################
    # Construct grad of a(x), altogether
    ################################################
    dax = -vd2pxv * dpx - px * dvd2pxv + 2*vdpx*dvdpx

    ################################################
    # Construct grad of b(x)
    ################################################
    dbx = 2*px*dpx

    ################################################
    # Construct dder3
    ################################################
    dder3 = 1/bx_val * dax - ax_val/bx_val^2 * dbx
    return dder3
end