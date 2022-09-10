"""
Data structure

eg1: x^2 + y^2 + z^2
eg2: x*y + y*z + x*z

Scope:
- polynomial
- homogenous
- hyperbolic -> how to check?

ref: https://discourse.julialang.org/t/how-to-do-partial-derivatives/19869/3
ref: http://www.cecm.sfu.ca/~mmonagan/talks/eccad2013.pdf
"""

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
# TODO: test this matrix val Function

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

# # =====================================================#
# # Get jacobian matrix function rep
# # =====================================================#
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
# TODO: Test

# =====================================================#
# Sample Cone struct
# =====================================================#
mutable struct Conesample{T <: Real}
    dim::Int

    init::Union{Bool, Vector{T}}
    point::Vector{T}
    grad::Vector{T}
    p_function::Matrix{T}
    dp_function::Matrix{Matrix{T}}
    ddp_function::Matrix{Matrix{T}}
    dder3::Vector{T}

    # function Conesample{T}(dim::Int64, p_function::Matrix{T}) where {T <: Real}
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
# =====================================================#
# Initialize point
# =====================================================#
function set_initial_point!(arr::AbstractVector, cone::Conesample)
    if cone.init == false
        arr .= 1
        return arr
    else
        # TODO: assert that cone.init has same size or dimensions as arr input
        return cone.init
    end
end

# =====================================================#
# Gradient barrier function
# =====================================================#
function update_grad(cone::Conesample)
    cone.grad = -mat_eval(cone.dp_function, cone.point)/func_eval(cone.p_function, cone.point)
end

# =====================================================#
# Hessian barrier function to a vector
# =====================================================#
function compute_hess_prod(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::Conesample
    )
    hess = mat_eval(cone.ddp_function, cone.point)
    grad = mat_eval(cone.dp_function, cone.point)
    px = func_eval(cone.p_function, cone.point)
    k1, k2 = -1/px, 1/(px^2) * (grad'arr)
    prod = k1*(hess * arr) + k2*grad
    return prod
end

# =====================================================#
# Hessian barrier function
# =====================================================#
function compute_hess(
    cone::Conesample
    )
    cone.grad = -mat_eval(cone.dp_function, cone.point)/func_eval(cone.p_function, cone.point)
    px = func_eval(cone.p_function, cone.point) # evaluates p(x) where x = cone.point
    dpx = mat_eval(cone.dp_function, cone.point)
    d2px = mat_eval(cone.ddp_function, cone.point) # evaluates nabla^2_p(x) or the hess of p(x) at x = cone.point
    return (-px * d2px + dpx * dpx')/(px^2)
end

###
# multiply polynomial representation by a coefficient
###
function mult_coeff_to_polynomial(
    poly_rep::Matrix{T},
    k::T
    ) where T <: Real
    _, col = size(poly_rep)
    for j = 1:col
        poly_rep[1, j] *= k
    end
    return poly_rep
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