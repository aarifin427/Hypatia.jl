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