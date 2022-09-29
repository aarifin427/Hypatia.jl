"""
$(TYPEDEF)

Cloned from the src/Cones/nonnegative.jl file

ConeSample cone of dimension `dim`.

    $(FUNCTIONNAME){T}(dim::Int)
"""
mutable struct Conesample{T <: Real} <: Cone{T}
    dim::Int
    nu::T
    d::Int

    init::Vector{T}
    p::Any  # p(x) function abstract
    barrier_grad_f::Any # grad(-log(p(x)))
    barrier_hess_f::Any # hess(-log(p(x)))

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T} # this stores value of gradient of barrier function at cone.point
    dder3::Vector{T}
    vec1::Vector{T}
    vec2::Vector{T}
    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_mat::Symmetric{T, Matrix{T}}
    hess_fact::Factorization{T}

    function Conesample{T}(dim::Int, p::Any, barrier_grad_f::Any, barrier_hess_f::Any, init::AbstractVector; d::Int=-1) where {T <: Real}
        @assert dim >= 1
        cone = new{T}()
        cone.dim = dim
        cone.d = d
        cone.p = p
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

function polyroot(d::Int, f::Any)
    lambda_set = [] # should be d+1 elements
    for i = 0:d
        push!(lambda_set, exp(2*pi*im*i/(d+1)))
    end    

    f_lambda_set = ComplexF64[]
    for i = 0:d
        push!(f_lambda_set, f(lambda_set[i+1]))
    end

    mat = zeros(ComplexF64, d+1,d+1)
    for i = 1:d + 1
        for j = 1:d + 1
            mat[i,j] = lambda_set[i]^(j-1)
        end
    end

    coeffs = mat^-1*f_lambda_set

    return Complex{Float64}.(PolynomialRoots.roots(big.(coeffs)))
end

function update_feas(cone::Conesample{T}) where T
    @assert !cone.feas_updated

    # for debugging
    if cone.feas_bypass
        cone.is_feas = true
        cone.feas_updated = true
        return cone.is_feas
    end

    e = cone.init
    x = cone.point/norm(cone.point)
    
    f(λ) = cone.p(λ*e-x)

    if cone.d > 0
        a = polyroot(cone.d, f)
        tol = 1e-7
    else
        @vars t
        a = solve(cone.p(t*e-x))
        tol = 1e-7
    end

    cone.is_feas = true
    for i in eachindex(a)
        # if roots' sign is negative, point not in the cone        
        if abs(f(real(a[i]))) > tol || sign(real(a[i])) < 0
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