"""
$(TYPEDEF)

Cloned from the src/Cones/nonnegative.jl file

Hyperbolicity cone of dimension `dim`.

    $(FUNCTIONNAME){T}(dim::Int)
"""
mutable struct Hyperbolicity{T <: Real} <: Cone{T}
    dim::Int        # dimension of the problem, equal to size of initial point
    nu::T           # barrier parameter
    d::Int          # degree of polynomial

    init::Vector{T}         # initial point
    p::Any                  # p(x) hyperbolic polynomial callable function
    barrier_grad_f::Any     # grad(-log(p(x))) callable function
    barrier_hess_f::Any     # hess(-log(p(x))) callable function

    point::Vector{T}                        # current point iterate
    dual_point::Vector{T}                   # dual of current point iterate, evaluated by parent class
    grad::Vector{T}                         # value of gradient of barrier function at cone.point
    dder3::Vector{T}                        # third order derivative, evaluated by parent class
    vec1::Vector{T}                         # holds temporary values in PDIPM
    vec2::Vector{T}                         # holds temporary values in PDIPM
    feas_updated::Bool                      # update state for feasibility oracle
    grad_updated::Bool                      # update state for gradient of barrier function oracle
    hess_updated::Bool                      # update state for hessian of barrier function oracle
    inv_hess_updated::Bool                  # update state for iverse of the hessian of barrier function oracle, evaluated by parent class
    hess_fact_updated::Bool                 # update state for hessian factorizatoin oracle, evaluated by parent class
    is_feas::Bool                           # feasibility at current point iterate
    hess::Symmetric{T, Matrix{T}}           # hessian matrix at current point iterate
    inv_hess::Symmetric{T, Matrix{T}}       # inverse of hessian matrix at current point iterate, evaluated by parent class
    hess_fact_mat::Symmetric{T, Matrix{T}}
    hess_fact::Factorization{T}

    """
        Hyperbolicity{T}(dim, p, barrier_grad_f, barrier_hess_f, init[, d]) where {T <: Real}
    
    If `d` is unspecified, feasibility oracle will use symbolic approach, generally slower but more accurate than the numerical approach.

    # Arguments
    - `dim`: dimension of the problem or number of variables
    - `p`: hyperbolic polynomial callable function
    - `barrier_grad_f`: gradient of the barrier function for hyperbolic polynomial, equivalent to ∇F(x), callable function
    - `barrier_hess_f`: hessian of the barrier function for hyperbolic polynomial, equivalent to ∇²F(x), callable function
    - `init`: initial point, usually vector e that p(x) is hyperbolic to
    - `d`: degree of hyperbolic polynomial, will trigger numerical approach for feasibility oracle
    """
    function Hyperbolicity{T}(dim::Int, p::Any, barrier_grad_f::Any, barrier_hess_f::Any, init::AbstractVector; d::Int=-1) where {T <: Real}
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

use_dual_barrier(::Hyperbolicity) = false

reset_data(cone::Hyperbolicity) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.hess_fact_updated = false)

use_sqrt_hess_oracles(::Int, cone::Hyperbolicity) = false

get_nu(cone::Hyperbolicity) = cone.nu

function set_initial_point!(arr::AbstractVector, cone::Hyperbolicity)
    arr .= cone.init
    return arr;
end


"""Computes the coefficients of function f with degree d using roots of unity and PolynomialRoots."""
function polyroot(d::Int, f::Any)
    # Roots of unity
    lambda_set = []
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

function update_feas(cone::Hyperbolicity{T}) where T
    @assert !cone.feas_updated

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

"""Computes gradient of barrier function at x = cone.point"""
function update_grad(cone::Hyperbolicity)
    # outputs gradient of barrier function -log(p(x)) where x = cone.point
    @assert cone.is_feas
    cone.grad = cone.barrier_grad_f(cone.point)
    cone.grad_updated = true
    return cone.grad
end

"""Computes hessian of barrier function at x = cone.point"""
function update_hess(cone::Hyperbolicity{T}) where T
    cone.grad_updated || update_grad(cone)
    # output hessian of barrier function -log(p(x)) where x = cone.point
    cone.hess = Symmetric(cone.barrier_hess_f(cone.point))
    cone.hess_updated = true
    return cone.hess
end

"""Not using third order derivative oracle"""
use_dder3(cone::Hyperbolicity) = false