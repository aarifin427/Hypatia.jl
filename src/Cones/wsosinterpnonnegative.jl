#=
interpolation-based weighted-sum-of-squares (multivariate) polynomial cone parametrized by interpolation matrices Ps

definition and dual barrier from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792

TODO
- perform loop for calculating g and H in parallel
- scale the interior direction
=#

mutable struct WSOSInterpNonnegative{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual_barrier::Bool
    use_heuristic_neighborhood::Bool
    dim::Int
    Ps::Vector{Matrix{R}}

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T}
    correction::Vector{T}
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
    hess_fact_cache

    tmpLL::Vector{Matrix{R}}
    tmpUL::Vector{Matrix{R}}
    tmpLU::Vector{Matrix{R}}
    tmpLU2::Vector{Matrix{R}}
    tmpLU3::Vector{Matrix{R}}
    tmpUU::Vector{Matrix{R}} # TODO for corrector, this can stay as a single matrix if we only use LU
    ΛF::Vector

    function WSOSInterpNonnegative{T, R}(
        U::Int,
        Ps::Vector{Matrix{R}};
        use_dual::Bool = false,
        use_heuristic_neighborhood::Bool = default_use_heuristic_neighborhood(),
        hess_fact_cache = hessian_cache(T),
        ) where {R <: RealOrComplex{T}} where {T <: Real}
        for Pk in Ps
            @assert size(Pk, 1) == U
        end
        cone = new{T, R}()
        cone.use_dual_barrier = !use_dual # using dual barrier
        cone.use_heuristic_neighborhood = use_heuristic_neighborhood
        cone.dim = U
        cone.Ps = Ps
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

function setup_extra_data(cone::WSOSInterpNonnegative{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    dim = cone.dim
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    Ls = [size(Pk, 2) for Pk in cone.Ps]
    cone.tmpLL = [zeros(R, L, L) for L in Ls]
    cone.tmpUL = [zeros(R, dim, L) for L in Ls]
    cone.tmpLU = [zeros(R, L, dim) for L in Ls]
    cone.tmpLU2 = [zeros(R, L, dim) for L in Ls]
    cone.tmpLU3 = [zeros(R, L, dim) for L in Ls]
    cone.tmpUU = [zeros(R, dim, dim) for L in Ls]
    cone.ΛF = Vector{Any}(undef, length(Ls))
    return cone
end

get_nu(cone::WSOSInterpNonnegative) = sum(size(Pk, 2) for Pk in cone.Ps)

# TODO find "central" initial point, like for other cones
set_initial_point(arr::AbstractVector, cone::WSOSInterpNonnegative) = (arr .= 1)

# TODO order the k indices so that fastest and most recently infeasible k are first
# TODO can be done in parallel
function update_feas(cone::WSOSInterpNonnegative)
    @assert !cone.feas_updated
    D = Diagonal(cone.point)

    cone.is_feas = true
    @inbounds for k in eachindex(cone.Ps)
        # Λ = Pk' * Diagonal(point) * Pk
        # TODO mul!(A, B', Diagonal(x)) calls extremely inefficient method but doesn't need ULk
        Pk = cone.Ps[k]
        ULk = cone.tmpUL[k]
        LLk = cone.tmpLL[k]
        mul!(ULk, D, Pk)
        mul!(LLk, Pk', ULk)

        ΛFk = cholesky!(Hermitian(LLk, :L), check = false)
        if !isposdef(ΛFk)
            cone.is_feas = false
            break
        end
        cone.ΛF[k] = ΛFk
    end

    cone.feas_updated = true
    return cone.is_feas
end

is_dual_feas(cone::WSOSInterpNonnegative) = true

# TODO decide whether to compute the LUk' * LUk in grad or in hess (only diag needed for grad)
# TODO can be done in parallel
# TODO may be faster (but less numerically stable) with explicit inverse here
function update_grad(cone::WSOSInterpNonnegative)
    @assert cone.is_feas

    cone.grad .= 0
    @inbounds for k in eachindex(cone.Ps)
        LUk = cone.tmpLU[k]
        ldiv!(LUk, cone.ΛF[k].L, cone.Ps[k]')
        @inbounds for j in 1:cone.dim
            cone.grad[j] -= sum(abs2, view(LUk, :, j))
        end
    end

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::WSOSInterpNonnegative)
    @assert cone.grad_updated

    cone.hess .= 0
    @inbounds for k in eachindex(cone.Ps)
        LUk = cone.tmpLU[k]
        UUk = mul!(cone.tmpUU[k], LUk', LUk) # TODO use syrk
        @inbounds for j in 1:cone.dim, i in 1:j
            cone.hess.data[i, j] += abs2(UUk[i, j])
        end
    end

    cone.hess_updated = true
    return cone.hess
end

function correction(cone::WSOSInterpNonnegative, primal_dir::AbstractVector)
    corr = cone.correction

    corr .= 0
    @inbounds for k in eachindex(cone.Ps)
        mul!(cone.tmpLU2[k], cone.tmpLU[k],  Diagonal(primal_dir))
        LpdU = mul!(cone.tmpLU3[k], cone.tmpLU2[k], cone.tmpUU[k])
        @inbounds @views for j in 1:cone.dim
            corr[j] += sum(abs2, LpdU[:, j])
        end
    end

    return corr
end
