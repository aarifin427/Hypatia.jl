#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

given data (xᵢ, yᵢ), find a polynomial p to solve
    min ∑ᵢℓ(p(xᵢ), yᵢ)
    ρⱼ × dᵏp/dtⱼᵏ ≥ 0 ∀ t ∈ D
where
    - dᵏp/dtⱼᵏ is the kᵗʰ derivative of p in direction j,
    - ρⱼ determines the desired sign of the derivative,
    - D is a domain such as a box or an ellipsoid,
    - ℓ is a convex loss function.
see e.g. Chapter 8 of thesis by G. Hall (2018)
=#

import DelimitedFiles
import Distributions

struct ShapeConRegrJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    X::Matrix{T}
    y::Vector{T}
    deg::Int
    # use_wsosinterppossemideftri::Bool # use WSOS matrix cone formulation if applicable, else use WSOS or SDP formulation
    # use_wsosinterpnonnegative::Bool # use WSOS cone formulation, else SDP formulation
    formulation::Symbol
    use_L1_obj::Bool # in objective function use L1 norm, else L2 norm
    use_monotonicity::Bool # if true add monotonicity constraints, else don't
    use_convexity::Bool # if true add convexity constraints, else don't
    is_fit_exact::Bool
end
function ShapeConRegrJuMP{Float64}(
    data_name::Symbol,
    args...)
    Xy = DelimitedFiles.readdlm(joinpath(@__DIR__, "data", "$data_name.txt"))
    (X, y) = (Xy[:, 1:(end - 1)], Xy[:, end])
    # TODO assert X data is on the domain [-1, 1]^n
    return ShapeConRegrJuMP{Float64}(X, y, args...)
end
function ShapeConRegrJuMP{Float64}(
    n::Int,
    num_points::Int,
    func::Symbol,
    signal_ratio::Real,
    args...)
    X = rand(Distributions.Uniform(-1, 1), num_points, n)
    f = shapeconregr_data[func]
    y = Float64[f(X[p, :]) for p in 1:num_points]
    if !iszero(signal_ratio)
        noise = randn(num_points)
        noise .*= norm(y) / sqrt(signal_ratio) / norm(noise)
        y .+= noise
    end
    return ShapeConRegrJuMP{Float64}(X, y, args...)
end

function build(inst::ShapeConRegrJuMP{T}) where {T <: Float64} # TODO generic reals
    (X, y, deg) = (inst.X, inst.y, inst.deg)
    n = size(X, 2)
    num_points = size(X, 1)
    mono_dom = ModelUtilities.Box{T}(-ones(size(X, 2)), ones(size(X, 2)))
    conv_dom = mono_dom
    mono_profile = ones(Int, size(X, 2))
    conv_profile = 1

    # setup interpolation (not actually using FreeDomain, just need points here)
    halfdeg = div(deg + 1, 2)
    free_dom = ModelUtilities.FreeDomain{Float64}(n)
    (U, points, Ps, V) = ModelUtilities.interpolate(free_dom, halfdeg, calc_V = true) # return F parts for qr(V') instead?? # TODO don't need points
    # TODO maybe incorporate this interp-basis transform into MU, and do something smarter for uni/bi-variate
    F = qr!(Array(V'), Val(true)) # TODO reuse QR parts
    V_X = ModelUtilities.make_chebyshev_vandermonde(X, 2halfdeg)
    X_points_polys = F \ V_X'

    model = JuMP.Model()
    JuMP.@variable(model, regressor[1:U])
    JuMP.@variable(model, z)
    JuMP.@objective(model, Min, z)

    # objective epigraph
    norm_vec = y - X_points_polys' * regressor
    if inst.use_L1_obj || (num_points <= U)
        obj_cone = (inst.use_L1_obj ? MOI.NormOneCone : MOI.SecondOrderCone)(1 + num_points)
        JuMP.@constraint(model, vcat(z, norm_vec) in obj_cone)
    else
        # using L2 norm objective and number of samples exceeds variables, so use qr trick to reduce dimension
        coef_mat = zeros(num_points, U + 1)
        for (i, expr_i) in enumerate(norm_vec)
            for (c, v) in JuMP.linear_terms(expr_i)
                coef_mat[i, JuMP.index(v).value] = c
            end
            coef_mat[i, end] = JuMP.constant(expr_i)
        end
        coef_R = qr(coef_mat).R
        JuMP.@constraint(model, vcat(z, coef_R * vcat(regressor, 1)) in MOI.SecondOrderCone(2 + U))
    end

    # monotonicity
    if inst.use_monotonicity
        gradient_halfdeg = div(deg, 2)
        (mono_U, mono_points, mono_Ps) = ModelUtilities.interpolate(mono_dom, gradient_halfdeg) # return F parts for qr(V') instead??
        univ_chebs_derivs = [ModelUtilities.calc_univariate_chebyshev(mono_points[:, i], 2halfdeg, calc_gradient = true) for i in 1:n]

        for j in 1:n
            iszero(mono_profile[j]) && continue

            univ_chebs_g = [univ_chebs_derivs[i][(i == j) ? 2 : 1] for i in 1:n]
            V_g = ModelUtilities.make_product_vandermonde(univ_chebs_g, ModelUtilities.n_deg_exponents(n, 2halfdeg))
            scal = inv(maximum(abs, V_g) / 10)
            scal < 1e-7 && @warn("model is numerically challenging to set up", maxlog = 1)
            lmul!(scal, V_g)
            g_points_polys = F \ V_g'

            gradient_interp = mono_profile[j] * g_points_polys' * regressor

            if inst.use_wsosinterpnonnegative
                JuMP.@constraint(model, gradient_interp in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(mono_U, mono_Ps))
            else
                psd_vars = []
                for (r, Pr) in enumerate(mono_Ps)
                    Lr = size(Pr, 2)
                    psd_r = JuMP.@variable(model, [1:Lr, 1:Lr], Symmetric)
                    push!(psd_vars, psd_r)
                    JuMP.@SDconstraint(model, psd_r >= 0)
                end
                coeffs_lhs = JuMP.@expression(model, [u in 1:mono_U], sum(sum(Pr[u, k] * Pr[u, l] * psd_r[k, l] * (k == l ? 1 : 2) for k in 1:size(Pr, 2) for l in 1:k) for (Pr, psd_r) in zip(mono_Ps, psd_vars)))
                JuMP.@constraint(model, coeffs_lhs .== gradient_interp)
            end
        end
    end

    # convexity
    if inst.use_convexity && !iszero(conv_profile)
        hessian_halfdeg = div(deg - 1, 2)
        (conv_U, conv_points, conv_Ps) = ModelUtilities.interpolate(conv_dom, hessian_halfdeg) # return F parts for qr(V') instead??
        univ_chebs_derivs = [ModelUtilities.calc_univariate_chebyshev(conv_points[:, i], 2halfdeg, calc_gradient = true, calc_hessian = true) for i in 1:n]

        deriv_num(i, j, k) = (k != i && k != j && return 1; k == i && k == j && return 3; return 2)

        V_Hs = Matrix{Float64}[]
        for i in 1:n, j in 1:i
            univ_chebs_H = [univ_chebs_derivs[k][deriv_num(i, j, k)] for k in 1:n]
            V_H = ModelUtilities.make_product_vandermonde(univ_chebs_H, ModelUtilities.n_deg_exponents(n, 2halfdeg))
            push!(V_Hs, V_H)
        end
        scal = inv(maximum(maximum(abs, V_H) for V_H in V_Hs))
        scal < 1e-7 && @warn("model is numerically challenging to set up", maxlog = 1)
        lmul!.(scal, V_Hs)

        hessian_interp = conv_profile * vcat([(F \ V_H')' * regressor for V_H in V_Hs]...)

        if inst.formulation == :nat_wsos_mat
            if n == 1
                conv_cone = Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(conv_U, conv_Ps)
            else
                ModelUtilities.vec_to_svec!(hessian_interp, rt2 = sqrt(2), incr = conv_U)
                conv_cone = Hypatia.WSOSInterpPosSemidefTriCone{Float64}(n, conv_U, conv_Ps)
            end
            JuMP.@constraint(model, hessian_interp in conv_cone)
        elseif inst.formulation == :nat_wsos
            svec_dim = div(n * (n + 1), 2)
            ypts = zeros(svec_dim, n)
            idx = 1
            idx_jj = 1
            for j in 1:n
                idx_ii = 1
                for i in 1:(j - 1)
                    hessian_interp[Cones.block_idxs(conv_U, idx)] *= 2
                    hessian_interp[Cones.block_idxs(conv_U, idx)] += hessian_interp[Cones.block_idxs(conv_U, idx_ii)] + hessian_interp[Cones.block_idxs(conv_U, idx_jj)]
                    ypts[idx, i] = ypts[idx, j] = 1
                    idx_ii += (i + 1)
                    idx += 1
                end
                ypts[idx, j] = 1
                idx += 1
                idx_jj += (j + 1)
            end
            new_Ps = Matrix{Float64}[]
            for P in conv_Ps
                push!(new_Ps, kron(ypts, P))
            end
            JuMP.@constraint(model, hessian_interp in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(conv_U * svec_dim, new_Ps))
        elseif inst.formulation == :ext
            psd_vars = []
            for (r, Pr) in enumerate(conv_Ps)
                Lr = size(Pr, 2)
                psd_r = JuMP.@variable(model, [1:(Lr * n), 1:(Lr * n)], Symmetric)
                push!(psd_vars, psd_r)
                JuMP.@SDconstraint(model, psd_r >= 0)
            end
            # for readability
            Ls = [size(Pr, 2) for Pr in conv_Ps]
            offset = 0
            for x1 in 1:n, x2 in 1:x1
                offset += 1
                # note that psd_vars[r][(x1 - 1) * Ls[r] + k, (x2 - 1) * Ls[r] + l] is not necessarily symmetric
                coeffs_lhs = JuMP.@expression(model, [u in 1:conv_U], sum(sum(conv_Ps[r][u, k] * conv_Ps[r][u, l] * psd_vars[r][(x1 - 1) * Ls[r] + k, (x2 - 1) * Ls[r] + l] for k in 1:Ls[r] for l in 1:Ls[r]) for r in eachindex(Ls)))
                JuMP.@constraint(model, coeffs_lhs .== hessian_interp[conv_U .* (offset - 1) .+ (1:conv_U)])
            end
        else
            error()
        end
    end

    return model
end

function test_extra(inst::ShapeConRegrJuMP{T}, model::JuMP.Model) where T
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    if JuMP.termination_status(model) == MOI.OPTIMAL && inst.is_fit_exact
        # check objective value is correct
        tol = eps(T)^0.25
        @test JuMP.objective_value(model) ≈ 0 atol = tol rtol = tol
    end
end

shapeconregr_data = Dict(
    :func1 => (x -> sum(x .^ 2)),
    :func2 => (x -> sum(x .^ 3)),
    :func3 => (x -> sum(x .^ 4)),
    :func4 => (x -> exp(norm(x)^2 / length(x)) - 1),
    :func5 => (x -> -inv(1 + exp(-10 * norm(x)))),
    :func6 => (x -> sum((x .+ 1) .^ 4)),
    :func7 => (x -> sum((x / 2 .+ 1) .^ 3)),
    :func8 => (x -> sum((x .+ 1) .^ 5 .- 2)),
    :func9 => (x -> (5x[1] + x[2] / 2 + x[3])^2 + sqrt(x[4]^2 + x[5]^2)),
    :func10 => (x -> sum(exp.(x))),
    )

# instances[ShapeConRegrJuMP]["minimal"] = [
#     ((:naics5811, 3, true, false, false, true, true, false),),
#     ((:naics5811, 3, true, false, false, true, false, false),),
#     ((:naics5811, 3, true, false, false, false, true, false),),
#     ((1, 5, :func1, 2, 5, true, false, false, true, true, false),),
#     ((2, 5, :func2, 2, 4, true, false, false, true, false, false),),
#     ((3, 5, :func3, 2, 3, true, false, false, false, true, false),),
#     ((1, 5, :func4, 2, 4, true, false, false, false, false, true),),
#     ((1, 5, :func5, 2, 4, true, false, true, true, true, false),),
#     ((1, 5, :func6, 2, 4, false, false, false, true, true, false),),
#     ((1, 5, :func7, 2, 4, false, false, true, true, true, false), ClassicConeOptimizer),
#     ((1, 5, :func8, 2, 4, false, false, true, true, true, false),),
#     ((1, 5, :func1, 2, 4, false, false, true, false, false, true), ClassicConeOptimizer),
#     ((1, 5, :func5, 2, 4, false, true, true, true, true, false),),
#     ((1, 5, :func6, 2, 4, false, true, false, true, true, false),),
#     ]
# instances[ShapeConRegrJuMP]["fast"] = [
#     ((:naics5811, 4, true, false, false, true, true, false),),
#     ((:naics5811, 4, true, false, true, true, true, false),),
#     ((:naics5811, 3, false, false, false, true, true, false),),
#     ((:naics5811, 3, false, false, true, true, true, false), ClassicConeOptimizer),
#     ((:naics5811, 3, false, false, true, true, true, false),),
#     ((:naics5811, 3, false, false, false, true, true, false),),
#     ((:naics5811, 3, false, false, true, true, false, false), ClassicConeOptimizer),
#     ((1, 100, :func1, 5, 10, true, false, false, true, true, false),),
#     ((1, 100, :func1, 5, 20, false, false, false, false, true, false),),
#     ((1, 100, :func1, 5, 50, true, false, false, false, true, false),),
#     ((1, 100, :func1, 5, 80, true, false, false, false, true, false),),
#     ((1, 100, :func1, 5, 100, true, false, false, false, true, false),),
#     ((1, 200, :func4, 5, 100, true, false, false, false, true, false),),
#     ((2, 50, :func1, 5, 5, true, false, false, true, true, false),),
#     ((2, 50, :func1, 5, 3, true, false, false, true, false, false),),
#     ((2, 50, :func1, 5, 3, true, false, false, false, true, false),),
#     ((2, 200, :func1, 0, 3, true, false, false, false, false, true),),
#     ((2, 50, :func2, 5, 3, true, false, true, true, true, false),),
#     ((2, 50, :func3, 10, 3, false, false, true, false, true, false),),
#     ((2, 50, :func3, 10, 3, true, false, true, false, true, false),),
#     ((2, 50, :func3, 5, 3, false, false, true, true, true, false), ClassicConeOptimizer),
#     ((2, 50, :func4, 5, 3, false, false, true, true, true, false),),
#     ((2, 50, :func4, 5, 3, false, false, true, true, true, false), ClassicConeOptimizer),
#     ((2, 50, :func5, 5, 4, true, false, false, true, true, false),),
#     ((2, 50, :func6, 5, 4, true, false, true, true, true, false),),
#     ((2, 50, :func7, 5, 4, false, false, false, true, true, false),),
#     ((2, 50, :func8, 5, 4, false, false, true, true, true, true, false),),
#     ((4, 150, :func6, 0, 4, true, false, false, true, true, true),),
#     ((4, 150, :func7, 0, 4, true, false, false, true, true, true),),
#     ((4, 150, :func7, 0, 4, true, false, true, true, true, true),),
#     ((4, 150, :func7, 0, 4, false, false, false, true, true, true),),
#     ((3, 150, :func8, 0, 6, true, false, false, true, true, true),),
#     ((2, 50, :func3, 10, 3, false, true, true, false, true, false),),
#     ]
# instances[ShapeConRegrJuMP]["slow"] = [
#     ((:naics5811, 3, false, false, true, false, true, false), ClassicConeOptimizer),
#     ((:naics5811, 7, true, false, false, true, true, false),),
#     ((:naics5811, 5, false, false, true, true, true, false), ClassicConeOptimizer),
#     ((2, 200, :func1, 5, 20, true, false, false, true, true, false),),
#     ((2, 5000, :func1, 5, 40, true, false, false, false, true, false),),
#     ((4, 150, :func6, 0, 4, false, false, false, true, true, true),),
#     ((4, 150, :func6, 0, 4, false, false, true, true, true, true), ClassicConeOptimizer),
#     ((3, 500, :func8, 0, 6, false, false, false, true, true, true),),
#     ((3, 500, :func8, 0, 6, false, false, false, true, false, true),),
#     ((3, 500, :func8, 0, 6, false, false, false, false, true, true),),
#     ((3, 500, :func8, 0, 6, false, false, true, true, true, true), ClassicConeOptimizer),
#     ((3, 500, :func8, 0, 6, false, false, true, true, false, true), ClassicConeOptimizer),
#     ((3, 500, :func8, 0, 6, false, false, true, false, true, true), ClassicConeOptimizer),
#     ((5, 500, :func9, 9, 4, false, false, true, true, true, false),),
#     ((5, 500, :func9, 9, 4, true, false, true, true, true, false),),
#     ((5, 500, :func9, 9, 4, false, false, true, true, true, false), ClassicConeOptimizer),
#     ((5, 500, :func10, 4, 4, false, false, true, true, true, false),),
#     ((5, 500, :func10, 4, 4, false, true, false, true, true, false),),
#     ((5, 500, :func10, 4, 4, false, false, true, false, true, false), ClassicConeOptimizer),
#     ((5, 500, :func10, 4, 4, false, false, true, false, false, false), ClassicConeOptimizer),
#     ((5, 500, :func10, 4, 4, false, false, true, true, false, false), ClassicConeOptimizer),
#     ((5, 500, :func10, 4, 4, false, false, true, true, true, false), ClassicConeOptimizer),
#     ]
#
# # benchmark 1 instances
# bench1_n_d = [
#     (1, 3), # compile run
#     (1, 10),
#     (1, 20),
#     (1, 30),
#     (1, 40),
#     (1, 50),
#     (1, 60),
#     (2, 2), # compile run
#     (2, 5),
#     (2, 10),
#     (2, 15),
#     (2, 20),
#     (3, 1), # compile run
#     (3, 2),
#     (3, 4),
#     (3, 6),
#     (3, 8),
#     (4, 2),
#     (4, 3),
#     (4, 4),
#     (4, 5),
#     (6, 2),
#     (6, 3),
#     (8, 2),
#     (10, 2),
#     (12, 2),
#     (14, 2),
#     ]
# instances[ShapeConRegrJuMP]["bench1"] = (
#     ((n, ceil(Int, 1.1 * binomial(n + 2d, n)), :func4, 100.0, 2d, use_wsosinterpnonnegative, false, false, true, false),)
#     for (n, d) in bench1_n_d
#     for use_wsosinterpnonnegative in (false, true)
#     )

return ShapeConRegrJuMP
