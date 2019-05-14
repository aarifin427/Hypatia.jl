#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

TODO description
=#

import LinearAlgebra
import Random
using Test
import MathOptInterface
const MOI = MathOptInterface
import JuMP
import DynamicPolynomials
const DP = DynamicPolynomials
import Hypatia
const HYP = Hypatia
const MU = HYP.ModelUtilities

const rt2 = sqrt(2)

function JuMP_polysoc_monomial(P, n)
    dom = MU.FreeDomain(n)
    d = div(maximum(DP.maxdegree.(P)) + 1, 2)
    (U, pts, P0, _, _) = MU.interpolate(dom, d, sample = false)
    cone = HYP.WSOSPolyInterpSOCCone(length(P), U, [P0])

    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true, max_iters = 200))
    JuMP.@constraint(model, [P[i](pts[u, :]) for i in 1:length(P) for u in 1:U] in cone)

    return model
end

function simple_feasibility()
    DP.@polyvar x
    for socpoly in [
            [2x^2 + 2, x, x],
            [x^2 + 2, x], [x^2 + 2, x, x],
            [2 * x^4 + 8 * x^2 + 4, x + 2 + (x + 1)^2, x],
            ]
        model = JuMP_polysoc_monomial(socpoly, 1)
        JuMP.optimize!(model)
        @test JuMP.termination_status(model) == MOI.OPTIMAL
        @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
    end
end

function simple_infeasibility()
    DP.@polyvar x
    for socpoly in [
        [x, x^2 + x],
        [x, x + 1],
        [x^2, x],
        [x + 2, x],
        [x - 1, x, x],
        ]
        @show socpoly
        model = JuMP_polysoc_monomial(socpoly, 1)
        JuMP.optimize!(model)
        @test JuMP.termination_status(model) == MOI.INFEASIBLE
        @test JuMP.primal_status(model) == MOI.INFEASIBLE_POINT
    end
end

@testset "everything" begin
    simple_feasibility()
    simple_infeasibility()

    Random.seed!(1)
    for deg in 1:2, n in 1:2, npolys in 1:2
        println()
        @show deg, n, npolys

        dom = MU.FreeDomain(n)
        d = div(deg + 1, 2)
        (U, pts, P0, _, w) = MU.interpolate(dom, d, sample = false, calc_w = true)
        lagrange_polys = MU.recover_lagrange_polys(pts, 2d)

        # generate vector of random polys using the Lagrange basis
        random_coefs = Random.rand(npolys, U)
        subpolys = [LinearAlgebra.dot(random_coefs[i, :], lagrange_polys) for i in 1:npolys]
        random_vec = [random_coefs[i, u] for i in 1:npolys for u in 1:U]

        model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true, max_iters = 100))
        JuMP.@variable(model, coefs[1:U])
        JuMP.@constraint(model, [coefs; random_vec...] in HYP.WSOSPolyInterpSOCCone(npolys + 1, U, [P0]))
        # JuMP.@objective(model, Min, dot(quad_weights, coefs))
        JuMP.optimize!(model)
        upper_bound = LinearAlgebra.dot(JuMP.value.(coefs), lagrange_polys)
        @test JuMP.termination_status(model) == MOI.OPTIMAL
        @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT

        for i in 1:50
            pt = randn(n)
            @test (upper_bound(pt))^2 >= sum(subpolys.^2)(pt)
        end
    end
end