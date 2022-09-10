using Test
using SymPy
using Random
using Distributions
using LinearAlgebra

@testset verbose = true "check if hyperbolic" begin
    T = Float64

    p(x) = x[1]*x[2]*(x[1]*x[2] + x[1]*x[3] + x[1]*x[4] + x[2]*x[3] + x[3]*x[4])
    init_point = 1.0*[1,1,0,0]
    n = 3

    t = symbols("t")
    e = init_point
    @testset "p(e) > 0" begin
        @test p(e) > 0
    end

    @testset "t -> p(x-t*e) have real roots" begin
        for k = 1:20
            local x = rand(Uniform(-50,50), n)
            
            a = SymPy.solve(p(x - t*e))

            is_feas = true
            for i in eachindex(a)
                # if !isreal(a[i]) || abs(a[i]) < eps(T)
                if abs(imag(a[i])) > eps(T) || abs(a[i]) < eps(T)
                    # not real, infeasible (?)
                    is_feas = false
                    break
                end
            end
            @test is_feas
        end
    end
end