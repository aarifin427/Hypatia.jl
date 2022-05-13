using Test
using ForwardDiff
using Random
include("diff.jl")
include("function_examples.jl")

@testset "Partial diff" begin
    @test partial_diff(p2, 1) == [
        2 2 -8 4
        1 1 0 1
        2 0 1 0
        0 2 1 0
        0 0 1 2
    ]
end

@testset "Compute Grad representation" begin
    dx1 = [
        2 2 -8 4
        1 1 0 1
        2 0 1 0
        0 2 1 0
        0 0 1 2
    ]
    dx2 = [
        2 2 -8 4
        2 0 1 0
        1 1 0 1
        0 2 1 0
        0 0 1 2
    ]
    dx3 = [
        2 2 4 -8
        2 0 0 1
        0 2 0 1
        1 1 3 0
        0 0 0 1
    ]
    dx4 = [
        -8 4 4
        1 2 0
        1 0 2
        1 0 0
        0 1 1
    ]
    x1, x2, x3, x4 = compute_grad_rep(p2)
    @test [x1, x2, x3, x4] == [dx1, dx2, dx3, dx4]
end

@testset "Matrix Representation to Value" begin
    @testset "Polynomial value" begin
        @testset "Function 1" begin
            for k = 1:100
                local point = [rand(1:10), rand(1:10), rand(1:10), rand(1:10)]
                @test f1(point) == func_eval(p1, point)
            end
        end
        @testset "Function 2" begin
            for k = 1:100
                local point = [rand(1:10), rand(1:10), rand(1:10), rand(1:10)]
                @test f2(point) == func_eval(p2, point)
            end
        end
        
    end
    @testset "Gradient" begin
        # function 1
        @testset "Function 1" begin
            local f(x) = f1(x)
            local p = p1
            local g = x -> ForwardDiff.gradient(f, x);

            for k = 1:100
                local point = [rand(1:10), rand(1:10), rand(1:10), rand(1:10)]
                res = g(point) .== mat_eval(compute_grad_rep(p), point)
                @test sum(res) == length(res)
            end
        end
        # function 2
        @testset "Function 2" begin
            local f(x) = f2(x)
            local p = p2
            local g = x -> ForwardDiff.gradient(f, x);

            for k = 1:100
                local point = [rand(1:10), rand(1:10), rand(1:10), rand(1:10)]
                res = g(point) .== mat_eval(compute_grad_rep(p), point)
                @test sum(res) == length(res)
            end
        end
    end
    @testset "Hessian" begin
        # function 1
        @testset "Function 1" begin
            local f(x) = f1(x)
            local p = p1
            local g = x -> ForwardDiff.hessian(f, x)

            for k = 1:100
                local point = [rand(1:10), rand(1:10), rand(1:10), rand(1:10)]
                hess = compute_hess_rep(compute_grad_rep(p))
                res = mat_eval(hess, point)
                @test g(point) == res
            end
        end
        # function 2
        @testset "Function 2" begin
            local f(x) = f2(x)
            local p = p2
            local g = x -> ForwardDiff.hessian(f, x);

            for k = 1:100
                local point = [rand(1:10), rand(1:10), rand(1:10), rand(1:10)]
                hess = compute_hess_rep(compute_grad_rep(p))
                res = mat_eval(hess, point)
                @test g(point) == res
            end
        end
    end
end

@testset "Substitute values into polynomial function representations" begin
    @testset "Function 1: Vamos polynomial" begin
        for k = 1:100
            local point = [rand(1:10), rand(1:10), rand(1:10), rand(1:10)]
            @test f1(point) == func_eval(p1, point)
        end
    end
    @testset "Function 2: Polynomial W from Blekherman" begin
        for k = 1:100
            local point = [rand(1:10), rand(1:10), rand(1:10), rand(1:10)]
            @test f2(point) == func_eval(p2, point)
        end
    end
end