using Test
using ForwardDiff
using Random
include("diff.jl")

@testset "y = x^a + y^b + z^c structure" begin
    @testset "Polynomial of 4 terms" begin
        pwc = 50 # power cap
        n = 100  # number of tests
        for k = 1:n
            local a, b, c = rand(1:pwc), rand(1:pwc), rand(1:pwc)
            local p = [
                rand(1:pwc) rand(1:pwc) rand(1:pwc)
                rand(1:pwc) 0           0          
                0           rand(1:pwc) 0          
                0           0           rand(1:pwc)
            ]
            local f(x) = p[1,1]*x[1]^p[2,1] + p[1,2]*x[2]^p[3,2] + p[1,3]*x[3]^p[4,3]
            local g = x -> ForwardDiff.gradient(f, x);
            local h = x -> ForwardDiff.hessian(f, x);
            local point = [rand(1:10), rand(1:10), rand(1:10)]
            
            @test g(point) == gradient(point, p)
            # @test h(point) == hessian(point, p)
        end;
    end

    @testset "Polynomial of 5 terms" begin
        pwc = 50 # power cap
        n = 100  # number of tests
        for k = 1:n
            local a, b, c = rand(1:pwc), rand(1:pwc), rand(1:pwc)
            local p = [
                rand(1:pwc) rand(1:pwc) rand(1:pwc) rand(1:pwc)
                rand(1:pwc) 0           0           0
                0           rand(1:pwc) 0           0
                0           0           rand(1:pwc) 0
                0           0           0           rand(1:pwc)
            ]
            local f(x) = p[1,1]*x[1]^p[2,1] + p[1,2]*x[2]^p[3,2] + p[1,3]*x[3]^p[4,3] + p[1,4]*x[4]^p[5,4]
            local g = x -> ForwardDiff.gradient(f, x);
            local h = x -> ForwardDiff.hessian(f, x);
            local point = [rand(1:10), rand(1:10), rand(1:10), rand(1:10)]
            
            @test g(point) == gradient(point, p)
            # @test h(point) == hessian(point, p)
        end;
    end
end