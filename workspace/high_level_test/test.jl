using Test
# using ForwardDiff
# using Hypatia
# using Hypatia.Cones
# using Hypatia.Models
# import Hypatia.Solvers
# using LinearAlgebra
# using JuMP
# using SCS
##################GLOBAL######################
tol = 1e-5
##############################################

@testset verbose=true "system level" begin
    @testset verbose=true "test against established cones" begin
        @testset verbose=true "nonneg" begin
            @testset "4D, 1 addend, init point feasible" begin
                include("nonneg/test1.jl")
                @test norm(ans_control - ans_test) < tol
            end
            @testset "4D, 1 addend, init point infeasible" begin
                include("nonneg/test2.jl")
                @test norm(ans_control - ans_test) < tol
            end
            @testset "3D, 1 addend" begin
                include("nonneg/test3.jl")
                @test norm(ans_control - ans_test) < tol
                include("nonneg/test4.jl")
                @test norm(ans_control - ans_test) < tol
            end
            @testset "4D, 1 addend, more constraints" begin
                include("nonneg/test5.jl")
                @test norm(ans_control - ans_test) < tol
            end
        end
        @testset verbose=true "sdp" begin
            @testset "SDP 3D 3x3 mat tol=1e-3" begin
                include("sdp/test1.jl")
                @test norm(ans_control - ans_test) < 1e-3
            end
            @testset "vamos 1 constraint" begin
                include("sdp/test2.jl")
                @test norm(ans_control - ans_test) < tol
            end
            @testset "vamos 3 constraints" begin
                include("sdp/test3.jl")
                @test norm(ans_control - ans_test) < tol
            end
        end
    end
    @testset verbose=true "test general functionality" begin
        # tests if it reaches "Optimal" flag
        @testset "3D, 2 addends" begin
            include("general_test/test1.jl")
            @test string(Solvers.get_status(solver)) == "Optimal"
        end
        @testset "3D, 3 addends" begin
            include("general_test/test2.jl")
            @test string(Solvers.get_status(solver)) == "Optimal"
        end
        @testset "3D, 4 addends" begin
            ##############################################################################################
            # test
            ##############################################################################################
            include("general_test/test3.jl")
            @test string(Solvers.get_status(solver)) == "Optimal"
            
            ##############################################################################################
            # test
            ##############################################################################################
            include("general_test/test4.jl")
            @test string(Solvers.get_status(solver)) == "Optimal"
        end
        @testset "4D, 4 addends, 1 constraint" begin
            include("general_test/test5.jl")
            @test string(Solvers.get_status(solver)) == "Optimal"
        end
        @testset "3D, 1st derivative of hyperbolic polynomials" begin
            include("general_test/test6.jl")
            @test string(Solvers.get_status(solver)) == "Optimal"
        end
        @testset "4D, 4 addends, 2 constraints" begin
            include("general_test/test7.jl")
            @test string(Solvers.get_status(solver)) == "Optimal"
        end
        @testset "8D, over 3D Hypercube" begin
            include("general_test/test8.jl")
            @test string(Solvers.get_status(solver)) == "Optimal"
        end
    end
end