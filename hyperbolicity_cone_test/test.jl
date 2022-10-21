using Test
##################GLOBAL######################
tol = 1e-5
##############################################

@testset verbose=true "system level" begin
    @testset verbose=true "test against established cones" begin
        @testset verbose=true "nonneg" begin
            for i=1:4
                str = "nonneg/test"*string(i)*".jl"
                include(str)
                @test norm(ans_control - ans_test) < tol
            end
        end
        @testset verbose=true "sdp" begin
            for i=1:4
                str = "sdp/test"*string(i)*".jl"
                include(str)
                @test norm(ans_control - ans_test) < tol
            end
        end
    end
    @testset verbose=true "test general functionality" begin
        # tests if it reaches "Optimal" or "NearOptimal" flag
        for i=1:25
            str = "general_test/test"*string(i)*".jl"
            include(str)
            @test string(Solvers.get_status(solver)) in ["Optimal", "NearOptimal"]
        end
    end
end