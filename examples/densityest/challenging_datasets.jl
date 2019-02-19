#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

using DataFrames
using CSV
include(joinpath(@__DIR__(), "jump.jl"))

function scale_X!(X)
    X .-= 0.5 * (minimum(X, dims=1) + maximum(X, dims=1))
    X ./= (0.5 * (maximum(X, dims=1) - minimum(X, dims=1)))
    return nothing
end

# iris dataset
function iris_data()
    df = CSV.read(joinpath(@__DIR__, "data", "iris.csv"))
    dropmissing!(df, disallowmissing = true)
    # only use setosa species
    # xcols = [:sepal_length, :sepal_width, :petal_length, :petal_width]
    xcols = [:petal_length, :sepal_length]
    dfsub = df[df.species .== "setosa", xcols]
    X = convert(Matrix{Float64}, dfsub)
    scale_X!(X)
    n = length(xcols)
    return (X, n)
end

# lung cancer dataset from https://github.com/therneau/survival (cancer.rda)
# description at https://github.com/therneau/survival/blob/master/man/lung.Rd
function cancer_data()
    df = CSV.read(joinpath(@__DIR__, "data", "cancer.csv"), missingstring = "NA")
    dropmissing!(df, disallowmissing = true)
    # only use males with status 2
    dfsub = df[df.status .== 2, :]
    # xcols = [:time, :age, :ph_ecog, :ph_karno, :pat_karno, :meal_cal, :wt_loss]
    xcols = [:time, :meal_cal]
    dfsub = dfsub[dfsub.sex .== 1, xcols]
    X = convert(Matrix{Float64}, dfsub)
    scale_X!(X)
    n = length(xcols)
    return (X, n)
end

function run_hard_densityest()
    degrees = 4:2:6

    datasets = [
        iris_data,
        cancer_data,
        ]

    for d in degrees, s in datasets
        model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer,
            use_dense = true,
            verbose = true,
            system_solver = SO.QRCholCombinedHSDSystemSolver,
            linear_model = MO.PreprocessedLinearModel,
            max_iters = 250,
            time_limit = 3.6e3,
            tol_rel_opt = 1e-5,
            tol_abs_opt = 1e-6,
            tol_feas = 1e-6,
            ))

        println()
        @show d
        @show s
        println()

        (X, n) = s()
        dom = MU.Box(-ones(n), ones(n))
        (_, f) = build_JuMP_densityest(model, X, d, dom, use_monomials = true)

        (val, runtime, bytes, gctime, memallocs) = @timed JuMP.optimize!(model)

        println()
        @show runtime
        @show bytes
        @show gctime
        @show memallocs
        println("\n\n")
    end
end

# run_hard_densityest()
# pl = densityest_plot(JuMP.value.(f), X, use_contour = true, random_data = false)