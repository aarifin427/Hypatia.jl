#=
utilities for benchmark scripts
=#

using Printf
import DataFrames
import CSV
examples_dir = joinpath(@__DIR__, "../examples")
include(joinpath(examples_dir, "common_JuMP.jl"))
include(joinpath(examples_dir, "common_native.jl"))

function setup_benchmark_dataframe()
    perf = DataFrames.DataFrame(
        example = String[],
        model_type = String[],
        inst_set = String[],
        inst_num = Int[],
        inst_data = Tuple[],
        extender = String[],
        real_T = Type{<:Real}[],
        solver = String[],
        solver_options = Tuple[],
        script_status = String[],
        n = Int[],
        p = Int[],
        q = Int[],
        nu = Float64[],
        cone_types = Vector{String}[],
        num_cones = Int[],
        max_q = Int[],
        status = String[],
        solve_time = Float64[],
        iters = Int[],
        primal_obj = Float64[],
        dual_obj = Float64[],
        rel_obj_diff = Float64[],
        compl = Float64[],
        x_viol = Float64[],
        y_viol = Float64[],
        z_viol = Float64[],
        time_rescale = Float64[],
        time_initx = Float64[],
        time_inity = Float64[],
        time_unproc = Float64[],
        time_loadsys = Float64[],
        time_upsys = Float64[],
        time_upfact = Float64[],
        time_uprhs = Float64[],
        time_getdir = Float64[],
        time_search = Float64[],
        setup_time = Float64[],
        check_time = Float64[],
        total_time = Float64[],
        )
    DataFrames.allowmissing!(perf, 11:DataFrames.ncol(perf))
    return perf
end

get_extender(inst::Tuple, ::Type{<:ExampleInstanceJuMP}) = (length(inst) > 1 ? string(inst[2]) : "")
get_extender(inst::Tuple, ::Type{<:ExampleInstance}) = ""

function write_perf(
    perf::DataFrames.DataFrame,
    results_path::Union{String, Nothing},
    new_perf::NamedTuple,
    )
    push!(perf, new_perf, cols = :subset)
    if !isnothing(results_path)
        CSV.write(results_path, perf[end:end, :], transform = (col, val) -> something(val, missing), append = true)
    end
    return
end

function run_instance_set(
    inst_subset::Vector,
    ex_type_T::Type{<:ExampleInstance},
    info_perf::NamedTuple,
    new_default_options::NamedTuple,
    script_verbose::Bool,
    perf::DataFrames.DataFrame,
    results_path::Union{String, Nothing},
    )
    for (inst_num, inst) in enumerate(inst_subset)
        extender_name = get_extender(inst, ex_type_T)
        test_info = "inst $inst_num: $(inst[1]) $extender_name"
        @testset "$test_info" begin
            println(test_info, " ...")

            total_time = @elapsed run_perf = run_instance(ex_type_T, inst...,
                default_options = new_default_options, verbose = script_verbose)

            new_perf = (;
                info_perf..., run_perf..., total_time, inst_num,
                :solver => "Hypatia", :inst_data => inst[1],
                :extender => extender_name,
                )
            write_perf(perf, results_path, new_perf)

            @printf("%8.2e seconds\n", total_time)
        end
    end
    return
end
