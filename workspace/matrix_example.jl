using LinearAlgebra
using JuMP
using Hypatia

# Taken from https://chriscoey.github.io/Hypatia.jl/dev/solving/

Hypatia.greet()

opt = Hypatia.Optimizer(verbose = false)
model = Model(() -> opt)
@variable(model, x[1:3] >= 0)
@constraint(model, sum(x) == 5)
@variable(model, hypo)
@objective(model, Max, hypo)
V = rand(2, 3)
Q = V * diagm(x) * V'
aff = vcat(hypo, [Q[i, j] for i in 1:2 for j in 1:i]...)

# The following cone only works with MOI's package, AND matrix inputs.
@constraint(model, aff in MOI.PositiveSemidefiniteConeSquare(2))

optimize!(model)
termination_status(model)
objective_value(model)
value.(x)


