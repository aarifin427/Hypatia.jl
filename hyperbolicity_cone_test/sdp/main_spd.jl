# src: https://jump.dev/JuMP.jl/stable/reference/constraints/#JuMP.PSDCone

using JuMP, LinearAlgebra
using SCS

function jump_SDP(A_mat, A, b, c, n)
    model = Model(SCS.Optimizer)

    @variable(model, X[1:n])
    # documentation on @constraint macro: https://jump.dev/JuMP.jl/stable/reference/constraints/
    @constraint(model, sum(A_mat.*X) in PSDCone())
    @constraint(model, A*X .== b)
    @objective(model, Min, c'*X)

    optimize!(model)

    status = JuMP.termination_status(model)
    X_sol = X
    obj_value = JuMP.objective_value(model)

    return status, X_sol, obj_value
end

# Example: Vamos Polynomial (https://arxiv.org/abs/1306.4483)
A1 = [
    0 0 0 0 0 0 0
    0 1 2 0 1 0 0
    0 2 6 0 4 0 0
    0 0 0 4 0 0 0
    0 1 4 0 4 0 0
    0 0 0 0 0 0 0
    0 0 0 0 0 0 0
];

A2 = [
    1 1 2 1 0 2 1
    1 4 4 4 0 4 4
    2 4 9 4 0 8 4
    1 4 4 4 0 4 4
    0 0 0 0 0 0 0
    2 4 8 4 0 8 4
    1 4 4 4 0 4 4
];

A3 = [
    3 3 10 0 3 4 0
    3 4 12 0 4 4 0
    10 12 41 0 12 16 0
    0 0 0 0 0 0 0
    3 4 12 0 4 4 0
    4 4 16 0 4 8 0
    0 0 0 0 0 0 0
];

A4 = [
    2 2 2 0 0 0 0
    2 7 3 0 0 0 4
    2 3 4 0 0 0 0
    0 0 0 0 0 0 0
    0 0 0 0 0 0 0
    0 0 0 0 0 0 0
    0 4 0 0 0 0 4
];

# (2)
A = [A1, A2, A3, A4];
c = [
    0.6485574918878491;
    0.9051391876123972;
    0.5169347627896711;
    0.8591212789374901
]
AA = 1.0*[0.1 1 1 0.1]
b = [0.5]
n = 4

"""
# Example usage
status, X_sol, obj_value = jump_SDP(A, AA, b, c, n);
println("status: ", status)

for i in 1:4
    println(X_sol[i], "=", value(X_sol[i]))
end
"""