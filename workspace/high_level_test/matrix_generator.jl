using Random
using SymPy
using Distributions
using LinearAlgebra

"""
(1) generates d x d symmetric matrix for copy pasting
(2) tests hyperbolicity

"""

d = 4
n = 4
max_range = 5
mat_list = []
for k = 2:n
    a = rand(1:max_range, d, d)
    a = a'*a
    push!(mat_list, a)
    println("A"*string(k)*" = [")
    for i=1:d
        print("    ")
        for j=1:d
            print(string(a[i,j]))
            print(" ")
        end
        if i != d
            print(";")
        end
        println()
    end
    println("]")
end

# test #
T = Float64
p(x) = det(1.0*I(d)*x[1] + mat_list[1]*x[2] + mat_list[2]*x[3] + mat_list[3]*x[4])
init_point = 1.0*[1,0,0,0]

t = symbols("t")
e = init_point

for k = 1:20
    local x = rand(Uniform(-50,50), n)
    
    a = SymPy.solve(p(x - t*e))

    is_feas = true
    for i in eachindex(a)
        # if !isreal(a[i]) || abs(a[i]) < eps(T)
        if abs(imag(a[i])) > 1e-7 || abs(a[i]) < eps(T)
            # not real, infeasible (?)
            is_feas = false
            println(false)
            return
            break
        end
    end
    println(true)
end