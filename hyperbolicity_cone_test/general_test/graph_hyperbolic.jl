using Combinatorics
"""
    get_p(n, edge_set, x)

Outputs hyperbolic polynomial anonymous function for r = 2. Ref: (https://arxiv.org/pdf/1512.05878.pdf)

# Arguments
- `n`: number of vertices in a network
- `edge_set`: edge list described as [a, b, w] where a,b ∈ {1, ..., n} and w ∈ [0,1]
- `x`: variable x for anonymous function

# Examples
```
f(x) = get_p(2, [[1,2,0.1], [1,3,0.3]], x)
```
"""
function get_p(n, edge_set, x)
    """
    n           = dimension of function p
    edge_set    = an edge described as [a, b, w] where a,b ∈ {1, ..., n} and w ∈ [0,1]
    x           = value
    """
    r = 2
    # Define e_(r-1)(x) as e1
    e1 = sum(x[i] for i=1:n)

    # Define e_(r+1)(x) as e3
    e3 = 0
    for (i, j, k) in collect(Combinatorics.combinations(1:n, r+1))
        e3 += x[i]*x[j]*x[k]
    end

    # Define M(x) as weighted sum
    weighted_sum = 0
    for (a,b,w) in edge_set
        weighted_sum += w*x[convert(Int64,a)]^2*x[convert(Int64,b)]^2
    end

    # 4*e_(r+1)(x)*e_(r-1)(x) + 3/(r+1)*M(x)
    return 4*e3*e1 + weighted_sum
end