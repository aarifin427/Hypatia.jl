## Constructing hyperbolic programming
Follow the example from `general_test/test2.jl`. You may toggle between numerical and symbolical approaches to compute the feasibility oracle.

All examples use the Hypatia's native interface to optimize problems.

#### Constructing the cone
Define the following:
1. Callable function `p` that evaluates the hyperbolic polynomial *p(x)* at point x. This can be defined as a mathematical expression or a black box containing loops or conditionals.
2. Initial point in the cone `init_point`. The vector *e* that corresponds to *p(x)* will suffice
3. Callable function `grad` that evaluates the gradient of the barrier function at point x. 
4. Callable function `hess` that evaluates the gradient of the barrier function at point x. 
5. Dimension `n`
6. [Optional] Degree of polynomial *p(x)*

For items no. 2-3, `ForwardDiff.jl` is one option to define the corresponding callable functions. All examples use `ForwardDiff.jl`.

*Note*: `ForwardDiff.jl` cannot accept negative values to a log function. The example takes extra steps of the chain rule for both the gradient and hessian.

For gradient:
```julia
# Instead of the following...
grad = x -> ForwardDiff.gradient(x->-log(p(x)),x)

# ... use this
grad = x -> - 1/p(x) * ForwardDiff.gradient(x->p(x),x)
```

For hessian:
```julia
# Instead of the following...
hess = x -> ForwardDiff.hessian(x -> -log(p(x)), x)

# ... use this
dpx = x -> ForwardDiff.gradient(x->p(x),x)
hess = x -> (-ForwardDiff.hessian(x -> p(x), x) * p(x) + dpx(x)*dpx(x)')/(p(x)^2)
```

#### Using numerical approach for feasibility oracle
The numerical approach uses `PolynomialRoots.jl`. 

Define the hyperbolicity cone with the optional input `d` for the degree of your hyperbolic polynomial.
```julia
a_cone = Cones.Hyperbolicity{T}(n, p, grad, hess, e, d=4)
```

#### Using symbolic approach for feasibility oracle
The numerical approach uses `SymPy.jl`. 

Define the hyperbolicity cone with only the required inputs.
```julia
a_cone = Cones.Hyperbolicity{T}(n, p, grad, hess, e)
```

##### Special case: derivative relaxation
Definition of derivative relaxation is taken from [here](https://arxiv.org/abs/1208.1443).

Taken from `general_test/test9.jl`, use `SymPy.jl` package to take the derivative of a hyperbolic polynomial as follows.

```julia
using SymPy.jl

# A hyperbolic polynomial
hp(x) = x[1]*x[2]*x[3]

@vars t
# derivative relaxation
p1 = x -> diff(hp(x + t*e), t).subs(t, 0)
```

Use `p1` as the hyperbolic polynomial input. Define other required inputs as per usual.
```julia
a_cone = Cones.Cone{T}[Cones.Hyperbolicity{T}(n, p1, grad, hess, e, d=2)]
```