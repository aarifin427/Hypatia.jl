#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

see "A Direct Formulation for Sparse PCA Using Semidefinite Programming" by
Alexandre d’Aspremont, Laurent El Ghaoui, Michael I. Jordan, Gert R. G. Lanckriet
==#

using LinearAlgebra
import Distributions
import Random
using Test
import Hypatia
import Hypatia.HypReal
const CO = Hypatia.Cones

function sparsepca(
    T::Type{<:HypReal},
    p::Int,
    k::Int;
    use_l1ball::Bool = true,
    noise_ratio::Float64 = 0.0,
    )
    @assert 0 < k <= p

    signal_idxs = Distributions.sample(1:p, k, replace = false) # sample components that will carry the signal
    if noise_ratio <= 0.0
        # noiseless model
        x = zeros(T, p)
        x[signal_idxs] = rand(T, k)
        sigma = x * x'
        sigma ./= tr(sigma)
        true_obj = -1
    else
        # simulate some observations with noise
        x = randn(p, 100)
        sigma = x * x'
        y = rand(Distributions.Normal(0, noise_ratio), k)
        sigma[signal_idxs, signal_idxs] .+= y * y'
        sigma ./= 100
        sigma = T.(sigma)
        true_obj = NaN
    end

    rt2 = sqrt(T(2))
    dimx = div(p * (p + 1), 2)
    # x will be the svec (lower triangle, row-wise) of the matrix solution we seek
    c = T[-sigma[i, j] * (i == j ? 1 : rt2) for i in 1:p for j in 1:i]
    b = T[1]
    A = zeros(T, 1, dimx)
    # PSD cone, x is already vectorized and scaled
    Gpsd = Matrix{T}(-I, dimx, dimx)
    for i in 1:p
        s = sum(1:i)
        A[s] = 1
    end
    hpsd = zeros(T, dimx)
    cones = CO.Cone{T}[CO.PosSemidef{T, T}(dimx)]
    cone_idxs = [1:dimx]

    if use_l1ball
        # l1 cone
        Gl1 = Matrix{T}(-rt2 * I, dimx, dimx) # double off-diagonals, which are already scaled by rt2
        for i in 1:p
            s = sum(1:i)
            Gl1[s, s] = -1
        end
        G = vcat(Gpsd, zeros(T, 1, dimx), Gl1)
        h = vcat(hpsd, T(k), zeros(T, dimx))
        push!(cones, CO.EpiNormInf{T}(1 + dimx, true))
        push!(cone_idxs, (dimx + 1):(2 * dimx + 1))
    else
        c = vcat(c, zeros(T, 2 * dimx))
        id = Matrix{T}(I, dimx, dimx)
        l1 = [(i == j ? one(T) : rt2) for i in 1:p for j in 1:i]
        A = T[
            A    zeros(T, 1, 2 * dimx);
            -id    -id    id;
            zeros(T, 1, dimx)    l1'    l1';
            ]
        b = vcat(b, zeros(T, dimx), k)
        G = [
            Gpsd    zeros(T, dimx, 2 * dimx);
            zeros(T, 2 * dimx, dimx)    Matrix{T}(-I, 2 * dimx, 2 * dimx);
            ]
        h = vcat(hpsd, zeros(T, 2 * dimx))
        push!(cones, CO.Nonnegative{T}(2 * dimx))
        push!(cone_idxs, (dimx + 1):(3 * dimx))
    end

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs, true_obj = true_obj)
end

sparsepca1(T::Type{<:HypReal}) = sparsepca(T, 5, 3)
sparsepca2(T::Type{<:HypReal}) = sparsepca(T, 5, 3, use_l1ball = false)
sparsepca3(T::Type{<:HypReal}) = sparsepca(T, 10, 3)
sparsepca4(T::Type{<:HypReal}) = sparsepca(T, 10, 3, use_l1ball = false)
sparsepca5(T::Type{<:HypReal}) = sparsepca(T, 10, 3, noise_ratio = 10.0)
sparsepca6(T::Type{<:HypReal}) = sparsepca(T, 10, 3, noise_ratio = 10.0, use_l1ball = false)

instances_sparsepca_all = [
    sparsepca1,
    sparsepca2,
    sparsepca3,
    sparsepca4,
    sparsepca5,
    sparsepca6,
    ]
instances_sparsepca_few = [
    sparsepca1,
    sparsepca2,
    ]

function test_sparsepca(instance::Function; T::Type{<:HypReal} = Float64, test_options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    d = instance(T)
    r = Hypatia.Solvers.build_solve_check(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs; test_options..., atol = tol, rtol = tol)
    @test r.status == :Optimal
    if !isnan(d.true_obj)
        @test r.primal_obj ≈ d.true_obj atol = tol rtol = tol
    end
    return
end