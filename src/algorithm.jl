#=
Copyright 2018, David Papp, Sercan Yildiz, and contributors

an implementation of the algorithm for non-symmetric conic optimization Alfonso (https://github.com/dpapp-github/alfonso) and analyzed in the paper:
D. Papp and S. Yildiz. On "A homogeneous interior-point algorithm for nonsymmetric convex conic optimization"
available at https://arxiv.org/abs/1712.00492
=#

function runalgorithm!(alf::AlfonsoOpt)
    (A, b, c) = (alf.A, alf.b, alf.c)
    (m, n) = size(A)
    coneobjs = alf.cones
    coneidxs = alf.coneidxs

    # calculate complexity parameter of the augmented barrier (nu-bar): sum of the primitive cone barrier parameters (# TODO plus 1?)
    bnu = 1.0 + sum(barpar(ck) for ck in coneobjs)

    # create cone object functions related to primal cone barrier
    function load_tx(_tx; save_prev=false)
        for k in eachindex(coneobjs)
            load_txk(coneobjs[k], _tx[coneidxs[k]], save_prev)
        end
        return nothing
    end

    function use_prev()
        for k in eachindex(coneobjs)
            use_prevk(coneobjs[k])
        end
        return nothing
    end

    function check_incone()
        for k in eachindex(coneobjs)
            if !inconek(coneobjs[k])
                return false
            end
        end
        return true
    end

    function calc_g!(_g)
        for k in eachindex(coneobjs)
            _g[coneidxs[k]] .= calc_gk(coneobjs[k])
        end
        return _g
    end

    function calc_Hinv_vec!(_Hi_vec, _v)
        for k in eachindex(coneobjs)
            _Hi_vec[coneidxs[k]] .= calc_Hinvk(coneobjs[k])*_v[coneidxs[k]]
        end
        return _Hi_vec
    end

    function calc_Hinv_At!(_Hi_At)
        for k in eachindex(coneobjs)
            _Hi_At[coneidxs[k],:] .= calc_Hinvk(coneobjs[k])*A[:,coneidxs[k]]' # TODO maybe faster with CSC to do IJV
        end
        return _Hi_At
    end

    function calc_nbhd(_ts, _mu, _tk)
        # sqrt(sum(abs2, L\(ts + mu*g)) + (tau*kap - mu)^2)/mu
        sumsqr = (_tk - _mu)^2
        for k in eachindex(coneobjs)
            sumsqr += sum(abs2, calc_HCholLk(coneobjs[k])\(_ts[coneidxs[k]] + _mu*calc_gk(coneobjs[k])))
        end
        return sqrt(sumsqr)/_mu
     end

    # set remaining algorithmic parameters based on precomputed safe values (from original authors)
    # parameters are chosen to make sure that each predictor step takes the current iterate from the eta-neighborhood to the beta-neighborhood and each corrector phase takes the current iterate from the beta-neighborhood to the eta-neighborhood. extra corrector steps are allowed to mitigate the effects of finite precision
    (beta, eta, cpredfix) = setbetaeta(alf.maxcorrsteps, bnu) # beta: large neighborhood parameter, eta: small neighborhood parameter
    alphapredfix = cpredfix/(eta + sqrt(2.0*eta^2 + bnu)) # fixed predictor step size
    alphapredls = min(100.0*alphapredfix, 0.9999) # initial predictor step size with line search
    alphapredthres = (alf.predlsmulti^alf.maxpredsmallsteps)*alphapredfix # minimum predictor step size
    alphapred = (alf.predlinesearch ? alphapredls : alphapredfix) # predictor step size

    #=
    setup data and functions needed in main loop
    =#
    # termination tolerances are infinity operator norms of submatrices of lhs
    tol_pres = max(1.0, maximum(sum(abs, A[i,:]) + abs(b[i]) for i in 1:m)) # first m rows
    tol_dres = max(1.0, maximum(sum(abs, A[:,j]) + abs(c[j]) + 1.0 for j in 1:n)) # next n rows
    tol_compl = max(1.0, maximum(abs, b), maximum(abs, c)) # row m+n+1

    # calculate initial primal iterate tx
    # scaling factor for the primal problem
    rp = maximum((1.0 + abs(b[i]))/(1.0 + abs(sum(A[i,:]))) for i in 1:m)
    # scaling factor for the dual problem
    g = ones(n)
    load_tx(g)
    @assert check_incone() # TODO will fail in general? TODO for some cones will not automatically calculate g,H
    calc_g!(g)
    rd = maximum((1.0 + abs(g[j]))/(1.0 + abs(c[j])) for j in 1:n)
    # initial primal iterate
    tx = fill(sqrt(rp*rd), n)

    # calculate central primal-dual iterate
    load_tx(tx)
    @assert check_incone()
    ty = zeros(m)
    tau = 1.0
    ts = -calc_g!(g)
    kap = 1.0
    mu = (dot(tx, ts) + tau*kap)/bnu

    # preallocate for test iterate and direction vectors
    rhs_ty = similar(ty)
    rhs_tx = similar(tx)
    dir_ty = similar(ty)
    dir_tx = similar(tx)
    dir_ts = similar(ts)
    Hic = similar(tx)
    HiAt = similar(A, n, m)
    Hirxrs = similar(tx)
    lhsdydtau = similar(A, m+1, m+1)
    rhsdydtau = similar(tx, m+1)
    dydtau = similar(rhsdydtau)
    sa_tx = similar(tx)
    sa_ts = similar(ts)

    #=
    main loop
    =#
    if alf.verbose
        @printf("\n%5s %12s %12s %9s %9s %9s %9s %9s %9s\n", "iter", "p_obj", "d_obj", "gap", "p_inf", "d_inf", "tau", "kap", "mu")
        flush(stdout)
    end

    alf.status = :StartedIterating
    iter = 0
    while true
        #=
        calculate convergence metrics, check criteria, print
        =#
        ctx = dot(c, tx)
        bty = dot(b, ty)
        p_obj = ctx/tau
        d_obj = bty/tau
        gap = abs(ctx - bty)/(tau + abs(bty))
        rhs_ty .= -A*tx + b*tau
        p_inf = maximum(abs, rhs_ty)/tol_pres
        rhs_tx .= A'*ty - c*tau + ts
        d_inf = maximum(abs, rhs_tx)/tol_dres
        rhs_tau = -bty + ctx + kap
        compl = abs(rhs_tau)/tol_compl

        if alf.verbose
            @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n", iter, p_obj, d_obj, gap, p_inf, d_inf, tau, kap, mu)
            flush(stdout)
        end

        if (p_inf <= alf.optimtol) && (d_inf <= alf.optimtol)
            if gap <= alf.optimtol
                println("Problem is feasible and an approximate optimal solution was found; terminating")
                alf.status = :Optimal
                break
            elseif (compl <= alf.optimtol) && (tau <= alf.optimtol*1e-02*max(1.0, kap))
                println("Problem is nearly primal or dual infeasible; terminating")
                alf.status = :NearlyInfeasible
                break
            end
        elseif (tau <= alf.optimtol*1e-02*min(1.0, kap)) && (mu <= alf.optimtol*1e-02)
            println("Problem is ill-posed; terminating")
            alf.status = :IllPosed
            break
        end

        iter += 1
        if iter >= alf.maxiter
            alf.status = :IterationLimit
        end

        #=
        prediction phase
        =#
        # determine prediction direction
        invmu = 1.0/mu
        calc_Hinv_vec!(Hic, c)
        Hic .*= invmu
        calc_Hinv_At!(HiAt)
        HiAt .*= invmu
        dir_ts .= invmu*(rhs_tx - ts)
        calc_Hinv_vec!(Hirxrs, dir_ts)

        # TODO maybe can use special structure of lhsdydtau: top left mxm is symmetric (L*A)^2, then last row and col are skew-symmetric
        lhsdydtau .= [A*HiAt (-b - A*Hic); (b' - c'*HiAt) (mu/tau^2 + dot(c, Hic))]
        rhsdydtau .= [(rhs_ty - A*Hirxrs); (rhs_tau - kap + dot(c, Hirxrs))]
        dydtau .= lhsdydtau\rhsdydtau

        dir_ty .= dydtau[1:m]
        dir_tau = dydtau[m+1]
        dir_tx .= Hirxrs + HiAt*dir_ty - Hic*dir_tau
        dir_ts .= -rhs_tx - A'*dir_ty + c*dir_tau
        dir_kap = -rhs_tau + dot(b, dir_ty) - dot(c, dir_tx)

        # determine step length alpha by line search
        alpha = alphapred
        nbhd_beta = Inf
        alphaprevok = true
        alphaprev = 0.0
        nbhd_betaprev = Inf
        predfail = false
        nprediters = 0
        while true
            nprediters += 1

            sa_tx .= tx + alpha*dir_tx
            load_tx(sa_tx, save_prev=(alphaprevok && (nprediters > 1)))

            if check_incone()
                # primal iterate tx is inside the cone
                sa_ts .= ts + alpha*dir_ts
                sa_tk = (tau + alpha*dir_tau)*(kap + alpha*dir_kap)
                sa_mu = (dot(sa_tx, sa_ts) + sa_tk)/bnu
                nbhd_beta = calc_nbhd(sa_ts, sa_mu, sa_tk)

                if nbhd_beta < beta
                    # iterate is inside the beta-neighborhood
                    if !alphaprevok || (alpha > alf.predlsmulti)
                        # either the previous iterate was outside the beta-neighborhood or increasing alpha again will make it > 1
                        if alf.predlinesearch
                            alphapred = alpha
                        end
                        break
                    end

                    alphaprevok = true
                    alphaprev = alpha
                    nbhd_betaprev = nbhd_beta
                    alpha = alpha/alf.predlsmulti
                    continue
                end
            end

            # primal iterate tx is outside the beta-neighborhood
            if alphaprevok && (nprediters > 1)
                # previous iterate was in the beta-neighborhood
                alpha = alphaprev
                nbhd_beta = nbhd_betaprev
                if alf.predlinesearch
                    alphapred = alpha
                end
                use_prev()
                break
            end

            # current and previous primal iterates are outside the beta-neighborhood
            if alpha < alphapredthres
                # alpha is very small, so predictor has failed
                predfail = true
                println("Predictor could not improve the solution; terminating")
                alf.status = :PredictorFail
                break
            end

            alphaprevok = false
            alphaprev = alpha
            nbhd_betaprev = nbhd_beta
            alpha = alf.predlsmulti*alpha
        end
        # @show nprediters
        if predfail
            break
        end

        # step distance alpha in the direction
        ty .+= alpha*dir_ty
        tx .+= alpha*dir_tx
        tau += alpha*dir_tau
        ts .+= alpha*dir_ts
        kap += alpha*dir_kap
        mu = (dot(tx, ts) + tau*kap)/bnu

        # skip correction phase if allowed and current iterate is in the eta-neighborhood
        if alf.corrcheck && (nbhd_beta <= eta)
            continue
        end

        #=
        correction phase: perform correction steps
        =#
        corrfail = false
        ncorrsteps = 0
        while true
            ncorrsteps += 1

            # calculate correction direction
            invmu = 1.0/mu
            calc_Hinv_vec!(Hic, c)
            Hic .*= invmu
            calc_Hinv_At!(HiAt)
            HiAt .*= invmu
            dir_ts .= -invmu*ts - calc_g!(g)
            calc_Hinv_vec!(Hirxrs, dir_ts)

            lhsdydtau .= [A*HiAt (-b - A*Hic); (b' - c'*HiAt) (mu/tau^2 + dot(c, Hic))]
            rhsdydtau .= [-A*Hirxrs; (-kap + mu/tau + dot(c, Hirxrs))]
            dydtau .= lhsdydtau\rhsdydtau

            dir_ty .= dydtau[1:m]
            dir_tau = dydtau[m+1]
            dir_tx .= Hirxrs + HiAt*dir_ty - Hic*dir_tau
            dir_ts .= -A'*dir_ty + c*dir_tau
            dir_kap = dot(b, dir_ty) - dot(c, dir_tx)

            # determine step length alpha by line search
            alpha = alf.alphacorr
            ncorrlsiters = 0
            while ncorrlsiters <= alf.maxcorrlsiters
                ncorrlsiters += 1

                sa_tx .= tx + alpha*dir_tx
                load_tx(sa_tx)

                if check_incone() # TODO only calculates everything for the nonnegpoly cone - for others need to make sure g,H are calculated after correction step
                    # primal iterate tx is inside the cone, so terminate line search
                    break
                end

                # primal iterate tx is outside the cone
                if ncorrlsiters == alf.maxcorrlsiters
                    # corrector failed
                    corrfail = true
                    println("Corrector could not improve the solution; terminating")
                    alf.status = :CorrectorFail
                    break
                end

                alpha = alf.corrlsmulti*alpha
            end
            # @show ncorrlsiters
            if corrfail
                break
            end

            # step distance alpha in the direction
            ty .+= alpha*dir_ty
            tx .= sa_tx
            tau += alpha*dir_tau
            ts .+= alpha*dir_ts
            kap += alpha*dir_kap
            mu = (dot(tx, ts) + tau*kap)/bnu

            # finish if allowed and current iterate is in the eta-neighborhood, or if taken max steps
            if (ncorrsteps == alf.maxcorrsteps) || alf.corrcheck
                if calc_nbhd(ts, mu, tau*kap) <= eta
                    break
                elseif ncorrsteps == alf.maxcorrsteps
                    # nbhd_eta > eta, so corrector failed
                    corrfail = true
                    println("Corrector phase finished outside the eta-neighborhood; terminating")
                    alf.status = :CorrectorFail
                    break
                end
            end
        end
        # @show ncorrsteps
        if corrfail
            break
        end
    end

    println("\nFinished in $iter iterations\nInternal status is $(alf.status)\n")

    #=
    calculate final solution and iteration statistics
    =#
    alf.niters = iter

    alf.x = tx./tau
    alf.y = ty./tau
    alf.tau = tau
    alf.s = ts./tau
    alf.kap = kap

    alf.pobj = dot(c, alf.x)
    alf.dobj = dot(b, alf.y)
    alf.dgap = alf.pobj - alf.dobj
    alf.cgap = dot(alf.s, alf.x)
    alf.rel_dgap = alf.dgap/(1.0 + abs(alf.pobj) + abs(alf.dobj))
    alf.rel_cgap = alf.cgap/(1.0 + abs(alf.pobj) + abs(alf.dobj))

    alf.pres = b - A*alf.x
    alf.dres = c - A'*alf.y - alf.s
    alf.pin = norm(alf.pres)
    alf.din = norm(alf.dres)
    alf.rel_pin = alf.pin/(1.0 + norm(b, Inf))
    alf.rel_din = alf.din/(1.0 + norm(c, Inf))

    return nothing
end


function setbetaeta(maxcorrsteps, bnu)
    if maxcorrsteps <= 2
        if bnu < 10.0
            return (0.1810, 0.0733, 0.0225)
        elseif bnu < 100.0
            return (0.2054, 0.0806, 0.0263)
        else
            return (0.2190, 0.0836, 0.0288)
        end
    elseif maxcorrsteps <= 4
        if bnu < 10.0
            return (0.2084, 0.0502, 0.0328)
        elseif bnu < 100.0
            return (0.2356, 0.0544, 0.0380)
        else
            return (0.2506, 0.0558, 0.0411)
        end
    else
        if bnu < 10.0
            return (0.2387, 0.0305, 0.0429)
        elseif bnu < 100.0
            return (0.2683, 0.0327, 0.0489)
        else
            return (0.2844, 0.0332, 0.0525)
        end
    end
end
