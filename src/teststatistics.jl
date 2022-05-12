function opt_maximize(f, inits; alg = Optim.NelderMead())
    Optim.maximize(f, inits, alg, Optim.Options(g_tol = 1e-5, iterations=10^4); autodiff=:forward)
end

abstract type AbstractTestStatistics <: Distributions.ContinuousUnivariateDistribution end
const ATS = AbstractTestStatistics

struct TS_t0 <: ATS end
struct TS_tmu <: ATS end
struct TS_tmutilde <: ATS end
struct TS_q0 <: ATS end
struct TS_qmu <: ATS end
struct TS_qmutilde <: ATS end

teststatistics(T::ATS) = error("Not implemented $T.")

function teststatistics(model, poi_test, ::TS_qmu)
    A_data, A_mubhathat = asimov_data(model, poi_test)
    qmu_f = get_qmu(model.LL, model.inits)
    qmuA_v, (mubhathat_A, muhatbhat_A) = qmu_f()

end

"""
    get_condLL(LL, μ)

Given the original log-likelihood function and a value for parameter of interest, return a function `condLL(nuisance_θs)`
that takes one less argument than the original `LL`. The `μ` is assumed to be the first element in input vector.
"""
function get_condLL(LL, μ)
    function condLL(nuisance_θs)
        θs = vcat(μ, nuisance_θs)
        LL(θs)
    end
end

@doc raw"""
    get_lnLR(LL, inits)

A functional that returns a function `lnLR(μ::Number)` that evaluates to log of likelihood-ratio:

```math
\ln\lambda(\mu) = \ln(\frac{L(\mu, \hat{\hat\theta)}}{L(\hat\mu, \hat\theta)}) = LL(\mu, \hat{\hat\theta}) - LL(\hat\mu, \hat\theta)
```

!!! warning
    we assume the POI is the first in the input array.
"""
function get_lnLR(LL, inits; POI_idx=1)
    fit = opt_maximize(LL, inits)
    θ0 = Optim.maximizer(fit)
    LL_doublehat = maximum(fit)
    nuisance_inits = inits[2:end]

    function lnLR(μ)
        cond_LL= get_condLL(LL, μ)
        cond_fit = opt_maximize(cond_LL, nuisance_inits)
        LL_hat = maximum(cond_fit)
        return LL_hat - LL_doublehat
    end
end

@doc raw"""
    get_lnLRtilde(LL, inits)

A functional that returns a function `lnLRtilde(μ::Number)` that evaluates to log of likelihood-ratio:

```math
\ln\widetilde{\lambda(\mu)}
```

See equation 10 in: https://arxiv.org/pdf/1007.1727.pdf for refercen.
"""
function get_lnLRtilde(LL, inits; POI_idx=1)
    fit = opt_maximize(LL, inits)
    θ_hat = Optim.maximizer(fit)
    if θ_hat[1] < 0 # re-fit with μ set to 0
        fit = opt_maximize(get_condLL(0.0), inits[2:end])
    end
    LL_doublehat = maximum(fit)
    nuisance_inits = inits[2:end]

    function lnLRtilde(μ)
        cond_LL = get_condLL(LL, μ)
        cond_fit = opt_maximize(cond_LL, nuisance_inits)
        LL_hat = maximum(cond_fit)
        return LL_hat - LL_doublehat
    end
end

@doc raw"""
    get_tmu(LL, inits)

Return a callable function `t(μ)` that is the test statistics:
```math
    t_\mu = -2\ln\lambda(\mu)
```
"""
function get_tmu(LL, inits)
    lnLR = get_lnLR(LL, inits)
    function t(μ)
        -2*lnLR(μ)
    end
end

@doc raw"""
    get_tmutilde(LL, inits)

Return a callable function `ttilde(μ)` that is:
```math
    \widetilde{t_\mu} = -2\ln\widetilde{\lambda(\mu)}
```
"""
function get_tmutilde(LL, inits)
    lnLRtilde = get_lnLRtilde(LL, inits)
    function ttilde(μ)
        -2*lnLRtilde(μ)
    end
end

@doc raw"""
Test statistic for discovery of a positive signal
q_0 = \tilde{t}_0
See equation 12 in: https://arxiv.org/pdf/1007.1727.pdf for reference.
Note that this IS NOT a special case of q_\mu for \mu = 0.
"""
function get_q0(LL, inits)
    tmutilde = get_tmutilde(LL, inits)
    res = tmutilde(0)
    function q0(μ)
        @assert iszero(μ) #q0 is forced to have μ == 0
        return res
    end
end

@doc raw"""
Test statistic for upper limits
See equation 14 in: https://arxiv.org/pdf/1007.1727.pdf for reference.
Note that q_0 IS NOT a special case of q_\mu for \mu = 0.
"""
function get_qmu(LL, inits)
    fit = opt_maximize(LL, inits)
    θ0 = Optim.maximizer(fit)
    μ_hat = θ0[1]
    function qmu(μ)
        if μ_hat <= μ
            lnLR = get_lnLR(LL, inits)
            -2*lnLR(μ)
        else
            0
        end
    end
end

"""
    asimovdata(model::PyHFModel, μ)

Generate the Asimov dataset and asimov parameters, which is the expected counts after fixing POI to `μ` and optimize the
nuisance parameters.
"""
function asimovdata(model, μ)
    condLL = get_condLL(model.LogLikelihood, μ)
    nuiscance_inits = model.inits[2:end]
    cond_res = opt_maximize(cond_LL, nuisance_inits)
    asimov_params = vcat(μ, Optim.maximizer(cond_res)) 
    model.expected(asimov_params), asimov_params
end

"""
Compute the `q_μ`.
"""
function qmu(model, μ, test_func)
    test_func(model, μ, )
end

"""
Compute the `q_μ,Asimov`.
"""
function qmuA(model, μ, test_func)
    Adata = asimovdata(model, μ)
    test_func(model, μ, Adata)
end

abstract type AbstractTestDistributions end
struct AsymptoticDist<: AbstractTestDistributions
    shift::Float64
    cutoff::Float64
end
pvalue(d::AsymptoticDist, value) = cdf(Normal(), -(value - d.shift))
exp_significance(d::AsymptoticDist, nsigma) = max(d.cutoff, d.shift + nsigma)
