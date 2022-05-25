function free_maximize(f, inits; alg = Optim.LBFGS())
    res = Optim.maximize(f, inits, alg, Optim.Options(g_tol = 1e-9, iterations=10^4); autodiff=:forward)
    @assert Optim.converged(res)
    maximum(res), Optim.maximizer(res)
end
free_maximize(m::PyHFModel; kwd...) = free_maximize(pyhf_logjointof(m), m.inits; kwd...)

function cond_maximize(LL, μ, partial_inits; kwd...)
    cond_LL= get_condLL(LL, μ)
    free_maximize(cond_LL, partial_inits; kwd...)
end
cond_maximize(m::PyHFModel, μ) = cond_maximize(pyhf_logjointof(m), μ, m.inits[2:end])

const fix_poi_fit = cond_maximize

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
    LL_hat, θ0_hat = free_maximize(LL, inits)
    nuisance_inits = inits[2:end]

    function lnLR(μ)
        LL_doublehat, _ = cond_maximize(LL, μ, nuisance_inits)
        return LL_doublehat - LL_hat
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
function get_lnLRtilde(LL, inits)
    LL_hat, θ_hat = free_maximize(LL, inits)
    nuisance_inits = inits[2:end]
    if θ_hat[1] < 0 # re-fit with μ set to 0
        LL_hat, _ = cond_maximize(LL, 0.0, nuisance_inits)
    end

    function lnLRtilde(μ)
        LL_doublehat, _ = cond_maximize(LL, μ, nuisance_inits)
        return LL_doublehat - LL_hat
    end
end

abstract type AbstractTestStatistics end
const ATS = AbstractTestStatistics

@doc raw"""
    T_tmu

```math
    t_\mu = -2\ln\lambda(\mu)
```
"""
struct T_tmu <: ATS end

@doc raw"""
    T_tmutilde(LL, inits)

```math
    \widetilde{t_\mu} = -2\ln\widetilde{\lambda(\mu)}
```
"""
struct T_tmutilde <: ATS end

@doc raw"""
Test statistic for discovery of a positive signal
q_0 = \tilde{t}_0
See equation 12 in: https://arxiv.org/pdf/1007.1727.pdf for reference.
Note that this IS NOT a special case of q_\mu for \mu = 0.
"""
struct T_q0 <: ATS end

@doc raw"""
Test statistic for upper limits
See equation 14 in: https://arxiv.org/pdf/1007.1727.pdf for reference.
Note that q_0 IS NOT a special case of q_\mu for \mu = 0.
"""
struct T_qmu <: ATS end

struct T_qmutilde <: ATS end

@doc raw"""
    get_teststat(LL, inits, ::Type{T}) where T <: ATS

Return a callable function `t(μ)` that evaluates to the value of corresponding test statistics.
"""
function get_teststat(LL, inits, ::Type{T_tmu})
    lnLR = get_lnLR(LL, inits)
    function t(μ)
        max(0.0, -2*lnLR(μ))
    end
end

function get_teststat(LL, inits, ::Type{T_tmutilde})
    lnLRtilde = get_lnLRtilde(LL, inits)
    function tmutilde(μ)
        max(0.0, -2*lnLRtilde(μ))
    end
end

function get_teststat(LL, inits, ::Type{T_q0})
    res = get_teststate(LL, inits, T_tmutilde)(0.0)
    function q0(μ=0)
        @assert iszero(μ) "q0 by definition demands μ=0" #q0 is forced to have μ == 0
        return max(res, 0.0)
    end
end

function get_teststat(LL, inits, ::Type{T_qmu})
    _, θ0 = free_maximize(LL, inits)
    μ_hat = θ0[1]
    function qmu(μ)
        if μ_hat <= μ
            lnLR = get_lnLR(LL, inits)
            -2*lnLR(μ)
        else
            0.0
        end
    end
end

function get_teststat(LL, inits, ::Type{T_qmutilde})
    _, θ0 = free_maximize(LL, inits)
    lnLRtilde = get_lnLRtilde(LL, inits)
    μ_hat = θ0[1]
    function qmutilde(μ)
        if μ_hat <= μ
            -2*lnLRtilde(μ)
        else
            0.0
        end
    end
end

"""
    asimovdata(model::PyHFModel, μ)

Generate the Asimov dataset and asimov priors, which is the expected counts after fixing POI to `μ` and optimize the
nuisance parameters.
"""
function asimovdata(model::PyHFModel, μ)
    LL = pyhf_logjointof(model)
    nuisance_inits = inits(model)[2:end]
    _, θs = cond_maximize(LL, μ, nuisance_inits)
    asimov_params = vcat(μ, θs)
    ps = priors(model)
    new_priors = NamedTuple{keys(ps)}(map(asimovprior, ps, asimov_params))
    expected(model, asimov_params), new_priors
end

"""
    AsimovModel(model::PyHFModel, μ)::PyHFModel

Generate the Asimov model when fixing `μ` (POI) to a value. Notice this changes the `priors` and `observed` compare
to the original `model`.
"""
function AsimovModel(model::PyHFModel, μ)
    A_data, A_priors = asimovdata(model, μ)
    PyHFModel(expected(model), A_data, A_priors, prior_names(model), inits(model))
end

asimovprior(dist::Normal, θ) = Normal(θ, dist.σ)
asimovprior(dist::FlatPrior, θ) = dist

function T_qmu(model::PyHFModel, qmuA_f)
    LL0 = pyhf_logjointof(model)
    qmu_f = get_teststat(LL0, inits(model), T_qmu)

    μ -> sqrt(qmu_f(μ)) - sqrt(qmuA_f(μ))
end

function T_qmu(model::PyHFModel)
    A_model = AsimovModel(model, 0.0)
    A_LL = pyhf_logjointof(A_model)
    qmuA_f = get_teststat(A_LL, inits(A_model), T_qmu)
    TS_q(model, qmuA_f)
end

function T_q0(model::PyHFModel, q0A_f)
    LL0 = pyhf_logjointof(model)
    q0_f = get_teststat(LL0, inits(model), T_q0)

    function (μ=0)
        return sqrt(q0_f(μ)) - sqrt(q0A_f(μ))
    end
end

function T_q0(model::PyHFModel)
    A_model = AsimovModel(model, 1.0)
    A_LL = pyhf_logjointof(A_model)
    q0A_f = get_q0(A_LL, inits(A_model))
    TS_q0(model, q0A_f)
end

function T_qmutilde(model::PyHFModel, qtildeA_f)
    LL0 = pyhf_logjointof(model)
    qtilde_f = get_teststat(LL0, inits(model), T_qmutilde)

    function (x)
        qmu = qtilde_f(x)
        qmu_A = qtildeA_f(x)
        sqrtqmu = sqrt(qmu)
        sqrtqmuA = sqrt(qmu_A)
        if sqrtqmu < sqrtqmuA
            sqrtqmu - sqrtqmuA
        else
            (qmu - qmu_A) / (2 * sqrtqmuA)
        end
    end
end

function T_qmutilde(model::PyHFModel)
    A_model = AsimovModel(model, 0.0)
    A_LL = pyhf_logjointof(A_model)
    qtildeA_f = get_teststat(A_LL, inits(A_model), T_qmutilde)
    TS_qtilde(model, qtildeA_f)
end
