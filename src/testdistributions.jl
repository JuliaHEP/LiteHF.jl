abstract type AbstractTestStatistics <: Distributions.ContinuousUnivariateDistribution end
const ATS = AbstractTestStatistics

struct TS_t0 <: ATS end
struct TS_tmu <: ATS end
struct TS_tmutilde <: ATS end
struct TS_q0 <: ATS end
struct TS_qmu <: ATS end
struct TS_qmutilde <: ATS end

teststatistics(T::ATS) = error("Not implemented $T.")

function TS_qmu(model)
    LL0 = pyhf_logjointof(model)
    qmu_f = get_qmu(LL0, model.inits)

    A_data, A_mubhathat = asimovdata(model, 0.0)
    A_LL = pyhf_logjointof(model.expected, A_data, model.priors)
    qmuA_f = get_qmu(A_LL, model.inits)

    μ -> sqrt(qmu_f(μ)) - sqrt(qmuA_f(μ))
end

function TS_q0(model)
    LL0 = pyhf_logjointof(model)
    q0_f = get_q0(LL0, model.inits)

    A_data, A_mubhathat = asimovdata(model, 1.0)
    A_LL = pyhf_logjointof(model.expected, A_data, model.priors)
    q0A_f = get_q0(A_LL, model.inits)

    function (μ=0)
        return sqrt(q0_f(μ)) - sqrt(q0A_f(μ))
    end
end

function TS_qmutilde(model)
    LL0 = pyhf_logjointof(model)
    qtilde_f = get_qmutilde(LL0, model.inits)

    A_data, A_mubhathat = asimovdata(model, 0.0)
    A_LL = pyhf_logjointof(model.expected, A_data, model.priors)
    qtildeA_f = get_qmutilde(A_LL, model.inits)

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
