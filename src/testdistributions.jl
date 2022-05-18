abstract type AbstractTestStatistics <: Distributions.ContinuousUnivariateDistribution end
const ATS = AbstractTestStatistics

struct TS_q0 <: ATS end
struct TS_q <: ATS end
struct TS_qtilde <: ATS end

teststatistics(T::ATS) = error("Not implemented $T.")

function TS_q(model::PyHFModel, qmuA_f)
    LL0 = pyhf_logjointof(model)
    qmu_f = get_qmu(LL0, model.inits)

    μ -> sqrt(qmu_f(μ)) - sqrt(qmuA_f(μ))
end
function TS_q(model::PyHFModel)
    A_model = AsimovModel(model, 0.0)
    A_LL = pyhf_logjointof(A_model)
    qmuA_f = get_qmu(A_LL, A_model.inits)
    TS_q(model, qmuA_f)
end

function TS_q0(model::PyHFModel, q0A_f)
    LL0 = pyhf_logjointof(model)
    q0_f = get_q0(LL0, model.inits)

    function (μ=0)
        return sqrt(q0_f(μ)) - sqrt(q0A_f(μ))
    end
end
function TS_q0(model::PyHFModel)
    A_model = AsimovModel(model, 1.0)
    A_LL = pyhf_logjointof(A_model)
    q0A_f = get_q0(A_LL, A_model.inits)
    TS_q0(model, q0A_f)
end

function TS_qtilde(model::PyHFModel, qtildeA_f)
    LL0 = pyhf_logjointof(model)
    qtilde_f = get_qmutilde(LL0, model.inits)

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
function TS_qtilde(model::PyHFModel)
    A_model = AsimovModel(model, 0.0)
    A_LL = pyhf_logjointof(A_model)
    qtildeA_f = get_qmutilde(A_LL, A_model.inits)
    TS_qtilde(model, qtildeA_f)
end
