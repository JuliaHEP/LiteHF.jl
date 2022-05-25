module LiteHF

using Distributions
import Random, Optim

# interpolations
# export InterpCode0, InterpCode1, InterpCode2, InterpCode4

# modifiers
export ExpCounts
export Normsys, Histosys, Normfactor, Lumi, Staterror, nmodifiers

# pyhf
export pyhf_loglikelihoodof, pyhf_logpriorof, pyhf_logjointof
export load_pyhfjson, build_pyhf, free_maximize, cond_maximize, PyHFModel,
       AsimovModel, inits, observed, expected, priors, prior_names

# Test Statistics & Test Statistics Distributions
export T_tmu, T_tmutilde, T_q0, T_qmu, T_qmutilde
export asymptotic_dists, expected_pvalue


include("./interpolations.jl")
include("./modifiers.jl")
include("./pyhfparser.jl")

abstract type AbstractModel end
include("./pyhfmodel.jl")
include("./teststatistics.jl")
include("./asymptotictests.jl")

function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    r = load_pyhfjson(joinpath(@__DIR__, "../test/pyhfjson/sample_lumi.json"))
    M = build_pyhf(r)
end

_precompile_()

end
