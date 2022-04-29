module LiteHF

using Distributions

struct FlatPrior{T} <: ContinuousUnivariateDistribution
    a::T
    b::T
end

Distributions.logpdf(d::FlatPrior, x::Real) = zero(x)

export pyhf_loglikelihoodof

# interpolations
# export InterpCode0, InterpCode1, InterpCode2, InterpCode4

# modifiers
export Normsys, Histosys, Normfactor, Lumi, Staterror, nmodifiers

# pyhf json parsing
export build_channel, build_sample, load_pyhfjson, build_pyhf

export ExpCounts

# Write your package code here.
include("./interpolations.jl")
include("./modifiers.jl")
include("./pyhfparser.jl")
include("./modelgen.jl")

end
