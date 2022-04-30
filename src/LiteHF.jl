module LiteHF

using Distributions
import Random

"""
    Pseudo flat prior in the sense that `logpdf()` always evaluates to zero,
    but `rand()`, `minimum()`, and `maximum()` behaves like `Uniform(a, b)`.
"""
struct FlatPrior{T} <: ContinuousUnivariateDistribution
    a::T
    b::T
end

Base.minimum(d::FlatPrior) = d.a
Base.maximum(d::FlatPrior) = d.b
Distributions.logpdf(d::FlatPrior, x::Real) = zero(x)
Base.rand(rng::Random.AbstractRNG, d::FlatPrior) = rand(rng, Uniform(d.a, d.b))

export pyhf_loglikelihoodof

# interpolations
# export InterpCode0, InterpCode1, InterpCode2, InterpCode4

# modifiers
export Normsys, Histosys, Normfactor, Lumi, Staterror, nmodifiers

# pyhf json parsing
export load_pyhfjson, build_pyhf

export ExpCounts

# Write your package code here.
include("./interpolations.jl")
include("./modifiers.jl")
include("./pyhfparser.jl")
include("./modelgen.jl")

function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    r = load_pyhfjson(joinpath(@__DIR__, "../test/pyhfjson/sample_lumi.json"))
    M = build_pyhf(r)
    M
end

_precompile_()

end
