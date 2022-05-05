module LiteHF

using Distributions
import Random, Optim

export pyhf_loglikelihoodof, pyhf_logpriorof, pyhf_logjointof

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
include("./teststatistics.jl")

function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    r = load_pyhfjson(joinpath(@__DIR__, "../test/pyhfjson/sample_lumi.json"))
    M = build_pyhf(r)
    M
end

_precompile_()

end
