using Unrolled
using SpecialFunctions: logabsgamma
using LogExpFunctions: xlogy

abstract type AbstractModifier end
function _prior end
function _init end

"""
Histosys is defined by two vectors represending bin counts
in `hi_data` and `lo_data`
"""
struct Histosys{T<:AbstractInterp} <: AbstractModifier
    interp::T
    function Histosys(interp::T) where T
        @assert T <: Union{InterpCode0, InterpCode4}
        new{T}(interp)
    end
end
Histosys(nominal, up, down) = Histosys(InterpCode0(nominal, up, down))
_prior(::Histosys) = Normal()
_init(::Histosys) = 0.0

struct Normsys{T<:AbstractInterp} <: AbstractModifier
    interp::T
end
_prior(::Normsys) = Normal()
_init(::Normsys) = 0.0

"""
Normsys is defined by two multiplicative scalars
"""
Normsys(up::Number, down::Number) = Normsys(InterpCode1(up, down))
Normsys(nominal, ups, downs) = Normsys(InterpCode1(nominal, ups, downs))

"""
Normfactor is unconstrained, so `interp` is just identity.
"""
struct Normfactor <: AbstractModifier # is unconstrained
    interp::typeof(identity)
    Normfactor() = new(identity)
end
_prior(::Normfactor) = FlatPrior(0, 5)
_init(::Normfactor) = 1.0

"""
Shapefactor is unconstrained, so `interp` is just identity. Unlike `Normfactor`,
this is per-bin
"""
struct Shapefactor{T} <: AbstractModifier # is unconstrained
    interp::T
    function Shapefactor(nbins, nthbin)
        f = binidentity(nbins, nthbin)
        new{typeof(f)}(f)
    end
end
_prior(::Shapefactor) = FlatPrior(0, 5)
_init(::Shapefactor) = 1.0

"""
Shapesys doesn't need interpolation, similar to `Staterror`
"""
struct Shapesys{T} <: AbstractModifier
    σn2::Float64
    interp::T
    function Shapesys(σ, nbins, nthbin)
        f = binidentity(nbins, nthbin)
        new{typeof(f)}(σ, f)
    end
end

"""
    RelaxedPoisson

Poisson with `logpdf` continuous in `k`. Essentially by replacing denominator with `gamma` function.

!!! warning
    The `Distributions.logpdf` has been redefined to be `logpdf(d::RelaxedPoisson, x) = logpdf(d, x*d.λ)`.
    This is to reproduce the Poisson constraint term in `pyhf`, which is a hack introduced for Asimov dataset.
"""
struct RelaxedPoisson{T} <: Distributions.ContinuousUnivariateDistribution
    λ::T
end
_relaxedpoislogpdf(d::RelaxedPoisson, x) = xlogy(x, d.λ) - d.λ - logabsgamma(x + 1.0)[1]
Distributions.logpdf(d::RelaxedPoisson, x) = _relaxedpoislogpdf(d, x*d.λ)
_prior(S::Shapesys) = RelaxedPoisson(S.σn2)
_init(S::Shapesys) = 1.0


"""
Staterror doesn't need interpolation, but it's a per-bin modifier.
Information regarding which bin is the target is recorded in `bintwoidentity`.

The `δ` is the absolute yield uncertainty in each bin, and the relative uncertainty:
`δ` / nominal is taken to be the `σ` of the prior, i.e. `α ~ Normal(1, δ/nominal)`
"""
struct Staterror{T} <: AbstractModifier
    σ::Float64
    interp::T
    function Staterror(σ, nbins, nthbin)
        f = binidentity(nbins, nthbin)
        new{typeof(f)}(σ, f)
    end
end
_prior(S::Staterror) = Normal(1, S.σ)
_init(S::Staterror) = 1.0

"""
Luminosity doesn't need interpolation, σ is provided at modifier construction time.
In `pyhf` JSON, this  information lives in the "Measurement" section, usually near the end of
the JSON file.
"""
struct Lumi <: AbstractModifier
    σ::Float64
    interp::typeof(identity)
    Lumi(σ) = new(σ, identity)
end
_prior(l::Lumi) = Normal(1, l.σ)
_init(l::Lumi) = 1.0

"""
    struct ExpCounts{T, M} #M is a long Tuple for unrolling purpose.
        nominal::T
        modifier_names::Vector{Symbol}
        modifiers::M
    end

A callable struct that returns the expected count given modifier nuisance parameter values.
The # of parameters passed must equal to length of `modifiers`. See [_expkernel](@ref)
"""
struct ExpCounts{T<:Number, M}
    nominal::Vector{T}
    modifier_names::Vector{Symbol}
    modifiers::M
end

# stop crazy stracktrace
function Base.show(io::IO, ::Type{<:ExpCounts}) 
    println(io, "ExpCounts{}")
end

function exp_mod!(modifier::Histosys, additive, factor, α)
    additive .+= modifier.interp(α)
    nothing
end
function exp_mod!(modifier, additive, factor, α)
    factor .*= modifier.interp(α)
    nothing
end


ExpCounts(nominal::Vector{<:Number}, names::Vector{Symbol}, modifiers::AbstractVector) = ExpCounts(nominal, names, tuple(modifiers...))

"""
    _expkernel(modifiers, nominal, αs)

The `Unrolled.@unroll` kernel function that computs the expected counts.
"""
@unroll function _expkernel(modifiers, nominal, αs)
    T = eltype(αs)
    additive = T.(nominal)
    factor = ones(T, length(additive))
    @unroll for i in 1:length(modifiers)
        @inbounds exp_mod!(modifiers[i], additive, factor, αs[i])
    end
    return factor .* additive
end

function (E::ExpCounts)(αs)
    (; modifiers, nominal) = E

    @assert length(modifiers) == length(αs)
    return _expkernel(modifiers, nominal, αs)
end

for T in (Normsys, Normfactor, Histosys, Staterror, Lumi, Shapefactor)
    @eval function Base.show(io::IO, E::$T)
        interp = Base.typename(typeof(E.interp)).name
        print(io, $T, "{$interp}")
    end
end

function Base.show(io::IO, E::ExpCounts)
    modifiers = E.modifiers
    elip = length(modifiers) > 5 ? "...\n" : ""
    println(io, "ExpCounts with $(length(modifiers)) modifiers:")
end
