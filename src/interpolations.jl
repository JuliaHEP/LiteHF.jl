abstract type AbstractInterp end

"""
    InterpCode0{T}

Callable struct for interpolation for additive modifier.
Code0 is the two-piece linear interpolation.
"""
struct InterpCode0{T} <: AbstractInterp
    Δ_up::T
    Δ_down::T
end

function InterpCode0(I0, I_up::T, I_down::T) where {T<:AbstractVector}
    Δ_up = I_up - I0
    Δ_down = I0 - I_down
    InterpCode0(Δ_up, Δ_down)
end

function (i::InterpCode0)(α)
    vs = Base.ifelse(α >= zero(α), i.Δ_up, i.Δ_down)
    return vs*α
end

"""
    InterpCode1{T}

Callable struct for interpolation for multiplicative modifier.
Code1 is the exponential interpolation.
"""
struct InterpCode1 <: AbstractInterp
    f_up::Float64
    f_down::Float64
end

function InterpCode1(I0, I_up::T, I_down::T) where {T<:AbstractVector}
    f_up = first(I_up ./ I0)
    f_down = first(I_down ./ I0)
    InterpCode1(f_up, f_down)
end

function (i::InterpCode1)(α)
    Base.ifelse(α >= zero(α), (i.f_up)^α, (i.f_down)^(-α))
end

struct InterpCode2{T} <: AbstractInterp
    a::T
    b::T
end

function InterpCode2(I0, I_up::T, I_down::T) where {T<:AbstractVector}
    a = @. 0.5 * (I_up + I_down) - I0
    b = @. 0.5 * (I_up - I_down)
    InterpCode2(a, b)
end

function (i::InterpCode2)(α)
    (; a, b) = i
    if α > 1
        @. (b + 2*a) * (α - 1)
    elseif α >= -1
        @. a * α^2 + b * α
    else
        @. (b - 2*a) * (α + 1)
    end
end


"""
    InterpCode4{T}

Callable struct for interpolation for additive modifier.
Code4 is the exponential + 6-order polynomial interpolation.
"""
struct InterpCode4{T<:AbstractVector, N<:Number} <: AbstractInterp
    f_ups::T
    f_downs::T
    α0::N
    inver::Matrix{Float64}
end

function InterpCode4(I0, I_up, I_down; α0=1)
    inver = _interp4_inverse(α0)
    InterpCode4(I_up ./ I0, I_down ./ I0, α0, inver)
end

function (i::InterpCode4)(α)
    (; f_ups, f_downs, inver, α0) = i
    delta_up = f_ups
    delta_down = f_downs
    mult = if α >= α0
        @. (delta_up) ^ α
    elseif α <= -α0
        @. (delta_down) ^ (-α)
    else
        delta_up_alpha0 = @. delta_up^α0
        delta_down_alpha0 = @. delta_down^α0
        b = @. [
             delta_up_alpha0 - 1,
             delta_down_alpha0 - 1,
             log(delta_up) * delta_up_alpha0,
             -log(delta_down) * delta_down_alpha0,
             log(delta_up)^2 * delta_up_alpha0,
             log(delta_down)^2 * delta_down_alpha0,
            ]
        coefficients = inver * b
        1 .+ sum(coefficients[i] * α^i for i=1:6)
    end
    mult
end

function _interp4_inverse(α0)
    alpha0 = α0
    [15/(16*alpha0)  -15/(16*alpha0)  -7/16            -7/16            1/16*alpha0      -1/16*alpha0
     3/(2*alpha0^2)  3/(2*alpha0^2)   -9/(16*alpha0)   9/(16*alpha0)    1/16             1/16
     -5/(8*alpha0^3) 5/(8*alpha0^3)   5/(8*alpha0^2)   5/(8*alpha0^2)   -1/(8*alpha0)    1/(8*alpha0)
     3/(-2*alpha0^4) 3/(-2*alpha0^4)  -7/(-8*alpha0^3) 7/(-8*alpha0^3)  -1/(8*alpha0^2)  -1/(8*alpha0^2)
     3/(16*alpha0^5) -3/(16*alpha0^5) -3/(16*alpha0^4) -3/(16*alpha0^4) 1/(16*alpha0^3)  -1/(16*alpha0^3)
     1/(2*alpha0^6)  1/(2*alpha0^6)   -5/(16*alpha0^5) 5/(16*alpha0^5)  1/(16*alpha0^4)  1/(16*alpha0^4) ]
end


"""
    MultOneHot{T} <: AbstractVector{T}

Internal type used to avoid allocation for per-bin multiplicative systematics. It behaves as a vector with length `nbins` and 
only has value `α` on `nthbin`-th index, the rest being `one(T)`. See also [binidentity](@ref).
"""
struct MultOneHot{T}<:AbstractVector{T}
    nbins::Int
    nthbin::Int
    α::T
end
Base.length(b::MultOneHot) = b.nbins
Base.getindex(b::MultOneHot{T}, n::Integer) where T = Base.ifelse(n==b.nthbin, b.α, one(T))
Base.size(b::MultOneHot) = (b.nbins, )

"""
    binidentity(nbins, nthbin)

A functional that used to track per-bin systematics. Returns the closure function over `nbins, nthbin`:
```julia
    α -> MultOneHot(nbins, nthbin, α)
```
"""
function binidentity(nbins, nthbin)
    α -> MultOneHot(nbins, nthbin, α)
end

"""
    Pseudo flat prior in the sense that `logpdf()` always evaluates to zero,
    but `rand()`, `minimum()`, and `maximum()` behaves like `Uniform(a, b)`.
"""
struct FlatPrior{T} <: Distributions.ContinuousUnivariateDistribution
    a::T
    b::T
end

Base.minimum(d::FlatPrior) = d.a
Base.maximum(d::FlatPrior) = d.b
Distributions.logpdf(d::FlatPrior, x::Real) = zero(x)
Base.rand(rng::Random.AbstractRNG, d::FlatPrior) = rand(rng, Uniform(d.a, d.b))
