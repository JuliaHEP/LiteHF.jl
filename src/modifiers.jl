using Unrolled

abstract type AbstractModifier end

"""
    Histosys is defined by two vectors represending bin counts
    in `hi_` and `lo_data`
"""
struct Histosys{T<:AbstractInterp} <: AbstractModifier
    interp::T
    function Histosys(interp::T) where T
        @assert T <: Union{InterpCode0, InterpCode4}
        new{T}(interp)
    end
end
Histosys(up, down) = Histosys(InterpCode4(up, down))

struct Normsys{T<:AbstractInterp} <: AbstractModifier
    interp::T
end

"""
    Normfactor is defined by two multiplicative scalars
"""
Normsys(up::Number, down::Number) = Normsys(InterpCode1(up, down))
function Normsys(nominal, ups, downs) 
    Normsys(InterpCode1(nominal, f_up, f_down))
end

"""
    Normfactor is unconstrained, so `interp` is always `identity()`
"""
struct Normfactor <: AbstractModifier # is unconstrained
    interp::typeof(identity)
    Normfactor() = new(identity)
end

struct ExpCounts{T, M}
    nominal::T
    modifiers::M
end

ExpCounts(nominal, modifiers::AbstractVector) = ExpCounts(nominal, tuple(modifiers...))

nmodifiers(E::ExpCounts) = length(E.modifiers)

@unroll function _expkernel(modifiers, nominal, αs)
    additive = float(nominal)
    factor = 1.0
    @unroll for i in 1:length(modifiers)
        @inbounds modifier = modifiers[i]
        @inbounds α = αs[i]
        if modifier isa Histosys
            # additive
            additive += modifier.interp(nominal, α)
        else
            # multiplicative
            factor *= modifier.interp(α)
        end
    end
    return (additive, factor)
end

function (E::ExpCounts)(αs)
    (; modifiers, nominal) = E

    res = prod(_expkernel(modifiers, nominal, αs))

    return res
end

for T in (Normsys, Normfactor, Histosys)
    @eval function Base.show(io::IO, E::$T)
        interp = Base.typename(typeof(E.interp)).name
        print(io, $T, "{$interp}")
    end
end

function Base.show(io::IO, E::ExpCounts)
    modifiers = E.modifiers
    elip = length(modifiers) > 5 ? "..." : ""
    println(io, "ExpCounts with $(length(modifiers)) modifiers:")
    println(io, join(first(modifiers, 5), ", "), elip)
end
