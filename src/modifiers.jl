using Unrolled

abstract type AbstractModifier end

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

Normsys(up::Number, down::Number) = Normsys(InterpCode1(up, down))
function Normsys(nominal, ups, downs) 
    Normsys(InterpCode1(nominal, f_up, f_down))
end

struct Normfactor <: AbstractModifier # is unconstrained
    interp::typeof(identity)
    Normfactor() = new(identity)
end

struct ExpCounts{T, M}
    nominal::T
    modifiers::M
end

ExpCounts(nominal, modifiers::AbstractVector) = ExpCounts(nominal, tuple(modifiers...))

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
    @assert length(αs) == length(modifiers)

    res = prod(_expkernel(modifiers, nominal, αs))

    return res
end
