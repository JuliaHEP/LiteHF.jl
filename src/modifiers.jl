abstract type AbstractModifier end

struct Histosys{T<:AbstractInterp} <: AbstractModifier
    interp::T
end
Histosys(up, down) = Histosys(InterpCode0(up, down))

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

struct ExpCounts{T<:Number, M}
    nominal::Vector{T}
    modifiers::M
end

function (E::ExpCounts)(αs...)
    @assert length(αs) == length(E.modifiers)
    res = E.nominal
    multi = 1.0

    for (m, α) in zip(E.modifiers, αs)
        if m isa Histosys
        # additive
            res += m.interp(E.nominal, α)
        else
        # multiplicative
            multi *= m.interp(α)
        end
    end

    return multi * res
end
