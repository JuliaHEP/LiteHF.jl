abstract type AbstractModifier end

struct Histosys{T<:AbstractInterp} <: AbstractModifier
    interp::T
end
Histosys(nominal, up, down) = Histosys(InterpCode0(nominal, up, down))

struct Normsys{T<:AbstractInterp} <: AbstractModifier
    interp::T
end

Normsys(nominal, up::Number, down::Number) = Normsys(InterpCode1(nominal, up, down))
function Normsys(nominal, ups, downs) 
    @assert length(nominal) == length(ups) == length(downs)
    f_up = first(ups) / first(nominal)
    f_down = first(downs) / first(nominal)
    Normsys(InterpCode1(nominal, f_up, f_down))
end


struct Normfactor <: AbstractModifier # is unconstrained
    interp::typeof(identity)
    Normfactor() = new(identity)
end

struct ExpCounts{T<:Number}
    nominal::Vector{T}
    modifiers::Vector{<:AbstractModifier}
end

function (E::ExpCounts)(αs...)
    @assert length(αs) == length(E.modifiers)
    res = E.nominal
    multi = 1.0

    for (m, α) in zip(E.modifiers, αs)
        if m isa Histosys
        # additive
            res = res .+ m.interp(α)
        else
        # multiplicative
            multi *= m.interp(α)
        end
    end

    return multi * res
end
