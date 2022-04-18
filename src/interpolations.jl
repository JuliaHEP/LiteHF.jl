abstract type AbstractInterp end

struct InterpCode0{T} <: AbstractInterp
    I0::T
    I_up::T
    I_down::T
end

function (i::InterpCode0)(α)
    if α >= 0
        @. α * (i.I_up - i.I0)
    else
        @. α * (i.I0 - i.I_down)
    end
end

struct InterpCode1{T} <: AbstractInterp
    I0::T
    f_up::Float64
    f_down::Float64
end

function InterpCode1(I0, I_up::T, I_down::T) where {T<:AbstractVector}
    f_up = first(I_up / I0)
    f_down = first(I_down / I0)
    InterpCode1(I0, f_up, f_down)
end

function (i::InterpCode1)(α)
    if α >= 0
        (i.f_up)^α
    else
        (i.f_down)^(-α)
    end
end
