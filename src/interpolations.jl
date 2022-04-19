abstract type AbstractInterp end

struct InterpCode0{T} <: AbstractInterp
    I_up::T
    I_down::T
end

function (i::InterpCode0)(I0, α)
    vs = if α >= 0
        (i.I_up - I0)
    else
        (I0 - i.I_down)
    end
    α * vs
end

struct InterpCode1 <: AbstractInterp
    f_up::Float64
    f_down::Float64
end

function InterpCode1(I0, I_up::T, I_down::T) where {T<:AbstractVector}
    f_up = first(I_up / I0)
    f_down = first(I_down / I0)
    InterpCode1(f_up, f_down)
end

function (i::InterpCode1)(I0, α)
    if α >= 0
        (i.f_up)^α
    else
        (i.f_down)^(-α)
    end
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
    if α >= 1
        @. (b + 2*a) * (α - 1)
    elseif α >= -1
        @. a * α^2 + b * α
    else
        @. (b - 2*a) * (α + 1)
    end
end


struct InterpCode4{T<:AbstractVector, N<:Number} <: AbstractInterp
    I_up::T
    I_down::T
    α0::N
    inver::Matrix{Float64}
end

function InterpCode4(I_up, I_down; α0=1)
    inver = _interp4_inverse(α0)
    InterpCode4(I_up, I_down, α0, inver)
end

function (i::InterpCode4)(I0, α)
    (; I_up, I_down, inver, α0) = i
    delta_up = @. I_up / I0
    delta_down = @. I_down / I0
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
    @. I0 * (mult - 1)
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
