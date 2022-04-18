using JSON3

const _modifier_dict = Dict(
                            "normfactor" => Normfactor,
                            "histosys" => Histosys,
                            "normsys" => Normsys,
                           )


function hilo_data(jobj)
    jobj[:hi_data], jobj[:lo_data]
end
function hilo_factor(jobj)
    jobj[:hi], jobj[:lo]
end

"""
    build_modifier(rawjdict[:channels][1][:samples][2][:modifiers][1]) =>
    <:AbstractModifier
"""
function build_modifier(jobj)
    build_modifier(jobj[:data], _modifier_dict[jobj[:type]])
end

"""
    build_modifier(...[:modifiers][1][:data], Type) =>
    <:AbstractModifier
"""
function build_modifier(jobj, modifier_type::Type{T}) where T
    if T == Histosys
        T(hilo_data(jobj)...)
    elseif T == Normsys
        T(hilo_factor(jobj)...)
    elseif T == Normfactor
        T()
    end
end

"""
    build_sample(rawjdict[:channels][1][:samples][2]) =>
    Pair{String, ExpCounts}
"""
function build_sample(jobj)
    jobj[:name] => ExpCounts(jobj[:data], build_modifier.(jobj[:modifiers]))
end

"""
    build_channel(rawjdict[:channels][1][:samples][2]) =>
    Dict{String, ExpCounts}
"""
function build_channel(jobj)
    Dict(build_sample.(jobj[:samples]))
end

"""
    load_pyhfjson(path)
"""
function load_pyhfjson(path)
    jobj = JSON3.read(read(path))
    Dict(obj[:name] => build_channel(obj) for obj in jobj[:channels])
end
