using JSON3

const _modifier_dict = Dict(
                            "normfactor" => Normfactor,
                            "histosys" => Histosys,
                            "normsys" => Normsys,
                            "shapesys" => Shapesys,
                            "staterror" => Staterror,
                            "lumi" => Lumi,
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
function build_modifier!(jobj, names; misc)
    mod = build_modifier(jobj, _modifier_dict[jobj[:type]]; misc)
    mod_name = Symbol(jobj[:name])
    if mod isa Vector
        # for stuff like Staterror, which inflates to multiple `γ`
        for i in eachindex(mod)
            push!(names, Symbol(mod_name, "_bin$i"))
        end
    else
        push!(names, mod_name)
    end
    return mod
end

"""
    build_modifier(...[:modifiers][1][:data], Type) =>
    <:AbstractModifier
"""
function build_modifier(modobj, modifier_type::Type{T}; misc) where T
    if T == Histosys
        T(hilo_data(modobj[:data])...)
    elseif T == Normsys
        T(hilo_factor(modobj[:data])...)
    elseif T == Staterror
        T.(modobj[:data])
    elseif T == Lumi
        paras = misc[:measurements][1][:config][:parameters]
        lumi_idx = findfirst(x->x[:name] == modobj[:name], paras)
        σ = only(paras[lumi_idx][:sigmas])
        T(σ)
    #FIXME: how does it work???
    # elseif T == Shapesys
    #     T.(dataobj)
    else
        T()
    end
end

"""
    build_sample(rawjdict[:channels][1][:samples][2]) =>
    Pair{String, ExpCounts}
"""
function build_sample(jobj, names=Symbol[]; misc)
    modifiers = build_modifier!.(jobj[:modifiers], Ref(names); misc)
    modifiers = any(x->x <: Vector, typeof.(modifiers)) ? reduce(vcat, modifiers) : modifiers #flatten it
    jobj[:name] => ExpCounts(jobj[:data], names, modifiers)
end

"""
    build_channel(rawjdict[:channels][1][:samples][2]) =>
    Dict{String, ExpCounts}
"""
function build_channel(jobj; misc)
    Dict(build_sample.(jobj[:samples]; misc))
end

"""
    load_pyhfjson(path)
"""
function load_pyhfjson(path)
    jobj = JSON3.read(read(path))
    mes = get(jobj, :measurements, Dict())
    obs = get(jobj, :observations, Dict())
    misc = Dict(:measurements => mes, :observations => obs)

    res = Dict{String, Any}(obj[:name] => build_channel(obj; misc) for obj in jobj[:channels])
    res["misc"] = misc
    res
end
