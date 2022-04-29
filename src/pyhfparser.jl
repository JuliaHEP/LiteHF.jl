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
function build_modifier!(jobj, names; misc, mcstats)
    mod = build_modifier(jobj, _modifier_dict[jobj[:type]]; misc, mcstats)
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
function build_modifier(modobj, modifier_type::Type{T}; misc, mcstats) where T
    modname = modobj[:name]
    moddata = modobj[:data]
    if T == Histosys
        T(hilo_data(moddata)...)
    elseif T == Normsys
        T(hilo_factor(moddata)...)
    elseif T == Staterror
        # each Staterror keepds track of which bin it should modifier
        nominalsums, sumδ2 = mcstats[modname]
        T.(sqrt.(sumδ2) ./ nominalsums, eachindex(moddata))
    elseif T == Lumi
        paras = misc[:measurements][1][:config][:parameters]
        lumi_idx = findfirst(x->x[:name] == modname, paras)
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
    ExpCounts
"""
function build_sample(jobj, names=Symbol[]; misc, mcstats)
    modifiers = build_modifier!.(jobj[:modifiers], Ref(names); misc, mcstats)
    modifiers = any(x->x <: Vector, typeof.(modifiers)) ? reduce(vcat, modifiers) : modifiers #flatten it
    @assert length(names) == length(modifiers)
    ExpCounts(jobj[:data], names, modifiers)
end

"""
    build_channel(rawjdict[:channels][1][:samples][2]) =>
    Dict{String, ExpCounts}
"""
function build_channel(jobj; misc)
    mcstats = Dict()
    # accumulate MC stats related quantities
    for sample in jobj[:samples], m in sample[:modifiers]
        m[:type] != "staterror" && continue
        modname = m[:name]
        if haskey(mcstats, modname)
            mcstats[modname] = mcstats[modname] .+ (sample[:data], m[:data] .^ 2)
        else
            mcstats[modname] = (sample[:data], m[:data] .^ 2)
        end
    end

    # build modifiers
    res = Dict()
    for sample in jobj[:samples]
        res[sample[:name]] = build_sample(sample; misc, mcstats)
    end
    res
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
