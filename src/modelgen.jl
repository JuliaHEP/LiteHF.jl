using ValueShapes

@generated function internal_expected(Es, Vs, αs)
    @assert Es <: Tuple
    expand(i) = i == 1 ? :(Es[1](αs[Vs[1]])) : :(+(Es[$i](αs[Vs[$i]]), $(expand(i-1))))
    return expand(length(Es.parameters))
end

function _idxspartition(counts)
    temp = UnitRange[]
    C = 1
    for N in counts
        push!(temp, C:C+N-1)
        C+=N
    end
    Tuple(temp)
end

"""
    build_pyhf_spec(load_pyhfjson(path)) 
        -> forward_model::Function, priors::NamedTupleDist, priornames::Tuple{Symbol}

    the `foward_model(αs)` is a function that takes vector or tuple of length `N`, where `N` is also the
    length of `priors` and `priornames`. In other words, the three objects returned
    are aligned.

    The output of `forward_model()` is always a `Distributions.Product{}` of Poissons(each bin).

"""
function build_pyhf_spec(pyhfmodel)
    #### all_* are before de-duplicating
    all_expcounts = Tuple(
                     sample[2]
                     for channel in pyhfmodel for sample in channel[2]
                    )
    all_names = reduce(vcat, 
                       E.modifier_names for E in all_expcounts
                      )
    all_modifiers = mapreduce(collect, vcat, 
                              E.modifiers for E in all_expcounts
                             )
    lookup = Dict(all_names .=> all_modifiers)
    unique_names = unique(all_names)
    unique_modifiers = [lookup[k] for k in unique_names]

    counts = nmodifiers.(all_expcounts)
    v_idxs = _idxspartition(counts)
    priornames = Tuple(Symbol.(unique_names))
    priors = NamedTupleDist(NamedTuple{priornames}(_prior.(unique_modifiers)))

    forward_model = let Es = all_expcounts, Vs = v_idxs
        αs -> product_distribution(Poisson.(internal_expected(Es, Vs, αs)))
    end

    return forward_model, priors, priornames

#     @model function refmodel(bincounts)
#         αs ~ arraydist(priors)

#         exp = internal_expected(all_expcounts, v_idxs, αs)

#         if any(<(0), exp)
#             Turing.@addlogprob! -Inf
#             return
#         end
#         @. bincounts ~ Poisson(exp)
#     end

#     return unique_names, refmodel
end
