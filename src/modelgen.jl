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
    build_pyhf(load_pyhfjson(path)) 
        -> expected::Function, priors::NamedTupleDist, priornames::Tuple{Symbol}

    the `expected(αs)` is a function that takes vector or tuple of length `N`, where `N` is also the
    length of `priors` and `priornames`. In other words, the three objects returned
    are aligned.

    The output of `forward_model()` is always a vector of length `N`, represending the expected
    bin counts when all parameters (`αs`) taking a set of specific values.

"""
function build_pyhf(pyhfmodel)
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
    expected = let Es = all_expcounts, Vs = v_idxs
        αs -> internal_expected(Es, Vs, αs)
    end

    return expected, priors, priornames
end

"""
    loglikelihoodof(expected, obs)
    Return a callable Function that would calculate the log likelihood

    !!!Note
    The "constraint" terms that come from prior is NOT included here.
"""
function loglikelihoodof(expected, obs)
    f(x, o) = logpdf(Poisson(x), o)
    L = let data = obs
        αs -> begin 
            expe = expected(αs)
            any(<(0), expe) && return -Inf
            measurement = mapreduce(f, +, expe, obs)

            return measurement
        end
    end
    return L
end

"""
    loglikelihoodof(expected, obs, priors)
    Return a callable Function that would calculate the log likelihood

    !!!Note
    The "constraint" terms that come from prior IS included here.
"""
function loglikelihoodof(expected, obs, priors)
    f(x, o) = logpdf(Poisson(x), o)
    L = let data = obs, pris = values(priors)
        αs -> begin 
            expe = expected(αs)
            any(<(0), expe) && return -Inf
            measurement = mapreduce(f, +, expe, obs)
            constraint = mapreduce(logpdf, +, pris, αs)

            return measurement + constraint
        end
    end
    return L
end
