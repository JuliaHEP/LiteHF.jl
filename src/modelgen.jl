using ValueShapes

struct PyHFModel
    expected
    priors
    prior_names
    prior_inits::Vector{Float64}
    LogLikelihood
end

function Base.show(io::IO, P::PyHFModel)
    Nprior = length(P.prior_names)
    print(io, "PyHFModel with $(Nprior) nuisance parameters.")
end

@generated function internal_expected(Es, Vs, αs)
    @assert Es <: Tuple
    @views expand(i) = i == 1 ? :(Es[1](αs[Vs[1]])) : :(+(Es[$i](αs[Vs[$i]]), $(expand(i-1))))
    return expand(length(Es.parameters))
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
                     for (name, channel) in pyhfmodel for sample in channel if name!="misc"
                    )
    all_v_names = [E.modifier_names for E in all_expcounts]
    all_names = reduce(vcat, all_v_names)
    all_modifiers = mapreduce(collect, vcat, 
                              E.modifiers for E in all_expcounts
                             )
    lookup = Dict(all_names .=> all_modifiers)
    unique_names = unique(all_names)
    input_modifiers = [lookup[k] for k in unique_names]
    priornames = Tuple(Symbol.(unique_names))
    priors = NamedTupleDist(NamedTuple{priornames}(_prior.(input_modifiers)))
    inits = Vector{Float64}(_init.(input_modifiers))
    obs = pyhfmodel["misc"][:observations][1][:data]

    # Special case: same name can appear multiple times with different modifier type
    
    # if masks[1] == [1,2,4] that means the first `ExpCounts(αs[[1,2,4]])`
    masks = Tuple([findfirst(==(i), unique_names) for i in names] for names in all_v_names)
    counts = nmodifiers.(all_expcounts)
    # each mask should have enough parameter to feed the ExpCount
    @assert all(length.(masks) .== counts)

    expected = let Es = all_expcounts, Vs = masks
        αs -> internal_expected(Es, Vs, αs)
    end

    LL = pyhf_loglikelihoodof(expected, obs, priors)

    return PyHFModel(expected, priors, priornames, inits, LL)
end

"""
    pyhf_loglikelihoodof(expected, obs)
    Return a callable Function that would calculate the log likelihood

    !!!Note
    The "constraint" terms that come from prior is NOT included here.
"""
function pyhf_loglikelihoodof(expected, obs)
    f(x, o) = logpdf(Poisson(x), o)
    L = let data = obs
        αs -> begin 
            expe = expected(αs)
            any(<(0), expe) && return -Inf
            measurement = mapreduce(f, + , expe, obs)

            return measurement
        end
    end
    return L
end

@generated function internal_constrainteval(pris, αs)
    @assert pris <: Tuple
    @views expand(i) = i == 1 ? :(logpdf(pris[1], αs[1])) : 
    :(+(logpdf(pris[$i], αs[$i]), $(expand(i-1))))
    return expand(length(pris.parameters))
end
"""
    pyhf_loglikelihoodof(expected, obs, priors)
    Return a callable Function that would calculate the log likelihood

    !!!Note
    The "constraint" terms that come from prior IS included here.
"""
function pyhf_loglikelihoodof(expected, obs, priors)
    f(x, o) = logpdf(Poisson(x), o)
    L = let data = obs, pris = values(priors)
        αs -> begin
            expe = expected(αs)
            any(<(0), expe) && return -Inf
            measurement = mapreduce(f, +, expe, obs)
            constraint = internal_constrainteval(pris, αs)
            return measurement + constraint
        end
    end
    return L
end
