using ValueShapes

"""
    struct PyHFModel{E, L}
        expected::E
        priors
        prior_names
        prior_inits::Vector{Float64}
        LogLikelihood::L
    end

Struct for holding result from [build_pyhf](@ref).
"""
struct PyHFModel{E, L}
    expected::E
    priors
    prior_names
    prior_inits::Vector{Float64}
    LogLikelihood::L
end

function Base.show(io::IO, P::PyHFModel)
    Nprior = length(P.prior_names)
    print(io, "PyHFModel with $(Nprior) nuisance parameters.")
end

"""
    internal_expected(Es, Vs, αs)

The `@generated` function that computes expected counts in `PyHFModel.expected()` evaluation.
The `Vs::NTuple{N, Vector{Int64}}` has the same length as `Es::NTuple{N, ExpCounts}`.

In general `αs` is shorter than `Es` and `Vs` because a given nuisance parameter `α` may appear
in multiple sample / modifier.

!!! note
    If for example `Vs[1] = [1,3,4]`, it means that the first `ExpCount` in `Es` is evaluated with
    ```
    Es[1](@view αs[[1,3,4]])
    ```
    and so on.
"""
@generated function internal_expected(Es, Vs, αs)
    @assert Es <: Tuple
    @views expand(i) = i == 1 ? :(Es[1](αs[Vs[1]])) : :(+(Es[$i](αs[Vs[$i]]), $(expand(i-1))))
    return expand(length(Es.parameters))
end

function find_obs_data(name, observations)
    i = findfirst(x->x[:name] == name, observations)
    observations[i][:data]
end

"""
    build_pyhf(load_pyhfjson(path)) -> PyHFModel

the `expected(αs)` is a function that takes vector or tuple of length `N`, where `N` is also the
length of `priors` and `priornames`. In other words, these three fields of the returned object
are aligned.


!!! note
    The bins from different channels are put into a `NTuple{Nbins, Vector}`.
"""
function build_pyhf(pyhfmodel)
    channels = [name => channel for (name, channel) in pyhfmodel if name != "misc"]
    v_obs = Tuple(find_obs_data(name, pyhfmodel["misc"][:observations]) for (name, _) in channels)
    global_unique = reduce(vcat, [sample[2].modifier_names for (name, C) in channels for sample in C]) |>
    unique

    # intentional type-insability, avoid latency
    all_expected = []
    all_lookup = Dict()
    for c in channels
        exp, lk = build_pyhfchannel(c, global_unique)
        push!(all_expected, exp)
        merge!(all_lookup, lk)
    end

    input_modifiers = [all_lookup[k] for k in global_unique]
    priornames = Tuple(Symbol.(global_unique))
    priors = NamedTupleDist(NamedTuple{priornames}(_prior.(input_modifiers)))
    inits = Vector{Float64}(_init.(input_modifiers))

    total_expected = let Es = Tuple(all_expected)
        αs -> map(E->E(αs), Es)
    end

    LL = pyhf_logjointof(total_expected, v_obs, priors)
    return PyHFModel(total_expected, priors, priornames, inits, LL)
end

function build_pyhfchannel(channel, global_unique)
    #### this is within the channel, we use `global_unique` vector to align back
    name, samples = channel
    all_expcounts = Tuple(sample[2] for sample in samples)
    all_v_names = Any[E.modifier_names for E in all_expcounts]
    all_names = reduce(vcat, all_v_names)
    channel_unique = unique(all_names)
    all_modifiers = []
    for E in all_expcounts
        append!(all_modifiers, E.modifiers)
    end
    lookup = Dict(all_names .=> all_modifiers)

    # Special case: same name can appear multiple times with different modifier type
    # if masks[1] == [1,2,4] that means the first `ExpCounts(αs[[1,2,4]])`
    masks = Tuple([findfirst(==(i), global_unique) for i in names] for names in all_v_names)

    expected = let Es = all_expcounts, Vs = masks
        αs -> internal_expected(Es, Vs, αs)
    end
    return expected, lookup
end

"""
    pyhf_loglikelihoodof(expected, obs)

Return a callable Function that would calculate the log likelihood

!!! note
    The so called "constraint" terms (from priors) are NOT included here.
"""
function pyhf_loglikelihoodof(expected, obs)
    f(x, o) = logpdf(Poisson(x), o)
    mes(E, O) = mapreduce(f, + , E, O)
    L = function (αs)
        expe = expected(αs)
        unrolled_any(E->any(<(0), E), expe) && return -Inf
        measurement = unrolled_reduce(+, 0.0, unrolled_map(mes, expe, obs))

        return measurement
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
    pyhf_logpriorof(priors)

Return a callable Function that would calculate the log likelihood
for the priors.

!!! note
    Sometimes these are called the "constraint" terms.
"""
function pyhf_logpriorof(priors)
    pris = values(priors)
    L = function (αs)
        constraint = internal_constrainteval(pris, αs)
        return constraint
    end
    return L
end


"""
    pyhf_logjointof(expected, obs, priors)

Return a callable Function that would calculate the joint log likelihood
of likelihood and priors.

Equivalent of adding `loglikelihood` and `logprior` together.

!!! note
    The "constraint" terms are included here.
"""
function pyhf_logjointof(expected, obs, priors)
    L1 = pyhf_loglikelihoodof(expected, obs)
    L2 = pyhf_logpriorof(priors)
    L = function (αs)
        L1(αs) + L2(αs)
    end
    return L
end
