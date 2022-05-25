abstract type AbstractTestDistributions end

struct AsymptoticDist <: AbstractTestDistributions
    shift::Float64
    cutoff::Float64
end

"""
    asymptotic_dists(sqrtqmuA; base_dist = :normal)

Return `S+B` and `B only` teststatistics distributions.
"""
function asymptotic_dists(sqrtqmuA; base_dist = :normal)
    cutoff = base_dist == :normal ? -Inf : -sqrtqmuA
    s_plus_b_dist = AsymptoticDist(-sqrtqmuA, cutoff)
    b_dist = AsymptoticDist(0.0, cutoff)

    return s_plus_b_dist, b_dist
end

"""
    pvalue(d::AsymptoticDist, value) -> Real

Compute the p-value for a single teststatistics distribution.
"""
pvalue(d::AsymptoticDist, value) = cdf(Normal(), -(value - d.shift))

"""
    pvalue(teststat, s_plus_b_dist::AsymptoticDist, b_only_dist::AsymptoticDist)
    -> (CLsb, CLb, CLs)

Compute the confidence level for `S+B`, `B` only, and `S`.
"""
function pvalue(teststat, s_plus_b_dist::AsymptoticDist, b_only_dist::AsymptoticDist)
    CLsb = pvalue(s_plus_b_dist, teststat)
    CLb = pvalue(b_only_dist, teststat)
    CLs = CLsb / CLb

    return CLsb, CLb, CLs
end

expected_sigma(d::AsymptoticDist, nsigma) = max(d.cutoff, d.shift + nsigma)

function expected_pvalue(s_plus_b_dist::AsymptoticDist, b_only_dist::AsymptoticDist)
    stats_range = expected_sigma.(Ref(b_only_dist), 2:-1:-2)

    return [pvalue(s_plus_b_dist, i) / pvalue(b_only_dist, i) for i in stats_range]
end
