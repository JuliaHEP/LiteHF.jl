var documenterSearchIndex = {"docs":
[{"location":"freq_baye/#Frequentist-Bayesian-usage","page":"-","title":"Frequentist <-> Bayesian usage","text":"","category":"section"},{"location":"freq_baye/","page":"-","title":"-","text":"HistFactory and thus pyhf are not Bayesian procedure (some prefer to call them not frequentist either, instead, they call what we do Likelihoodist).","category":"page"},{"location":"freq_baye/","page":"-","title":"-","text":"However, there's a very straightforward connection between the two, with a subtle but important twist – evaluate prior at x (Bayesian) vs. shitfting \"prior\" to x and evlauate at 0 (pyhf).","category":"page"},{"location":"freq_baye/","page":"-","title":"-","text":"To use a pyhf model in Bayesian procedure, it's almost enough to just take pyhf_logjointof(model::PyHFModel) and sample posterior, bypassing all the likelihood ratio, teststatistics, and Asimov data.","category":"page"},{"location":"freq_baye/","page":"-","title":"-","text":"More specifically, these two likelihood are almost exactly the same:","category":"page"},{"location":"freq_baye/","page":"-","title":"-","text":"pyhf, frequentist likelihood of the bin counts (Poisson) + constraint terms for systematcs\nBayesian joint likelihood of the bin counts (Poisson) + nuisance parameters priors.","category":"page"},{"location":"freq_baye/","page":"-","title":"-","text":"In fact, if we only ever have priors like Normal(0, 1), the above difference coincidentally doesn't matter, because pdf(x | Normal(0, 1)) === pdf(0 | Normal(x, 1)) – it's numerically equivalent to evaluate nuisance parameter prior at x (like a Bayesian), or shift the unit  Gaussian to x and evaluate at still 0.","category":"page"},{"location":"freq_baye/","page":"-","title":"-","text":"This numerical coincidence goes away for the (relaxed, continuous) Poisson prior we need for MC Stat systematics. In this type of prior, the distribution is not symmetric around the mean λ – causing discrepency between the two procedures.","category":"page"},{"location":"#LiteHF.jl","page":"APIs","title":"LiteHF.jl","text":"","category":"section"},{"location":"","page":"APIs","title":"APIs","text":"Documentation for LiteHF.jl","category":"page"},{"location":"","page":"APIs","title":"APIs","text":"Modules = [LiteHF]\nOrder   = [:type, :function]","category":"page"},{"location":"#LiteHF.ExpCounts","page":"APIs","title":"LiteHF.ExpCounts","text":"struct ExpCounts{T, M} #M is a long Tuple for unrolling purpose.\n    nominal::T\n    modifier_names::Vector{Symbol}\n    modifiers::M\nend\n\nA callable struct that returns the expected count given modifier nuisance parameter values. The # of parameters passed must equal to length of modifiers. See _expkernel\n\n\n\n\n\n","category":"type"},{"location":"#LiteHF.FlatPrior","page":"APIs","title":"LiteHF.FlatPrior","text":"Pseudo flat prior in the sense that `logpdf()` always evaluates to zero,\nbut `rand()`, `minimum()`, and `maximum()` behaves like `Uniform(a, b)`.\n\n\n\n\n\n","category":"type"},{"location":"#LiteHF.Histosys","page":"APIs","title":"LiteHF.Histosys","text":"Histosys is defined by two vectors represending bin counts in hi_data and lo_data\n\n\n\n\n\n","category":"type"},{"location":"#LiteHF.InterpCode0","page":"APIs","title":"LiteHF.InterpCode0","text":"InterpCode0{T}\n\nCallable struct for interpolation for additive modifier. Code0 is the two-piece linear interpolation.\n\n\n\n\n\n","category":"type"},{"location":"#LiteHF.InterpCode1","page":"APIs","title":"LiteHF.InterpCode1","text":"InterpCode1{T}\n\nCallable struct for interpolation for multiplicative modifier. Code1 is the exponential interpolation.\n\n\n\n\n\n","category":"type"},{"location":"#LiteHF.InterpCode4","page":"APIs","title":"LiteHF.InterpCode4","text":"InterpCode4{T}\n\nCallable struct for interpolation for additive modifier. Code4 is the exponential + 6-order polynomial interpolation.\n\n\n\n\n\n","category":"type"},{"location":"#LiteHF.Lumi","page":"APIs","title":"LiteHF.Lumi","text":"Luminosity doesn't need interpolation, σ is provided at modifier construction time. In pyhf JSON, this  information lives in the \"Measurement\" section, usually near the end of the JSON file.\n\n\n\n\n\n","category":"type"},{"location":"#LiteHF.MultOneHot","page":"APIs","title":"LiteHF.MultOneHot","text":"MultOneHot{T} <: AbstractVector{T}\n\nInternal type used to avoid allocation for per-bin multiplicative systematics. It behaves as a vector with length nbins and  only has value α on nthbin-th index, the rest being one(T). See also binidentity.\n\n\n\n\n\n","category":"type"},{"location":"#LiteHF.Normfactor","page":"APIs","title":"LiteHF.Normfactor","text":"Normfactor is unconstrained, so interp is just identity.\n\n\n\n\n\n","category":"type"},{"location":"#LiteHF.Normsys-Tuple{Number, Number}","page":"APIs","title":"LiteHF.Normsys","text":"Normsys is defined by two multiplicative scalars\n\n\n\n\n\n","category":"method"},{"location":"#LiteHF.PyHFModel","page":"APIs","title":"LiteHF.PyHFModel","text":"struct PyHFModel{E, O, P} <: AbstractModel\n    expected::E\n    observed::O\n    priors::P\n    prior_names\n    inits::Vector{Float64}\nend\n\nStruct for holding result from build_pyhf. List of accessor functions is \n\nexpected(p::PyHFModel)\nobserved(p::PyHFModel)\npriors(p::PyHFModel)\nprior_names(p::PyHFModel)\ninits(p::PyHFModel)\n\n\n\n\n\n","category":"type"},{"location":"#LiteHF.RelaxedPoisson","page":"APIs","title":"LiteHF.RelaxedPoisson","text":"RelaxedPoisson\n\nPoisson with logpdf continuous in k. Essentially by replacing denominator with gamma function.\n\nwarning: Warning\nThe Distributions.logpdf has been redefined to be logpdf(d::RelaxedPoisson, x) = logpdf(d, x*d.λ). This is to reproduce the Poisson constraint term in pyhf, which is a hack introduced for Asimov dataset.\n\n\n\n\n\n","category":"type"},{"location":"#LiteHF.Shapefactor","page":"APIs","title":"LiteHF.Shapefactor","text":"Shapefactor is unconstrained, so interp is just identity. Unlike Normfactor, this is per-bin\n\n\n\n\n\n","category":"type"},{"location":"#LiteHF.Shapesys","page":"APIs","title":"LiteHF.Shapesys","text":"Shapesys doesn't need interpolation, similar to Staterror\n\n\n\n\n\n","category":"type"},{"location":"#LiteHF.Staterror","page":"APIs","title":"LiteHF.Staterror","text":"Staterror doesn't need interpolation, but it's a per-bin modifier. Information regarding which bin is the target is recorded in bintwoidentity.\n\nThe δ is the absolute yield uncertainty in each bin, and the relative uncertainty: δ / nominal is taken to be the σ of the prior, i.e. α ~ Normal(1, δ/nominal)\n\n\n\n\n\n","category":"type"},{"location":"#LiteHF.T_q0","page":"APIs","title":"LiteHF.T_q0","text":"Test statistic for discovery of a positive signal q0 = \\tilde{t}0 See equation 12 in: https://arxiv.org/pdf/1007.1727.pdf for reference. Note that this IS NOT a special case of q_\\mu for \\mu = 0.\n\n\n\n\n\n","category":"type"},{"location":"#LiteHF.T_qmu","page":"APIs","title":"LiteHF.T_qmu","text":"Test statistic for upper limits See equation 14 in: https://arxiv.org/pdf/1007.1727.pdf for reference. Note that q0 IS NOT a special case of q\\mu for \\mu = 0.\n\n\n\n\n\n","category":"type"},{"location":"#LiteHF.T_tmu","page":"APIs","title":"LiteHF.T_tmu","text":"T_tmu\n\n    t_mu = -2lnlambda(mu)\n\n\n\n\n\n","category":"type"},{"location":"#LiteHF.T_tmutilde","page":"APIs","title":"LiteHF.T_tmutilde","text":"T_tmutilde(LL, inits)\n\n    widetildet_mu = -2lnwidetildelambda(mu)\n\n\n\n\n\n","category":"type"},{"location":"#LiteHF.AsimovModel-Tuple{PyHFModel, Any}","page":"APIs","title":"LiteHF.AsimovModel","text":"AsimovModel(model::PyHFModel, μ)::PyHFModel\n\nGenerate the Asimov model when fixing μ (POI) to a value. Notice this changes the priors and observed compare to the original model.\n\n\n\n\n\n","category":"method"},{"location":"#LiteHF._expkernel-Tuple{}","page":"APIs","title":"LiteHF._expkernel","text":"_expkernel(modifiers, nominal, αs)\n\nThe Unrolled.@unroll kernel function that computs the expected counts.\n\n\n\n\n\n","category":"method"},{"location":"#LiteHF.asimovdata-Tuple{PyHFModel, Any}","page":"APIs","title":"LiteHF.asimovdata","text":"asimovdata(model::PyHFModel, μ)\n\nGenerate the Asimov dataset and asimov priors, which is the expected counts after fixing POI to μ and optimize the nuisance parameters.\n\n\n\n\n\n","category":"method"},{"location":"#LiteHF.asymptotic_dists-Tuple{Any}","page":"APIs","title":"LiteHF.asymptotic_dists","text":"asymptotic_dists(sqrtqmuA; base_dist = :normal)\n\nReturn S+B and B only teststatistics distributions.\n\n\n\n\n\n","category":"method"},{"location":"#LiteHF.binidentity-Tuple{Any, Any}","page":"APIs","title":"LiteHF.binidentity","text":"binidentity(nbins, nthbin)\n\nA functional that used to track per-bin systematics. Returns the closure function over nbins, nthbin:\n\n    α -> MultOneHot(nbins, nthbin, α)\n\n\n\n\n\n","category":"method"},{"location":"#LiteHF.build_channel-Tuple{Any}","page":"APIs","title":"LiteHF.build_channel","text":"build_channel(rawjdict[:channels][1][:samples][2]) =>\nDict{String, ExpCounts}\n\n\n\n\n\n","category":"method"},{"location":"#LiteHF.build_modifier!-Tuple{Any, Any}","page":"APIs","title":"LiteHF.build_modifier!","text":"build_modifier(rawjdict[:channels][1][:samples][2][:modifiers][1]) =>\n<:AbstractModifier\n\n\n\n\n\n","category":"method"},{"location":"#LiteHF.build_modifier-Union{Tuple{T}, Tuple{Any, Type{T}}} where T","page":"APIs","title":"LiteHF.build_modifier","text":"build_modifier(...[:modifiers][1][:data], Type) =>\n<:AbstractModifier\n\n\n\n\n\n","category":"method"},{"location":"#LiteHF.build_pyhf-Tuple{Any}","page":"APIs","title":"LiteHF.build_pyhf","text":"build_pyhf(load_pyhfjson(path)) -> PyHFModel\n\nthe expected(αs) is a function that takes vector or tuple of length N, where N is also the length of priors and priornames. In other words, these three fields of the returned object are aligned.\n\nnote: Note\nThe bins from different channels are put into a NTuple{Nbins, Vector}.\n\n\n\n\n\n","category":"method"},{"location":"#LiteHF.build_sample","page":"APIs","title":"LiteHF.build_sample","text":"build_sample(rawjdict[:channels][1][:samples][2]) =>\nExpCounts\n\n\n\n\n\n","category":"function"},{"location":"#LiteHF.get_condLL-Tuple{Any, Any}","page":"APIs","title":"LiteHF.get_condLL","text":"get_condLL(LL, μ)\n\nGiven the original log-likelihood function and a value for parameter of interest, return a function condLL(nuisance_θs) that takes one less argument than the original LL. The μ is assumed to be the first element in input vector.\n\n\n\n\n\n","category":"method"},{"location":"#LiteHF.get_lnLR-Tuple{Any, Any}","page":"APIs","title":"LiteHF.get_lnLR","text":"get_lnLR(LL, inits)\n\nA functional that returns a function lnLR(μ::Number) that evaluates to log of likelihood-ratio:\n\nlnlambda(mu) = ln(fracL(mu hathattheta)L(hatmu hattheta)) = LL(mu hathattheta) - LL(hatmu hattheta)\n\nwarning: Warning\nwe assume the POI is the first in the input array.\n\n\n\n\n\n","category":"method"},{"location":"#LiteHF.get_lnLRtilde-Tuple{Any, Any}","page":"APIs","title":"LiteHF.get_lnLRtilde","text":"get_lnLRtilde(LL, inits)\n\nA functional that returns a function lnLRtilde(μ::Number) that evaluates to log of likelihood-ratio:\n\nlnwidetildelambda(mu)\n\nSee equation 10 in: https://arxiv.org/pdf/1007.1727.pdf for refercen.\n\n\n\n\n\n","category":"method"},{"location":"#LiteHF.get_teststat-Tuple{Any, Any, Type{T_tmu}}","page":"APIs","title":"LiteHF.get_teststat","text":"get_teststat(LL, inits, ::Type{T}) where T <: ATS\n\nReturn a callable function t(μ) that evaluates to the value of corresponding test statistics.\n\n\n\n\n\n","category":"method"},{"location":"#LiteHF.internal_expected-Tuple{Any, Any, Any}","page":"APIs","title":"LiteHF.internal_expected","text":"internal_expected(Es, Vs, αs)\n\nThe @generated function that computes expected counts in expected(PyHFModel, parameters) evaluation. The Vs::NTuple{N, Vector{Int64}} has the same length as Es::NTuple{N, ExpCounts}.\n\nIn general αs is shorter than Es and Vs because a given nuisance parameter α may appear in multiple sample / modifier.\n\nnote: Note\nIf for example Vs[1] = [1,3,4], it means that the first ExpCount in Es is evaluated withEs[1](@view αs[[1,3,4]])and so on.\n\n\n\n\n\n","category":"method"},{"location":"#LiteHF.load_pyhfjson-Tuple{Any}","page":"APIs","title":"LiteHF.load_pyhfjson","text":"load_pyhfjson(path)\n\n\n\n\n\n","category":"method"},{"location":"#LiteHF.pvalue-Tuple{Any, LiteHF.AsymptoticDist, LiteHF.AsymptoticDist}","page":"APIs","title":"LiteHF.pvalue","text":"pvalue(teststat, s_plus_b_dist::AsymptoticDist, b_only_dist::AsymptoticDist)\n-> (CLsb, CLb, CLs)\n\nCompute the confidence level for S+B, B only, and S.\n\n\n\n\n\n","category":"method"},{"location":"#LiteHF.pvalue-Tuple{LiteHF.AsymptoticDist, Any}","page":"APIs","title":"LiteHF.pvalue","text":"pvalue(d::AsymptoticDist, value) -> Real\n\nCompute the p-value for a single teststatistics distribution.\n\n\n\n\n\n","category":"method"},{"location":"#LiteHF.pyhf_logjointof-Tuple{Any, Any, Any}","page":"APIs","title":"LiteHF.pyhf_logjointof","text":"pyhf_logjointof(expected, obs, priors)\n\nReturn a callable Function that would calculate the joint log likelihood of likelihood and priors.\n\nEquivalent of adding loglikelihood and logprior together.\n\nnote: Note\nThe \"constraint\" terms are included here.\n\n\n\n\n\n","category":"method"},{"location":"#LiteHF.pyhf_loglikelihoodof-Tuple{Any, Any}","page":"APIs","title":"LiteHF.pyhf_loglikelihoodof","text":"pyhf_loglikelihoodof(expected, obs)\n\nReturn a callable Function L(αs) that would calculate the log likelihood. expected is a callable of αs as well.\n\nnote: Note\nThe so called \"constraint\" terms (from priors) are NOT included here.\n\n\n\n\n\n","category":"method"},{"location":"#LiteHF.pyhf_logpriorof-Tuple{Any}","page":"APIs","title":"LiteHF.pyhf_logpriorof","text":"pyhf_logpriorof(priors)\n\nReturn a callable Function L(αs) that would calculate the log likelihood for the priors.\n\nnote: Note\nSometimes these are called the \"constraint\" terms.\n\n\n\n\n\n","category":"method"},{"location":"#LiteHF.sortpoi!-Tuple{Any, Any, Any}","page":"APIs","title":"LiteHF.sortpoi!","text":"Ensure POI parameter always comes first in the input array.\n\n\n\n\n\n","category":"method"},{"location":"tips/","page":"Tips & Recommendations","title":"Tips & Recommendations","text":"function f(path)\n    pydict = load_pyhfjson(path)\n    pyhfmodel = build_pyhf(pydict)\n    pyhfmodel\nend\n\nR = f(\"./blah.json\")\n\nmaximize(\n        R.LogLikelihood, \n        R.inits, \n        BFGS(), \n        Optim.Options(f_tol=1e-5, time_limit=10); \n        autodiff=:forward\n        );","category":"page"}]
}
