## Frequentist <-> Bayesian usage

HistFactory and thus pyhf are not Bayesian procedure (some prefer to call them not frequentist
either, instead, they call what we do Likelihoodist).

However, there's a very straightforward connection between the two, with a subtle but important
twist -- evaluate prior at `x` (Bayesian) vs. shitfting "prior" to `x` and evlauate at `0` (pyhf).

To use a pyhf model in Bayesian procedure, it's almost enough to just take
`pyhf_logjointof(model::PyHFModel)` and sample posterior, bypassing all the
likelihood ratio, teststatistics, and Asimov data.

More specifically, these two likelihood are almost exactly the same:
- pyhf, frequentist likelihood of the bin counts (Poisson) + constraint terms for systematcs
- Bayesian joint likelihood of the bin counts (Poisson) + nuisance parameters priors.

In fact, if we only ever have priors like `Normal(0, 1)`, the above difference coincidentally
doesn't matter, because `pdf(x | Normal(0, 1)) === pdf(0 | Normal(x, 1))` -- it's numerically
equivalent to evaluate nuisance parameter prior at `x` (like a Bayesian), or shift the unit 
Gaussian to `x` and evaluate at still `0`.

This numerical coincidence goes away for the (relaxed, continuous) Poisson prior we need
for MC Stat systematics. In this type of prior, the distribution is not symmetric around
the mean `λ` -- causing discrepency between the two procedures.
