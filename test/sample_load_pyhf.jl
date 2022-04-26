using LiteHF, Optim

pydict = load_pyhfjson("./test/sample.json");

expe, pri, pri_names = build_pyhf(pydict);

observed_data = [34,22,13,11];

# the 3-argument includes the prior ("constraint") term in likelihood
NLL(αs) = -loglikelihoodof(expe, pri, observed_data)(αs)

optimize(NLL, [1.0, 1.0]).minimizer
# 2-element Vector{Float64}:
#   1.3064374172547253
#  -0.060413406717672286
