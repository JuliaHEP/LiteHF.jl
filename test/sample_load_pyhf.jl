using LiteHF, Optim

dict = load_pyhfjson("./test/sample.json");

pyhfmodel = build_pyhf(dict);

@show Optim.maximizer(maximize(pyhfmodel.LogLikelihood, pyhfmodel.inits))
# 2-element Vector{Float64}:
#   1.3064374172547253
#  -0.060413406717672286
@show pyhfmodel.prior_names
# (:mu, :theta)
