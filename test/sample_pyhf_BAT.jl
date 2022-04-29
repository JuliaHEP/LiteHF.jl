using LiteHF, BAT

pydict = load_pyhfjson("./test/sample.json");

pyhfmodel = build_pyhf(pydict);

# the 2-argument version does NOT include prior ("constraint") terms in likelihood
mylikelihood(αs) = BAT.LogDVal(pyhfmodel.LogLikelihood(αs))
posterior = PosteriorDensity(mylikelihood, pyhfmodel.priors)

@show bat_findmode(posterior).result
# (mu = 1.3064647047644158, theta = -0.06049852104383994)
