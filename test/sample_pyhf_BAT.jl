using LiteHF, BAT

pydict = load_pyhfjson("./test/sample.json");

expe, pri, pri_names = build_pyhf(pydict);

observed_data = [34,22,13,11];

# the 2-argument version does NOT include prior ("constraint") terms in likelihood
mylikelihood = BAT.LogDVal(loglikelihoodof(expe, observed_data))
posterior = PosteriorDensity(mylikelihood, pri)

bat_findmode(posterior).result
# (mu = 1.3064647047644158, theta = -0.06049852104383994)
