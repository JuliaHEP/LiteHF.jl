using LiteHF, Turing

pydict = load_pyhfjson("./test/sample.json");

expe, pri, pri_names = build_pyhf(pydict);

observed_data = [34,22,13,11];

# the 2-argument version does NOT include prior ("constraint") terms in likelihood
mylikelihood = loglikelihoodof(expe, observed_data)

@model function mymodel(observed)
    αs ~ arraydist(collect(values(pri)))
    Turing.@addlogprob! mylikelihood(αs)
end

optimize(mymodel(observed_data), MAP(), [1.0, 1.0])
#ModeResult with maximized lp of -13.51
# 2-element Named Vector{Float64}
# A               │ 
# ────────────────┼───────────
# Symbol("αs[1]") │    1.30648
# Symbol("αs[2]") │ -0.0605151
