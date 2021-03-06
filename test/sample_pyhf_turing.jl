using LiteHF, Turing, Optim

dict = load_pyhfjson("./test/pyhfjson/sample.json");

const pyhfmodel = build_pyhf(dict);
const priors_array = collect(values(pyhfmodel.priors))

@model function mymodel(observed)
    αs ~ arraydist(priors_array)
    expected = pyhfmodel.expected(αs)
    @. observed ~ Poisson(expected)
end

observed_data = [34,22,13,11];
@show optimize(mymodel(observed_data), MAP(), pyhfmodel.inits)
#ModeResult with maximized lp of -13.51
# 2-element Named Vector{Float64}
# A               │ 
# ────────────────┼───────────
# Symbol("αs[1]") │    1.30648
# Symbol("αs[2]") │ -0.0605151
