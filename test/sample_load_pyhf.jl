using Turing, LiteHF, Optim

model = load_pyhfjson("./sample.json");
const bkgexp = model["mychannel"]["bkg_MC"]
const sigexp = model["mychannel"]["signal_MC"]

###### Expected counts as a function of μ and θs
function expected_bincounts2(μ, θs)
    sigexp(μ) + bkgexp(θs)
end

###### Turing.jl models
@model function binned_b(bincounts)
    μ ~ Uniform(0, 6)
    θs ~ filldist(Normal(), 2)

    expected = expected_bincounts2(μ, θs)

    @. bincounts ~ Poisson(expected)
end

###### Feed observed data to model to construct a posterior/likelihood object
const v_data = [34,22,13,11] # observed data
const mymodel = binned_b(v_data);

###### Inference
chain_map = optimize(mymodel, MAP(), [1,1,1])
display(chain_map)
