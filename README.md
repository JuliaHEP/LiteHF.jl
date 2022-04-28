# LiteHF.jl [WIP]

## Load `pyhf` JSON:
```julia
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

julia> modifier_names
2-element Vector{String}:
 "mu"
 "theta"
```

## Manual Example
```julia
using Turing, LiteHF, Optim

###### Dummy data ######
const v_data = [34,22,13,11] # observed data
const v_sig = [2,3,4,5] # signal
const v_bg = [30,19,9,4] # BKG
const variations = [1,2,3,3]

###### Background and Signal modifier definitions ######
const bkgmodis =[
                 Histosys(v_bg .+ variations, v_bg .- variations),
                 Normsys(1.1, 0.9)
                ]
const bkgexp = ExpCounts(v_bg, ["theta1", "theta2"], bkgmodis)

const sigmodis = [Normfactor()];
const sigexp = ExpCounts(v_sig, ["mu"], sigmodis);


###### Expected counts as a function of μ and θs
function expected_bincounts2(μ, θs)
    sigexp((mu = μ, )) + bkgexp((theta1=θs[1], theta2=θs[2]))
end

###### Turing.jl models
@model function binned_b(bincounts)
    μ ~ Uniform(0, 6)
    θs ~ filldist(Normal(), 2)

    expected = expected_bincounts2(μ, θs)
    @. bincounts ~ Poisson(expected)
end

###### Feed observed data to model to construct a posterior/likelihood object
const mymodel = binned_b(v_data);

###### Inference
chain_map = optimize(mymodel, MAP(), [1,1,1]) # initial guesses
display(chain_map)
```
Result:
```
ModeResult with maximized lp of -13.23
3-element Named Vector{Float64}
A               │ 
────────────────┼───────────
:μ              │     1.0383
Symbol("θs[1]") │   0.032979
Symbol("θs[2]") │ -0.0352236⏎  
```

