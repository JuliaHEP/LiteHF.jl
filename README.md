# LiteHF.jl
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliahep.github.io/LiteHF.jl/dev/)
[![Build Status](https://github.com/JuliaHEP/LiteHF.jl/workflows/CI/badge.svg)](https://github.com/JuliaHEP/LiteHF.jl/actions)
[![Codecov](https://codecov.io/gh/JuliaHEP/LiteHF.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaHEP/LiteHF.jl)


## TODO
- [ ] Implement teststatistics helper functions
- [ ] Re-structre the `PyHFModel` such that the `POI` component can be swapped out.

## Load `pyhf` JSON:
```julia
using LiteHF, Optim

dict = load_pyhfjson("./test/sample.json");

pyhfmodel = build_pyhf(dict);

@show Optim.maximizer(maximize(pyhfmodel.LogLikelihood, pyhfmodel.inits))
# 2-element Vector{Float64}:
#   1.3064374172547253
#  -0.060413406717672286
@show pyhfmodel.prior_names
# (:mu, :theta)
```

## `pyhf` JSON + Turing.jl:
```julia
using LiteHF, Turing, Optim

dict = load_pyhfjson("./test/sample.json");

const pyhfmodel = build_pyhf(dict);
# unpack `NamedTuple` into just an array of prior distributions
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
```

## `pyhf` JSON + BAT.jl:
```julia
using LiteHF, BAT

pydict = load_pyhfjson("./test/sample.json");

pyhfmodel = build_pyhf(pydict);

mylikelihood(αs) = BAT.LogDVal(pyhfmodel.LogLikelihood(αs))
posterior = PosteriorDensity(mylikelihood, pyhfmodel.priors)

@show bat_findmode(posterior).result
# (mu = 1.3064647047644158, theta = -0.06049852104383994)
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
    μ ~ Turing.Flat()
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

