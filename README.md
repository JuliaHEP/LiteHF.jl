# LiteHF.jl [WIP]

## Example
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
const bkgexp = ExpCounts(v_bg, bkgmodis)

const sigmodis = [Normfactor()];
const sigexp = ExpCounts(v_sig, sigmodis);


###### Expected counts as a function of μ and θs
function expected_bincounts2(μ, θs)
    sigexp(μ) + bkgexp(θs...)
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

## Load `pyhf` JSON:
```julia
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
```
