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

    if any(<(0), expected)
        Turing.@addlogprob! -Inf
        return
    end

    @. bincounts ~ Poisson(expected)
end

###### Feed observed data to model to construct a posterior/likelihood object
const mymodel = binned_b(v_data);

###### Inference
chain_map = optimize(mymodel, MAP())
display(chain_map)
