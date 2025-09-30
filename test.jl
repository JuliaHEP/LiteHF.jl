using LiteHF, Optim

###### Dummy data ######
const v_data = [34,22,13,11] # observed data
const v_sig = [2,3,4,5] # signal
const v_bg = [30,19,9,4] # BKG
const variations = [1,2,3,3]

###### Background and Signal modifier definitions ######
const bkgmodis =[
                 Histosys(v_bg, v_bg .+ variations, v_bg .- variations),
                 Normsys(1.1, 0.9)
                ]
const bkgexp = ExpCounts(v_bg, [:theta1, :theta2], bkgmodis)

const sigmodis = [Normfactor()];
const sigexp = ExpCounts(v_sig, [:mu], sigmodis);

function func_bkg(ps)
    return [30, 19, 9, 4] .* ps[1]
end

###### Expected counts as a function of μ and θs
function expected_bincounts(θs)
    sigexp((mu = θs[1], )) + bkgexp((theta1=θs[2], theta2=θs[3]))
end

@show expected_bincounts([1.0, 1.0, 1.0])
priors = Tuple(LiteHF._prior.([sigmodis; bkgmodis]))
L1 = LiteHF.pyhf_loglikelihoodof(expected_bincounts, v_data)
L2 = LiteHF.pyhf_logpriorof(priors)
@show L1([1.0, 0.0, 0.0])
@show L2([1.0, 0.0, 0.0])

function expected_bincounts2(θs)
    sigexp((mu = θs[1], )) + func_bkg(θs[2:3])
end

priors2 = Tuple(LiteHF._prior.([sigmodis; Normfactor(); Normfactor()]))
L1_2 = LiteHF.pyhf_loglikelihoodof(expected_bincounts2, v_data)
L2_2 = LiteHF.pyhf_logpriorof(priors2)
@show L1_2([1.0, 1.0, 1.0])
@show L2_2([1.0, 1.0, 1.0])
