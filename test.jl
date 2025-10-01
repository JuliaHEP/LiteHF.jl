using LiteHF, LiteHF.Optim, NumericalDistributions

###### Dummy data ######
const v_data = [2,6,10,11] # observed data
const v_sig = [1,1,4,2] # nominal signal
const sigmodis = [
                    Normfactor(),
                    Histosys(v_sig, v_sig .* 1.05, v_sig .* 0.95),
];
const sigexp = ExpCounts(v_sig, [:mu, :jes], sigmodis);

# linear
f(x, p) = x*p
function func_bkg(ps)
    norm_pdf = NumericallyIntegrable(x -> f(x, ps[2]), (0.5, 4.5))
    return ps[1] * pdf.(norm_pdf, 1:4)
end

@show func_bkg([20, 1.5])

# const v_bg = [30,19,9,4] # BKG
# const variations = [1,2,3,3]
# const bkgmodis =[
#                  Histosys(v_bg, v_bg .+ variations, v_bg .- variations),
#                  Normsys(1.1, 0.9)
#                 ]
# const bkgexp = ExpCounts(v_bg, [:theta1, :theta2], bkgmodis)
# function expected_bincounts(θs)
#     sigexp((mu = θs[1], )) + bkgexp((theta1=θs[2], theta2=θs[3]))
# end

# @show expected_bincounts([1.0, 1.0, 1.0])
# priors = Tuple(LiteHF._prior.([sigmodis; bkgmodis]))
# L1 = LiteHF.pyhf_loglikelihoodof(expected_bincounts, v_data)
# L2 = LiteHF.pyhf_logpriorof(priors)
# @show L1([1.0, 0.0, 0.0])
# @show L2([1.0, 0.0, 0.0])

function expected_bincounts2(θs)
    sigexp((mu = θs[1], jes=θs[2])) + func_bkg(θs[3:4])
end
L_joint = LiteHF.pyhf_logjointof(expected_bincounts2, v_data, priors2)

priors2 = Tuple(LiteHF._prior.([sigmodis; Normfactor(); Normfactor()]))
L1_2 = LiteHF.pyhf_loglikelihoodof(expected_bincounts2, v_data)
L2_2 = LiteHF.pyhf_logpriorof(priors2)
@show L1_2([1.0, 0.0, 20, 1.5])
@show L2_2([1.0, 0.0, 20, 1.5])

@show Optim.maximizer(Optim.maximize(L_joint, [1.0, 0.0, 20.0, 1.5]))
