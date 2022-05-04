using LiteHF, Optim
using Test

function testload(path)
    pydict = load_pyhfjson(path)
    pyhfmodel = build_pyhf(pydict)
    res = maximize(pyhfmodel.LogLikelihood, pyhfmodel.prior_inits,
                   BFGS(), Optim.Options(g_tol=1e-5); autodiff=:forward)
    best_paras = Optim.maximizer(res)
    Dict(pyhfmodel.prior_names .=> best_paras)
    twice_nll = -2*pyhfmodel.LogLikelihood(best_paras)
end

@testset "Full model" begin
    # Write your tests here.
    @test isapprox(testload(joinpath(@__DIR__, "./pyhfjson/single_channel_big.json")), 80.67893633848638;
                  rtol=0.0001)
    @test isapprox(testload(joinpath(@__DIR__, "./pyhfjson/multi_channel.json")), 39.02800819146104;
                  rtol=0.0001)
end
