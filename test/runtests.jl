using LiteHF, Optim
using Test

@testset "Interpolations" begin
    # Ref https://cds.cern.ch/record/1456844/files/CERN-OPEN-2012-016.pdf
    nominal = [2.5, 2.0]
    hi_shape_var= [1.4, 3.8]
    hi_shape_temp = nominal + hi_shape_var
    lo_shape_var= [1.3, 1.2]
    lo_shape_temp = nominal - lo_shape_var

    @testset "InterpCode0" begin
        i0 = LiteHF.InterpCode0(nominal, hi_shape_temp, lo_shape_temp)
        i0two = LiteHF.InterpCode0(hi_shape_var, lo_shape_var)
        @test i0.Δ_up ≈ i0two.Δ_up
        @test i0.Δ_down ≈ i0two.Δ_down

        @test i0(0.0) == zeros(2)
        @test i0(1.0) + nominal ==  hi_shape_temp
        @test i0(-1.0) + nominal == lo_shape_temp
        @test i0(-0.5) + nominal == nominal - 0.5*lo_shape_var
    end

    @testset "InterpCode2" begin
        i2 = LiteHF.InterpCode2(nominal, hi_shape_temp, lo_shape_temp)
        @test i2.a == 0.5 * (hi_shape_temp + lo_shape_temp) - nominal
        @test i2.b == 0.5 * (hi_shape_temp - lo_shape_temp)

        @test i2(0.0) == zeros(2)
        @test i2(1.0) + nominal ==  hi_shape_temp
        @test i2(-1.0) + nominal ≈ lo_shape_temp
        α = -0.5
        @test i2(α) ≈ i2.a * α^2 + i2.b * α
        α = 1.2
        @test i2(α) ≈ @. (i2.b + 2*i2.a) * (α - 1)
        α = -1.4
        @test i2(α) ≈ @. (i2.b - 2*i2.a) * (α + 1)
    end

    @testset "InterpCode4" begin
        i4 = LiteHF.InterpCode4(nominal, hi_shape_temp, lo_shape_temp)
        @test i4(0.0) == ones(length(nominal))
        @test i4(1.0) .* nominal ≈ hi_shape_temp
        @test i4(-1.0) .* nominal ≈ lo_shape_temp
    end

    hi_sf= 1.1
    lo_sf= 0.84
    hi_sf_temp = hi_sf * nominal
    lo_sf_temp = lo_sf * nominal

    @testset "InterpCode1" begin
        i1 = LiteHF.InterpCode1(nominal, hi_sf_temp, lo_sf_temp)
        i1two = LiteHF.InterpCode1(hi_sf, lo_sf)
        @test i1.f_up ≈ i1two.f_up
        @test i1.f_down ≈ i1two.f_down
        @test i1(1.0)*nominal ≈ hi_sf_temp
        @test i1(-1.0)*nominal ≈ lo_sf_temp
    end
end

function loadmodel(path)
    pydict = load_pyhfjson(path)
    pyhfmodel = build_pyhf(pydict)
end

testmodel(path::String, OPT = BFGS()) = testmodel(loadmodel(path), OPT)
function testmodel(pyhfmodel, OPT = BFGS())
    LL = pyhf_logjointof(pyhfmodel)
    res = maximize(LL, pyhfmodel.inits,
                   OPT, Optim.Options(g_tol=1e-5); autodiff=:forward)
    best_paras = Optim.maximizer(res)
    twice_nll = -2*LL(best_paras)
end

stateerror_shape = loadmodel(joinpath(@__DIR__, "./pyhfjson/sample_staterror_shapesys.json"))
@testset "Basic expected tests" begin
    R = stateerror_shape
    @test R.expected(ones(5))[1] == [23.0, 15.0]
    @test R.expected(R.inits)[1] == [23.0, 15.0]
    @test R.expected([0.81356312, 0.99389009, 1.01090199, 0.99097032, 1.00290362])[1] ≈
    [22.210797046385544, 14.789399036653428]
end

@testset "Conditional maximizer" begin
    RR = loadmodel(joinpath(@__DIR__, "./pyhfjson/sample_normsys.json"))
    likelihood, _ = cond_maximize(pyhf_logjointof(RR), 1.0, RR.inits[2:end])
    @test -2*likelihood <= 21.233919574137236 # better than pyhf value
    likelihood, _ = cond_maximize(pyhf_logjointof(RR), 0.0, RR.inits[2:end])
    @test -2*likelihood <= 27.6021945001722
end

@testset "Full model" begin
    @test testmodel(joinpath(@__DIR__, "./pyhfjson/single_channel_big.json")) ≈ 80.67893633848638 rtol=0.0001
    @test testmodel(joinpath(@__DIR__, "./pyhfjson/multi_channel.json")) ≈ 39.02800819146104 rtol=0.0001
    # _logabsgamma doesn't have DiffRule right now
    @test testmodel(stateerror_shape) ≈ 16.66838236805484 rtol = 0.0001
end
