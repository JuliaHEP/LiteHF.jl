function opt_maximize(f, inits, alg = NelderMeld())
    Optim.maximize(f, inits, alg, Optim.Options(g_tol = 1e-5, iterations=10^4); autodiff=:forward)
end



