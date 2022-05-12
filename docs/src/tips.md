```julia
function f(path)
    pydict = load_pyhfjson(path)
    pyhfmodel = build_pyhf(pydict)
    pyhfmodel
end

R = f("./blah.json")

maximize(
        R.LogLikelihood, 
        R.inits, 
        BFGS(), 
        Optim.Options(f_tol=1e-5, time_limit=10); 
        autodiff=:forward
        );
```
