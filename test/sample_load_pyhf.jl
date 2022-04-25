using Turing, Optim, LiteHF

pyhfmodel = load_pyhfjson("./sample.json");

modifier_names, model = LiteHF.genmodel(pyhfmodel);

observed = [34,22,13,11];

optimize(model(observed), MAP(), [1,1])
