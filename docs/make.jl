using Documenter
using LiteHF

makedocs(
    sitename = "LiteHF",
    format = Documenter.HTML(),
    modules = [LiteHF]
)

deploydocs(;
    repo="github.com/JuliaHEP/LiteHF.jl",
)
