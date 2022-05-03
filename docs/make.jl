using Documenter
using LiteHF

makedocs(
    sitename = "LiteHF",
    format = Documenter.HTML(),
    modules = [LiteHF],
    pages=[
        "APIs" => "index.md",
        "Tips & Recommendations" => "tips.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaHEP/LiteHF.jl",
)
