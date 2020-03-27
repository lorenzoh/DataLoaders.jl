using DataLoaders
using Documenter

makedocs(;
    modules=[DataLoaders],
    authors="lorenzoh <lorenz.ohly@gmail.com>",
    repo="https://github.com/lorenzoh/DataLoaders.jl/blob/{commit}{path}#L{line}",
    sitename="DataLoaders.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://lorenzoh.github.io/DataLoaders.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/lorenzoh/DataLoaders.jl",
)
