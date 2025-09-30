using Documenter
using TemporalMixtureModels

push!(LOAD_PATH, "../src/")

makedocs(
    sitename="TemporalMixtureModels.jl",
    pages= [
        "Getting Started" => "index.md",
        "Tutorials" => [
            "Univariate" => "tutorials/univariate.md",
            "Multivariate" => "tutorials/multivariate.md",
            "Implementing Custom Components" => "tutorials/custom.md",
            ],
        "Manual" => [
            "Model Components" => "api/components.md",
            "Mixture Models" => "api/mixtures.md",
            "Fitting Models" => "api/fitting.md",
        ]
    ]
    )

deploydocs(
    repo = "github.com/Computational-Biology-TUe/TemporalMixtureModels.jl.git",
)