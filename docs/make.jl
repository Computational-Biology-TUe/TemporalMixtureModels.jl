using Documenter
using TemporalMixtureModels

push!(LOAD_PATH, "../src/")

makedocs(
    sitename="TemporalMixtureModels.jl",
    pages= [
        "Home" => "index.md",
        "Manual" => [
            "Basic Usage" => "manual/usage.md",
            "Multiple Output Variables" => "manual/multivariate.md",
            "Custom Model Components" => "manual/custom_components.md",
            "Model Selection" => "manual/model_selection.md",
            "Bootstrapping for Uncertainty Estimation" => "manual/bootstrap.md",
        ],
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