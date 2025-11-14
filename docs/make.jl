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
            "Bootstrapping for Uncertainty Estimation" => "manual/bootstrap.md",
            "Custom Model Components" => "manual/custom_components.md",
            "Regularized Regression Components" => "manual/regularized_regression.md",
            "Evaluation Metrics" => "manual/evaluation_metrics.md",
            #"Model Selection" => "manual/model_selection.md",
        ],
        # "Tutorials" => [
        #     "Univariate" => "tutorials/univariate.md",
        #     "Multivariate" => "tutorials/multivariate.md",
        #     "Implementing Custom Components" => "tutorials/custom.md",
        #     ],
        # "Manual" => [
        #     "Model Components" => "api/components.md",
        #     "Mixture Models" => "api/mixtures.md",
        #     "Fitting Models" => "api/fitting.md",
        # ]
    ]
    )

deploydocs(
    repo = "github.com/Computational-Biology-TUe/TemporalMixtureModels.jl.git",
)