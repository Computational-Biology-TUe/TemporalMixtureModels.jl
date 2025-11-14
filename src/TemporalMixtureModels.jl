module TemporalMixtureModels
    using Random
    using Distributions
    using LinearAlgebra
    using Statistics
    import LinearSolve as LS
    using Hungarian
    using ProgressMeter
    using RequiredInterfaces

    include("data.jl")
    include("components/core.jl")
    include("components/regression.jl")
    include("composition.jl")
    include("errormodels.jl")
    include("solve.jl")
    include("bootstrap.jl")
    include("evaluation.jl")

    # sample data for testing and examples
    export example_bp_data

    # components base
    export Component, n_parameters, initialize_parameters, fit!
    
    # regression components
    export PolynomialRegression, RidgeRegression, LassoRegression

    # composition
    export CompositeComponent, @component

    # error models
    export ErrorModel, NormalError

    # solving mixture models
    export fit_mixture, predict, posterior_responsibilities
    export MixtureResult, MixtureData
    
    # bootstrap
    export bootstrap

    # evaluation metrics
    export loglikelihood, aic, bic
    
end # module TemporalMixtureModels

