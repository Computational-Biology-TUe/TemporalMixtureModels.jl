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
    include("components/polynomialregression.jl")
    include("composition.jl")
    include("errormodels.jl")
    include("solve.jl")
    include("bootstrap.jl")

    # sample data for testing and examples
    export example_bp_data

    # components base
    public Component, n_parameters, initialize_parameters, fit!
    
    # polynomial regression component
    export PolynomialRegression

    # composition
    export CompositeComponent, @component

    # error models
    export ErrorModel, NormalError

    # solving mixture models
    export fit_mixture, predict, posterior_responsibilities
    public MixtureResult, MixtureData
    
    # bootstrap
    export bootstrap
    
end # module TemporalMixtureModels

