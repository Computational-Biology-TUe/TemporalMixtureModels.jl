module TemporalMixtureModels

    using Random, LinearAlgebra, StructArrays, Statistics
    using DataFrames: DataFrame
    import LinearSolve as LS
    import Convex, SCS
    using ProgressMeter
    using Hungarian: hungarian
    include("input.jl")
    include("models.jl")
    include("mixturemodel.jl")
    include("solve.jl")
    include("uncertainty.jl")

    export UnivariateMixtureModel, MultivariateMixtureModel
    export PolynomialRegression, RidgePolynomialRegression, LassoPolynomialRegression
    export fit!, predict, bootstrap_ci
    export log_likelihood, posterior_responsibilities

end