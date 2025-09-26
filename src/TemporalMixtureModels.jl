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
end

# Test stuff
using .TemporalMixtureModels, DataFrames

# create a univariate dataset
individuals_per_group = 20
t_values = 0:0.1:10
n_groups = 2
group_coefficients_1 = [ [2.0, -0.5, 0.05],  # Group 1: y = 2 - 0.5*t + 0.05*t^2
                       [1.0, 0.3, -0.02] ] # Group 2: y = 1 + 0.3*t - 0.02*t^2
group_coefficients_2 = [ [1.0, 0.4, -0.03],  # Group 1: y = 1 + 0.4*t - 0.03*t^2
                       [3.0, -0.2, 0.04] ] # Group 2: y = 3 - 0.2*t + 0.04*t^2
individual_id = 1

ids = Int[]
timepoints = Float64[]
measurements_1 = Float64[]
measurements_2 = Float64[]
for group in 1:n_groups
    for individual in 1:individuals_per_group
        for t in t_values
            y = group_coefficients_1[group][1] + group_coefficients_1[group][2]*t + group_coefficients_1[group][3]*t^2 + randn()*0.5
            z = group_coefficients_2[group][1] + group_coefficients_2[group][2]*t + group_coefficients_2[group][3]*t^2 + randn()*0.5
            push!(ids, individual_id)
            push!(timepoints, t)
            push!(measurements_1, y)
            push!(measurements_2, z)
        end
        individual_id += 1
    end
end

input_data = DataFrame(id = [ids; ids], time = [timepoints; timepoints], value = [measurements_1; measurements_2], var_name = repeat(["y", "z"], inner=length(ids)))
components = Dict(:y => PolynomialRegression(2), :z => PolynomialRegression(2))
# create a univariate mixture model with 2 components, each a polynomial of degree 2
model = MultivariateMixtureModel(2, components)

# fit the model to the data
TemporalMixtureModels.fit!(model, input_data; verbose=false, max_iter=100, tol=1e-9, hard_assignment=false)

cis, samples = TemporalMixtureModels.bootstrap_ci(model, input_data; n_bootstrap=500, alpha=0.05)


println("Log-likelihood: ", model.log_likelihood)
println("Converged: ", model.converged)
# print the fitted coefficients
for (i, comp) in enumerate(model.components)
    for (var, model) in comp
        confidence_bounds = cis[i][var]
        coefficients = ["$coeff ($lower, $upper)" for (coeff, lower, upper) in zip(model.coefficients, confidence_bounds.lower, confidence_bounds.upper)]
        println("Component $i, Variable $var coefficients: ", coefficients)
    end
end

