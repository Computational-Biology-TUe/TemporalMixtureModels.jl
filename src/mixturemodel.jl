abstract type AbstractMixtureModel{T<:Real} end
const ComponentDict{T} = Dict{Symbol, AbstractMixtureModelComponent{T}}

mutable struct UnivariateMixtureModel{T<:Real} <: AbstractMixtureModel{T}
    components::Vector{AbstractMixtureModelComponent{T}}
    weights::Vector{T}  # mixture weights, should sum to 1
    variances::Vector{T}  # variance for each component
    log_likelihood::T
    converged::Bool
    iterations::Int
end


"""
Create a univariate mixture model with `n_components` where each component is defined as the model passed in `component`. 

## Arguments
- `n_components::Int`: Number of mixture components. 
- `component::AbstractMixtureModelComponent{T}`: An instance of a component model (e.g., `PolynomialRegression(2)` for a second order polynomial regression).

A univariate mixture model is defined as:

```math
y_i(t) \\sim \\sum_{k=1}^{K} \\pi_k f(y_i(t) | t, \\theta_k)
```

where ``f(y_i(t) | t, \\theta_k)`` is the density defined by the regression model (e.g., polynomial regression) with parameters ``\\theta_k``, and ``\\pi_k`` are the mixture weights and ``t`` is time.

## Example
Creating a univariate mixture model with 3 components, each a polynomial of degree 2:
```julia
component = PolynomialRegression(2)
model = UnivariateMixtureModel(3, component)
```
"""
function UnivariateMixtureModel(n_components::Int, component::AbstractMixtureModelComponent{T}) where T
    components = [deepcopy(component) for _ in 1:n_components]
    variances = zeros(T, n_components)
    return UnivariateMixtureModel{T}(components, fill(one(T) / n_components, n_components), variances, -Inf, false, 0)
end

mutable struct MultivariateMixtureModel{T<:Real} <: AbstractMixtureModel{T}
    components::Vector{ComponentDict{T}}  # each row is a component, each column a variable
    weights::Vector{T}  # mixture weights, should sum to 1
    variances::Matrix{T}  # variance for each variable in each component
    log_likelihood::T
    converged::Bool
    iterations::Int
end

"""
Create a multivariate mixture model with `n_components` where each component is defined by the models passed in the `components` dictionary.
## Arguments
- `n_components::Int`: Number of mixture components. 
- `components::Dict{Symbol, <:AbstractMixtureModelComponent{T}}`: A dictionary where keys are variable names (as `Symbol`) and values are instances of component models (e.g., `PolynomialRegression(2)` for a second order polynomial regression).

In this case, a multivariate mixture model is defined using *independent* components as:

```math
(y_{i1}, y_{i2}, ..., y_{iJ}) \\sim \\sum_{k=1}^{K} \\pi_k \\prod_{j = 1}^{J} f_j(y_{ij} | t, \\theta_{kj})
```
where ``f_j(y_{ij} | t, \\theta_{kj})`` is the density defined by the regression model for variable ``j``` with parameters ``\\theta_{kj}``, and ``\\pi_k`` are the mixture weights.

It is assumed that the variables are independent given the component assignment, i.e., the joint density is the product of the individual densities.

## Example
Creating a multivariate mixture model with 2 components, each a polynomial of degree 2 for variables `:y` and `:z`:
```julia
components = Dict(:y => PolynomialRegression(2), :z => PolynomialRegression(2))
model = MultivariateMixtureModel(2, components)
```
"""
function MultivariateMixtureModel(n_components::Int, components::Dict{Symbol, <:AbstractMixtureModelComponent{T}}) where T
    components = [ComponentDict{T}(deepcopy(components)) for _ in 1:n_components]
    variances = zeros(T, length(components[1]), n_components)  # rows: variables, columns: components
    return MultivariateMixtureModel{T}(components, fill(one(T) / n_components, n_components), variances, -Inf, false, 0)
end

function n_components(model::AbstractMixtureModel)
    return length(model.components)
end

function reinit!(model::UnivariateMixtureModel{T}) where T
    model.weights .= T(1.0) / n_components(model)
    model.variances .= 1.0
    model.log_likelihood = T(-Inf)
    model.converged = false
    model.iterations = 0
    for comp in model.components
        randinit!(comp)
    end
end

function reinit!(model::MultivariateMixtureModel{T}) where T
    model.weights .= T(1.0) / n_components(model)
    model.variances .= 1.0
    model.log_likelihood = T(-Inf)
    model.converged = false
    model.iterations = 0
    for comp_dict in model.components
        for comp in values(comp_dict)
            randinit!(comp)
        end
    end
end

function duplicate(model::UnivariateMixtureModel{T}) where T
    new_components = [deepcopy(comp) for comp in model.components]
    return UnivariateMixtureModel{T}(new_components, model.weights, model.variances, model.log_likelihood, model.converged, model.iterations)
end

function duplicate(model::MultivariateMixtureModel{T}) where T
    new_components = [ComponentDict{T}(deepcopy(comp_dict)) for comp_dict in model.components]
    return MultivariateMixtureModel{T}(new_components, model.weights, model.variances, model.log_likelihood, model.converged, model.iterations)
end

function log_likelihoods(model::UnivariateMixtureModel{T}, X::UnivariateMixtureData) where T
    LLs = zeros(T, length(X.ids), n_components(model))

    for j in eachindex(X.ids)
        # Get the samples for this individual
        samples = X.grouped_view.data[X.ids[j]]
        timepoints = X.grouped_view.time[X.ids[j]]
        lls = [log_likelihood(model.components[k], timepoints, samples, model.variances[k]) + log(model.weights[k]) for k in 1:n_components(model)]
        LLs[j,:] .= lls
    end
    return LLs
end

function log_likelihoods(model::MultivariateMixtureModel{T}, X::MultivariateMixtureData) where T
    LLs = zeros(T, length(X.ids), n_components(model))

    for (i,var) in enumerate(X.variables)
        for j in eachindex(X.ids)
            # Get the samples for this individual and variable
            samples = X.grouped_view.data[(X.ids[j], var)]
            timepoints = X.grouped_view.time[(X.ids[j], var)]
            lls = [log_likelihood(model.components[k][var], timepoints, samples, model.variances[i,k]) + log(model.weights[k]) for k in 1:n_components(model)]
            LLs[j,:] .+= lls
        end
    end
    return LLs
end

"""
Predict the values at given timepoints for each component of the mixture model for a univariate mixture model.

## Arguments
- `model::UnivariateMixtureModel{T}`: The fitted mixture model.
- `timepoints::Vector{T}`: A vector of timepoints at which to predict

## Returns
A matrix of size `(length(timepoints), n_components)` where each column corresponds to the predictions from one component.
"""
function predict(model::UnivariateMixtureModel{T}, timepoints::AbstractVector{T}) where T
    n = length(timepoints)
    n_comp = n_components(model)
    preds = zeros(T, n, n_comp)
    for k in 1:n_comp
        preds[:,k] .= predict(model.components[k], timepoints)
    end
    return preds
end

"""
Predict the values at given timepoints for each component of the mixture model for a multivariate mixture model.

## Arguments
- `model::MultivariateMixtureModel{T}`: The fitted mixture model.
- `timepoints::Vector{T}`: A vector of timepoints at which to predict

## Returns
A dictionary where keys are variable names (as `Symbol`) and values are matrices of size `(length(timepoints), n_components)` where each column corresponds to the predictions from one component for that variable.
"""
function predict(model::MultivariateMixtureModel{T}, timepoints::AbstractVector{T}) where T
    n = length(timepoints)
    n_comp = n_components(model)
    preds = Dict{Symbol, Matrix{T}}()
    for (var, _) in model.components[1]
        preds[var] = zeros(T, n, n_comp)
        for k in 1:n_comp
            preds[var][:,k] .= predict(model.components[k][var], timepoints)
        end
    end
    return preds
end
