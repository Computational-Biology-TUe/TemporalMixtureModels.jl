abstract type AbstractMixtureModel{T<:Real} end
const ComponentDict{T} = Dict{Symbol, AbstractMixtureModelComponent{T}}

mutable struct UnivariateMixtureModel{T<:Real} <: AbstractMixtureModel{T}
    components::Vector{AbstractMixtureModelComponent{T}}
    weights::Vector{T}  # mixture weights, should sum to 1
    log_likelihood::T
    converged::Bool
    iterations::Int
end

function UnivariateMixtureModel(n_components::Int, component::AbstractMixtureModelComponent{T}) where T
    components = [deepcopy(component) for _ in 1:n_components]
    return UnivariateMixtureModel{T}(components, fill(one(T) / n_components, n_components), -Inf, false, 0)
end

mutable struct MultivariateMixtureModel{T<:Real} <: AbstractMixtureModel{T}
    components::Vector{ComponentDict{T}}  # each row is a component, each column a variable
    weights::Vector{T}  # mixture weights, should sum to 1
    log_likelihood::T
    converged::Bool
    iterations::Int
end

function MultivariateMixtureModel(n_components::Int, components::Dict{Symbol, <:AbstractMixtureModelComponent{T}}) where T
    components = [ComponentDict{T}(deepcopy(components)) for _ in 1:n_components]
    return MultivariateMixtureModel{T}(components, fill(one(T) / n_components, n_components), -Inf, false, 0)
end

function n_components(model::AbstractMixtureModel)
    return length(model.components)
end

function reinit!(model::UnivariateMixtureModel{T}) where T
    model.weights .= T(1.0) / n_components(model)
    model.log_likelihood = T(-Inf)
    model.converged = false
    model.iterations = 0
    for comp in model.components
        randinit!(comp)
    end
end

function reinit!(model::MultivariateMixtureModel{T}) where T
    model.weights .= T(1.0) / n_components(model)
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
    return UnivariateMixtureModel{T}(new_components, model.weights, model.log_likelihood, model.converged, model.iterations)
end

function duplicate(model::MultivariateMixtureModel{T}) where T
    new_components = [ComponentDict{T}(deepcopy(comp_dict)) for comp_dict in model.components]
    return MultivariateMixtureModel{T}(new_components, model.weights, model.log_likelihood, model.converged, model.iterations)
end

function log_likelihoods(model::UnivariateMixtureModel{T}, X::UnivariateMixtureData, variances) where T
    LLs = zeros(T, length(X.ids), n_components(model))

    for j in eachindex(X.ids)
        # Get the samples for this individual
        samples = X.grouped_view.data[X.ids[j]]
        timepoints = X.grouped_view.time[X.ids[j]]
        lls = [log_likelihood(model.components[k], timepoints, samples, variances[k]) + log(model.weights[k]) for k in 1:n_components(model)]
        LLs[j,:] .= lls
    end
    return LLs
end

function log_likelihoods(model::MultivariateMixtureModel{T}, X::MultivariateMixtureData, variances) where T
    LLs = zeros(T, length(X.ids), n_components(model))

    for (i,var) in enumerate(X.variables)
        for j in eachindex(X.ids)
            # Get the samples for this individual and variable
            samples = X.grouped_view.data[(X.ids[j], var)]
            timepoints = X.grouped_view.time[(X.ids[j], var)]
            lls = [log_likelihood(model.components[k][var], timepoints, samples, variances[i,k]) + log(model.weights[k]) for k in 1:n_components(model)]
            LLs[j,:] .+= lls
        end
    end
    return LLs
end

function predict(model::UnivariateMixtureModel{T}, timepoints::Vector{T}) where T
    n = length(timepoints)
    n_comp = n_components(model)
    preds = zeros(T, n, n_comp)
    for k in 1:n_comp
        preds[:,k] .= predict(model.components[k], timepoints)
    end
    return preds
end

function predict(model::MultivariateMixtureModel{T}, timepoints::Vector{T}) where T
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
