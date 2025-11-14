# Model evaluation metrics (AIC, BIC, AICc)

function total_free_parameters(result::MixtureResult)
    n_params = 0
    for (p_comp, p_err) in zip(result.parameters, result.params_error)
        n_params += length(p_comp)  # component parameters
        n_params += length(p_err)  # error model parameters
    end
    # mixture weights (K-1 free parameters)
    n_params += result.n_clusters - 1
    return n_params
end

function loglikelihood(result::MixtureResult, data::MixtureData; inputs=nothing)
    loglik = compute_total_loglikelihood(data, result.component, result.parameters, result.params_error,
                                            result.cluster_probs, result.error_model, inputs)
    return loglik
end

function aic(result::MixtureResult, data::MixtureData; inputs=nothing)
    k = total_free_parameters(result)
    ll = loglikelihood(result, data; inputs=inputs)
    return 2k - 2ll
end

function bic(result::MixtureResult, data::MixtureData; inputs=nothing)
    k = total_free_parameters(result)
    n = size(data.y, 1)
    ll = loglikelihood(result, data; inputs=inputs)
    return log(n)*k - 2ll
end

function loglikelihood(result::MixtureResult, t::AbstractVector{T}, y::AbstractVector{Y}, ids::AbstractVector{Int}; inputs=nothing) where {T<:Real, Y<:Union{Real, Missing}}
    return loglikelihood(result, MixtureData(t, reshape(y, :, 1), ids); inputs=inputs)
end

function aic(result::MixtureResult, t::AbstractVector{T}, y::AbstractVector{Y}, ids::AbstractVector{Int}; inputs=nothing) where {T<:Real, Y<:Union{Real, Missing}}  
    return aic(result, MixtureData(t, reshape(y, :, 1), ids); inputs=inputs)
end

function bic(result::MixtureResult, t::AbstractVector{T}, y::AbstractVector{Y}, ids::AbstractVector{Int}; inputs=nothing) where {T<:Real, Y<:Union{Real, Missing}}
    return bic(result, MixtureData(t, reshape(y, :, 1), ids); inputs=inputs)
end

"""
    loglikelihood(result::MixtureResult, t::AbstractVector{T}, y::AbstractMatrix{Y}, ids::AbstractVector{Int}; inputs=nothing) where {T<:Real, Y<:Union{Real, Missing}}

Compute the total log-likelihood of the fitted mixture model on the given data.

# Arguments
- `result::MixtureResult`: The result of fitting the mixture model.
- `t::AbstractVector`: Time points of the observations.
- `y::AbstractMatrix`: Observed data (vector or matrix).
- `ids::AbstractVector{Int}`: Cluster assignments for each observation.
- `inputs`: Optional additional inputs for prediction.

# Returns
- `Float64`: The total log-likelihood of the model on the data.
"""
function loglikelihood(result::MixtureResult, t::AbstractVector{T}, y::AbstractMatrix{Y}, ids::AbstractVector{Int};
                        inputs=nothing) where {T<:Real, Y<:Union{Real, Missing}}
    data = MixtureData(t, y, ids)
    return loglikelihood(result, data; inputs=inputs)
end

"""
    aic(result::MixtureResult, t::AbstractVector{T}, y::AbstractMatrix{Y}, ids::AbstractVector{Int}; inputs=nothing) where {T<:Real, Y<:Union{Real, Missing}}

Compute the Akaike information criterion of the fitted mixture model on the given data.

# Arguments
- `result::MixtureResult`: The result of fitting the mixture model.
- `t::AbstractVector`: Time points of the observations.
- `y::AbstractMatrix`: Observed data (vector or matrix).
- `ids::AbstractVector{Int}`: Cluster assignments for each observation.
- `inputs`: Optional additional inputs for prediction.

# Returns
- `Float64`: The Akaike information criterion of the model on the data.
"""
function aic(result::MixtureResult, t::AbstractVector{T}, y::AbstractMatrix{Y}, ids::AbstractVector{Int};
                inputs=nothing) where {T<:Real, Y<:Union{Real, Missing}}
    data = MixtureData(t, y, ids)
    return aic(result, data; inputs=inputs)
end


"""
    bic(result::MixtureResult, t::AbstractVector{T}, y::AbstractMatrix{Y}, ids::AbstractVector{Int}; inputs=nothing) where {T<:Real, Y<:Union{Real, Missing}}
    
Compute the Bayesian information criterion of the fitted mixture model on the given data.

# Arguments
- `result::MixtureResult`: The result of fitting the mixture model.
- `t::AbstractVector`: Time points of the observations.
- `y::AbstractMatrix`: Observed data (vector or matrix).
- `ids::AbstractVector{Int}`: Cluster assignments for each observation.
- `inputs`: Optional additional inputs for prediction.

# Returns
- `Float64`: The Bayesian information criterion of the model on the data.
"""
function bic(result::MixtureResult, t::AbstractVector{T}, y::AbstractMatrix{Y}, ids::AbstractVector{Int};
                inputs=nothing) where {T<:Real, Y<:Union{Real, Missing}}
    data = MixtureData(t, y, ids)
    return bic(result, data; inputs=inputs)
end