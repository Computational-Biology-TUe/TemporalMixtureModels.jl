"""
Abstract type for error models.
Custom error models should subtype this and implement:
- `loglikelihood(model, y_obs, y_pred)`: compute log-likelihood
- `estimate_parameters(model, residuals)`: estimate distribution parameters
"""
abstract type ErrorModel end

# ============================================================================
# Error Models
# ============================================================================

"""
    NormalError()

Gaussian/Normal error model with estimated variance.
"""
struct NormalError <: ErrorModel end

function variance(::NormalError, residuals::AbstractVector{Float64}, responsibilities::AbstractVector{Float64})
    weighted_residuals = residuals .^ 2 .* responsibilities
    return sum(weighted_residuals) / (sum(responsibilities) + eps())
end

function loglikelihood(::NormalError, residuals::Vector{Float64}, variance::Float64)
    return Distributions.loglikelihood(Normal(0.0, sqrt(variance)), residuals)
end

function n_free_parameters(::NormalError)
    return 1  # variance
end