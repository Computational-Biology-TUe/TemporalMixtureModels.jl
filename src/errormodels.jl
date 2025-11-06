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

"""
    PoissonError()

Poisson error model (for count data).
Predictions are exponentiated to ensure positive rates.
"""
struct PoissonError <: ErrorModel end

function loglikelihood(::PoissonError, y_obs::AbstractVector{Float64},
                       y_pred::AbstractVector{Float64})
    # Ensure positive rate parameters
    λ = max.(exp.(y_pred), 1e-10)
    ll = 0.0
    for i in 1:length(y_obs)
        # Poisson log-likelihood: y*log(λ) - λ - log(y!)
        ll += y_obs[i] * log(λ[i]) - λ[i] - loggamma(y_obs[i] + 1)
    end
    return ll
end

"""
    LogNormalError()

Log-normal error model (for positive continuous data).
"""
struct LogNormalError <: ErrorModel end

function loglikelihood(::LogNormalError, y_obs::AbstractVector{Float64},
                       y_pred::AbstractVector{Float64})
    # Log-transform observations and compute normal likelihood
    log_y_obs = log.(max.(y_obs, 1e-10))
    residuals = log_y_obs .- y_pred
    σ² = max(var(residuals), 1e-6)
    n = length(y_obs)
    # Add Jacobian correction for log transformation
    jacobian = -sum(log_y_obs)
    return -0.5 * sum(residuals.^2) / σ² - 0.5 * n * log(2π * σ²) + jacobian
end