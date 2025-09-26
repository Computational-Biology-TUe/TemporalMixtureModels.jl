# EM logic for solving the mixture models

# ============================================================
# General utility functions
# ============================================================

function total_loglikelihood(log_num::AbstractMatrix{T}) where T<:Real
    J = size(log_num, 1)
    ll = zero(T)
    @inbounds for j in 1:J
        row = view(log_num, j, :)
        ll += log(sum(exp, row .- maximum(row))) + maximum(row)
    end
    return ll
end

function responsibilites!(output::Matrix{T}, log_probs::Matrix{T}) where T
    @inbounds for i in 1:size(log_probs, 1)
        row = view(log_probs, i, :)
        m = maximum(row)
        denom = zero(T)
        @simd for j in 1:size(log_probs, 2)
            output[i, j] = exp(log_probs[i, j] - m)
            denom += output[i, j]
        end
        @simd for j in 1:size(log_probs, 2)
            output[i, j] /= denom
        end
    end
    return output
end

function responsibilites(log_probs::Matrix{T}) where T
    output = zeros(T, size(log_probs))
    @inbounds for i in 1:size(log_probs, 1)
        row = view(log_probs, i, :)
        m = maximum(row)
        denom = zero(T)
        @simd for j in 1:size(log_probs, 2)
            output[i, j] = exp(log_probs[i, j] - m)
            denom += output[i, j]
        end
        @simd for j in 1:size(log_probs, 2)
            output[i, j] /= denom
        end
    end
    return output
end

function get_id_idx(X::UnivariateMixtureData)
    id_to_index = Dict(id => idx for (idx, id) in enumerate(X.ids))
    return [id_to_index[x.id] for x in X.data]
end

function get_id_idx(X::MultivariateMixtureData)
    id_to_index = Dict(id => idx for (idx, id) in enumerate(X.ids))
    return Dict(
        var => [id_to_index[x.id] for x in X.data[var]]
    for var in X.variables)
end

function e_step!(Γ, model::AbstractMixtureModel, X::MixtureData)
    likelihoods = log_likelihoods(model, X)
    return total_loglikelihood(likelihoods), responsibilites!(Γ, likelihoods)
end

function class_probabilities(model::AbstractMixtureModel, df::DataFrame)
    X = _prepare_data(df)
    likelihoods = log_likelihoods(model, X)
    return responsibilites(likelihoods)
end

# ============================================================
# Univariate Mixture Models
# ============================================================
function init_em!(model::UnivariateMixtureModel, X::UnivariateMixtureData; rng::AbstractRNG=Random.default_rng())

    # randomly assign individuals to components
    assignments = rand(rng, 1:n_components(model), length(X.ids))

    # initialize coefficients for each component
    for k in 1:n_components(model)
        ids_in_component = X.ids[assignments .== k]
        
        ids_set = Set(ids_in_component)
        samples = filter(x -> x.id in ids_set, X.data)

        if length(samples) > model.components[k].degree + 1
            fit!(model.components[k], samples.t, samples.y)
        else
            randinit!(model.components[k], rng)  # Random initialization if not enough samples
        end
        # initialize variances
        model.variances[k] = variance(model.components[k], samples.t, samples.y)
    end
end

function m_step!(model::UnivariateMixtureModel, X::UnivariateMixtureData, Γ, id_idx, n_k; rng=Random.default_rng())
    # update coefficients and variances
    for k in 1:n_components(model)
        Wv = [Γ[id_idx[i], k] for i in 1:length(X.data)]
        if n_k[k] > 1e-8
            # Fit polynomial with weights
            fit!(model.components[k], X.data.t, X.data.y, Wv)
        else
            # If no samples assigned to this component, reinitialize
            randinit!(model.components[k], rng)
        end
        model.variances[k] = variance(model.components[k], X.data.t, X.data.y)
    end
end

# ============================================================
# Multivariate Mixture Models
# ============================================================

function init_em!(model::MultivariateMixtureModel, X::MultivariateMixtureData; rng=Random.default_rng())

    # randomly assign individuals to components
    assignments = rand(rng, 1:n_components(model), length(X.ids))

    for (i, var) in enumerate(X.variables)
        # initialize coefficients for each component
        for k in 1:n_components(model)
            ids_in_component = X.ids[assignments .== k]
            ids_set = Set(ids_in_component)
            samples = filter(x -> x.id in ids_set, X.data[var])

            if length(samples) > model.components[k][var].degree + 1
                fit!(model.components[k][var], samples.t, samples.y)
            else
                randinit!(model.components[k][var], rng)  # Random initialization if not enough samples
            end
            # initialize variances
            model.variances[i,k] = variance(model.components[k][var], samples.t, samples.y)
        end
    end
end

function m_step!(model::MultivariateMixtureModel, X::MultivariateMixtureData, Γ, id_idx, n_k; rng=Random.default_rng())
    # update coefficients and variances
    for (i, var) in enumerate(X.variables)
        # initialize coefficients for each component
        for k in 1:n_components(model)
            samples = X.data[var]
            Wv = [Γ[id_idx[var][i], k] for i in 1:length(samples)]
            if n_k[k] > 1e-8
                # Fit polynomial with weights
                fit!(model.components[k][var], samples.t, samples.y, Wv)
            else
                # If no samples assigned to this component, reinitialize
                randinit!(model.components[k][var], rng)
            end
            model.variances[i, k] = variance(model.components[k][var], samples.t, samples.y)
        end
    end
end

# ============================================================
# Main fit function
# ============================================================

function fit!(model::AbstractMixtureModel, X::MixtureData; rng::AbstractRNG=Random.default_rng(), verbose::Bool=true, max_iter::Int=100, tol::Real=1e-6, hard_assignment::Bool=true)

    init_em!(model, X; rng=rng)
    id_idx = get_id_idx(X)
    Γ = zeros(length(X.ids), n_components(model))

    for it in 1:max_iter
        # E-step
        LL, Γ = e_step!(Γ, model, X)

        if hard_assignment
            assignments = argmax(Γ, dims=2)
            Γ .= 0.0
            Γ[assignments] .= 1.0
        end

        # Check convergence
        if maximum(abs.(LL - model.log_likelihood)) < tol
            model.log_likelihood = LL
            model.converged = true
            model.iterations = it
            if verbose
                println("Converged after $it iterations with log-likelihood: $LL")
            end

            return model, variances
        end
        model.log_likelihood = LL

        # update mixing weights
        n_k = sum(Γ, dims=1)[:]
        model.weights = n_k ./ length(X.ids)

        # M-step
        m_step!(model, X, Γ, id_idx, n_k; rng=rng)
    end
    model.iterations = max_iter
    if verbose
        println("Reached maximum iterations ($max_iter) without convergence. Final log-likelihood: $model.log_likelihood")
    end

    return model

end

"""
Fit a mixture model to the given DataFrame.

## Arguments
- `model::AbstractMixtureModel`: The mixture model to fit.
- `df::DataFrame`: The input data.

## Example
```julia
model = UnivariateMixtureModel(2, PolynomialRegression(2))
fit!(model, df)
```
"""
function fit!(model::AbstractMixtureModel, df::DataFrame;
    id_col = "id", time_col = "time", value_col = "value", var_name_col = "var_name", kwargs...)
    X = _prepare_data(df, id_col=id_col, time_col=time_col, value_col=value_col, var_name_col=var_name_col)
    return fit!(model, X; kwargs...)
end