# ============================================================================
# Mixture Model Result
# ============================================================================

"""
    MixtureResult

Results from fitting a mixture model.
"""
struct MixtureResult
    component::Component
    n_clusters::Int
    parameters::Vector{Vector{Float64}}
    params_error::Vector{Vector{Float64}}
    cluster_probs::Vector{Float64}
    responsibilities::Matrix{Float64}
    loglikelihood::Float64
    converged::Bool
    n_iterations::Int
    error_model::ErrorModel
end

function predict(result::MixtureResult, t::Real, inputs=nothing)
    n_clusters = result.n_clusters
    y_preds = [predict(result.component, result.parameters[k], [t], inputs)[1, :] for k in 1:n_clusters]
    return y_preds
end

function predict(result::MixtureResult, t::AbstractVector, inputs=nothing)
    n_clusters = result.n_clusters
    y_preds = [predict(result.component, result.parameters[k], t, inputs) for k in 1:n_clusters]
    return y_preds
end

# ============================================================================
# E-Step: Compute Responsibilities
# ============================================================================

function component_loglikelihood(component::Component,
                                 params::AbstractVector{Float64},
                                 t::AbstractVector{Float64},
                                 y::AbstractMatrix{Float64},
                                 error_model::ErrorModel,
                                 component_params_error::Vector{Float64},
                                 inputs)
    y_pred = predict(component, params, t, inputs)
    ll = 0.0

    for j in axes(y, 2)
        residuals = y[:, j] - y_pred[:, j]
        ll += loglikelihood(error_model, residuals, component_params_error[j])
    end

    return ll
end

"""
    e_step!(responsibilities, data, component, parameters, probs, error_model, inputs)

E-step: Compute responsibilities for all subjects and clusters.
Updates responsibilities matrix in-place.
"""
function e_step!(responsibilities::Matrix{Float64},
                data::MixtureData,
                component::Component,
                parameters::Vector{Vector{Float64}},
                mixture_weights::Vector{Float64},
                error_model::ErrorModel,
                params_error::Vector{Vector{Float64}},
                inputs)
    
    n_clusters = length(parameters)
    log_probs = zeros(n_clusters)
    
    for subject in unique(data.ids)

        tv, yv = subset_view(data, subject)

        # find missing observations
        missing_mask = map(i -> any(ismissing, yv[i, :]), axes(yv, 1))
        tv = tv[.!missing_mask]
        yv = Float64.(yv[.!missing_mask, :])
               
        # Compute log probabilities for each cluster
        for k in 1:n_clusters
            log_probs[k] = log(mixture_weights[k]) + component_loglikelihood(
                component, parameters[k], tv, yv, error_model, params_error[k], inputs
            )
        end
        
        # Normalize using log-sum-exp trick
        max_log_prob = maximum(log_probs)
        for k in 1:n_clusters
            responsibilities[subject, k] = exp(log_probs[k] - max_log_prob)
        end
        
        # Normalize to sum to 1
        row_sum = sum(view(responsibilities, subject, :))
        for k in 1:n_clusters
            responsibilities[subject, k] /= row_sum
        end
    end
    
    return nothing
end

function posterior_responsibilities(result::MixtureResult, data::MixtureData)
    n_obs = length(unique(data.ids))
    n_clusters = result.n_clusters
    R = zeros(n_obs, n_clusters)

    e_step!(R, data, result.component, result.parameters, 
            result.cluster_probs, result.error_model, 
            result.params_error, nothing)

    return R
end

function posterior_responsibilities(result::MixtureResult, t::AbstractVector, y::AbstractMatrix, ids::AbstractVector; inputs=nothing)

    data = MixtureData(t, y, ids)

    return posterior_responsibilities(result, data)
end

# ============================================================================
# M-Step: Update Parameters
# ============================================================================

"""
    fit_weighted_optimized(component, data, responsibilities, cluster, error_model, 
                          weighted_data, inputs)

Fit weighted regression for a single cluster using pre-allocated buffers.
"""
function fit_weighted!(parameters, component::Component,
                    data::MixtureData,
                    responsibilities::Matrix{Float64},
                    k::Int,
                    inputs)

    Wv = [responsibilities[id, k] for id in data.ids]

    missing_mask = map(i -> any(ismissing, data.y[i, :]), axes(data.y, 1))
    tv = data.t[.!missing_mask]
    yv = Float64.(data.y[.!missing_mask, :])
    Wv = Wv[.!missing_mask]

    if sum(Wv) > n_parameters(component)
        # Fit polynomial with weights
        fit!(parameters, component, tv, yv[:, 1], Wv, inputs)
    else
        # If no samples assigned to this component, reinitialize by fitting to all data
        fit!(parameters, component, tv, yv[:, 1], inputs)
        
    end
end

function fit_weighted!(parameters, component::CompositeComponent,
                               data::MixtureData,
                               responsibilities::Matrix{Float64},
                               k::Int,
                               inputs)

    Wv = [responsibilities[id, k] for id in data.ids]
    # fit each sub-component separately

    for (comp, y_range, param_range) in zip(component.components, component.y_ranges, component.param_ranges)

        missing_mask = map(i -> any(ismissing.(data.y[i, y_range])), axes(data.y, 1))
        tv = data.t[.!missing_mask]
        yv = Float64.(data.y[.!missing_mask, y_range])
        Wv_c = Wv[.!missing_mask]

        #println(param_range)
        if sum(Wv_c) > n_parameters(comp)
            # Fit component with weights
            fit!(view(parameters, param_range), comp, tv, yv, Wv_c, inputs)
        else
            # If no samples assigned to this component, reinitialize by fitting to all data
            fit!(view(parameters, param_range), comp, tv, yv, inputs)
        end

    end

end

"""
    m_step!(parameters, probs, data, component, responsibilities, error_model, 
            weighted_data_buffers, inputs)

M-step: Update parameters and mixing proportions.
Updates parameters and probs in-place.
"""
function m_step!(parameters::Vector{Vector{Float64}},
                probs::AbstractVector{Float64},
                data::MixtureData,
                component::Component,
                responsibilities::Matrix{Float64},
                inputs)
    
    n_clusters = length(parameters)
    
    # Update mixing proportions
    @inbounds for k in 1:n_clusters
        probs[k] = mean(view(responsibilities, :, k))
    end
    
    # Update component parameters for each cluster
    @inbounds for k in 1:n_clusters
        fit_weighted!(
            parameters[k], component, data, responsibilities, k, inputs
        )
    end
    
    return nothing
end

# ============================================================================
# Compute Log-Likelihood
# ============================================================================

"""
    compute_total_loglikelihood(data, component, parameters, probs, error_model, inputs)

Compute total log-likelihood of the mixture model.
"""
function compute_total_loglikelihood(data::MixtureData,
                                    component::Component,
                                    parameters::Vector{Vector{Float64}},
                                    params_error::Vector{Vector{Float64}},
                                    probs::Vector{Float64},
                                    error_model::ErrorModel,
                                    inputs)
    loglik = 0.0
    n_clusters = length(parameters)
    
    for subject in unique(data.ids)

        # get subject data
        tv, yv = subset_view(data, subject)

        # find missing observations
        missing_mask = map(i -> any(ismissing, yv[i, :]), axes(yv, 1))
        tv = tv[.!missing_mask]
        yv = Float64.(yv[.!missing_mask, :])
        
        # Compute weighted sum over clusters
        prob_sum = 0.0
        for k in 1:n_clusters
            log_prob = log(probs[k])
            
            # Sum over measurements
            log_prob += component_loglikelihood(
                component, parameters[k], tv, yv, error_model, params_error[k], inputs
            )

            
            prob_sum += exp(log_prob)
        end
        
        loglik += log(prob_sum)
    end
    
    return loglik
end

"""
    row_normalize!(mat)

    Normalize each row of matrix `mat` in-place to sum to 1.
"""
function row_normalize!(mat::Matrix{Float64})
    @inbounds for i in axes(mat, 1)
        row_sum = sum(view(mat, i, :))
        for j in axes(mat, 2)
            mat[i, j] /= row_sum
        end
    end
    return nothing
end

"""
    cluster_assignments_to_responsibilities(assignments::AbstractVector{Int}, n_clusters::Int)

Convert hard cluster assignments to a responsibility matrix.

# Arguments
- `assignments`: Vector of cluster assignments (1 to n_clusters) for each subject
- `n_clusters`: Total number of clusters

# Returns
- Matrix of responsibilities where each row sums to 1, with 1.0 for the assigned cluster and 0.0 elsewhere
"""
function cluster_assignments_to_responsibilities(assignments::AbstractVector{Int}, n_clusters::Int)
    n_subjects = length(assignments)
    responsibilities = zeros(Float64, n_subjects, n_clusters)
    
    for (i, cluster) in enumerate(assignments)
        @assert 1 <= cluster <= n_clusters "Cluster assignments must be between 1 and $n_clusters"
        responsibilities[i, cluster] = 1.0
    end
    
    return responsibilities
end

# ============================================================================
# Main Fitting Function
# ============================================================================

function error_model_parameters(component::Component, error_model::NormalError, data::MixtureData, 
                          parameters::Vector{Vector{Float64}},
                          responsibilities::Matrix{Float64},
                          inputs)

    n_obs = size(data.y, 1)
    n_variables = size(data.y, 2)
    n_components = length(parameters)

    # get responsibilities per data point
    responsibilities_per_point = zeros(n_obs, n_components)
    for (id_idx, id) in enumerate(data.ids)
        for k in 1:n_components
            responsibilities_per_point[id_idx, k] = responsibilities[id, k]
        end
    end

    # Compute variance
    y_pred = [predict(component, p, data.t, inputs) for p in parameters]
    params_error = [zeros(n_variables) for _ in 1:n_components]
    for k in 1:n_components
        predictions = y_pred[k]
        for j in 1:n_variables
            missing_mask = .!ismissing.(data.y[:, j])

            residuals = data.y[missing_mask, j] - predictions[missing_mask, j]
            params_error[k][j] = variance(error_model, residuals, view(responsibilities_per_point, missing_mask, k))
        end
    end

    return params_error
end

function _fit_single_mixture(component::Component, n_components::Int, 
                     data::MixtureData,
                     error_model::ErrorModel,
                     inputs,
                     max_iter::Int,
                     tol::Float64,
                     verbose::Bool,
                     initial_responsibilities::Union{Nothing, Matrix{Float64}}=nothing)
    
    # Initialize parameters
    parameters = [initialize_parameters(component) for _ in 1:n_components]
    mixture_weights = ones(n_components) ./ n_components
    
    # Initialize responsibilities
    if initial_responsibilities === nothing
        # Random initialization
        responsibilities = rand(length(unique(data.ids)), n_components)
        row_normalize!(responsibilities)
    else
        # Use provided initial responsibilities
        responsibilities = copy(initial_responsibilities)
    end

    params_error = error_model_parameters(component, error_model, data, parameters, responsibilities, inputs)
    
    # EM iterations
    prev_loglik = -Inf
    converged = false
    iter = 0

    for iter in 1:max_iter
        # M-step: Update parameters based on current responsibilities
        m_step!(parameters, mixture_weights, data, component, responsibilities, inputs)

        # Update error model parameters (part of M-step)
        params_error = error_model_parameters(component, error_model, data, parameters, responsibilities, inputs)

        # Compute log-likelihood with updated parameters
        loglik = compute_total_loglikelihood(data, component, parameters, params_error,
                                            mixture_weights, error_model, inputs)
        
        # E-step: Update responsibilities for next iteration
        e_step!(responsibilities, data, component, parameters, 
                mixture_weights, error_model, params_error, inputs)
        
        # Check convergence
        if abs(loglik - prev_loglik) < tol
            converged = true
            verbose && println("Converged at iteration $iter")
            verbose && println("Final log-likelihood: $(round(loglik, digits=4))")
            return MixtureResult(
                component,
                n_components,
                parameters,
                params_error,
                mixture_weights,
                responsibilities,
                loglik,
                converged,
                iter,
                error_model
            )
        end
        
        prev_loglik = loglik
        
        if verbose && (iter % 10 == 0 || iter == 1)
            println("Iteration $iter: log-likelihood = $(round(loglik, digits=4))")
        end
    end
    
    if !converged && verbose
        println("Did not converge after $max_iter iterations")
        println("Final log-likelihood: $(round(prev_loglik, digits=4))")
    end
    
    return MixtureResult(
        component,
        n_components,
        parameters,
        params_error,
        mixture_weights,
        responsibilities,
        prev_loglik,
        converged,
        max_iter,
        error_model
    )
end

function _fit_mixtures(component::Component, n_components::Int, 
                     data::MixtureData,
                     error_model::ErrorModel,
                     inputs,
                     max_iter::Int,
                     tol::Float64,
                     verbose::Bool,
                     n_repeats::Int,
                     initial_responsibilities::Union{Nothing, Matrix{Float64}}=nothing)
    
    results = MixtureResult[]
    for repeat in 1:n_repeats
        verbose && println("Starting EM repeat $repeat/$n_repeats")
        # Only use initial responsibilities for the first repeat
        init_resps = (repeat == 1) ? initial_responsibilities : nothing
        result = _fit_single_mixture(
            component, n_components, data, error_model, inputs, max_iter, tol, verbose, init_resps
        )
        push!(results, result)
    end

    # Select best result based on log-likelihood
    best_result = argmax(r -> r.loglikelihood, results)
    return best_result
end


"""
    fit_mixture(component, n_components, t, y, ids;
                n_repeats=5, error_model=NormalError(), inputs=nothing, 
                max_iter=100, tol=1e-6, verbose=true, initial_assignments=nothing)

Fit a mixture model using the Expectation-Maximization (EM) algorithm. By default, `fit_mixture` will run 5 EM restarts and return the best fitting model based on log-likelihood.

# Arguments
- `component`: A Component model (e.g., PolynomialRegression(2))
- `n_components`: Number of mixture components/clusters
- `t`: Time points (vector)
- `y`: Measurements (vector for single measurement, matrix for multiple)
- `ids`: Subject identifiers (vector)

# Keyword Arguments
- `n_repeats`: Number of EM restarts (default: 5)
- `error_model`: Error distribution (currently only NormalError() is supported, included for possible future extensions)
- `inputs`: Additional input variables to be passed to the component model (default: nothing)
- `max_iter`: Maximum EM iterations (default: 100)
- `tol`: Convergence tolerance (default: 1e-6)
- `verbose`: Print progress (default: true)
- `initial_assignments`: Optional vector of initial cluster assignments (1 to n_components) for each subject (default: nothing, uses random initialization)

# Returns
- `MixtureResult` containing fitted parameters and cluster assignments, with fields:
    - `component`: The component model used
    - `n_clusters`: Number of clusters
    - `parameters`: Fitted parameters for each cluster
    - `params_error`: Estimated error model parameters for each measurement and cluster
    - `cluster_probs`: Mixing proportions for each cluster
    - `responsibilities`: Posterior responsibilities for each subject and cluster
    - `loglikelihood`: Final log-likelihood of the fitted model
    - `converged`: Boolean indicating if the EM algorithm converged
    - `n_iterations`: Number of iterations performed
    - `error_model`: The error model used

# Examples
```julia
# Single measurement with normal errors
result = fit_mixture(PolynomialRegression(2), 3, t, y, ids)

# With initial cluster assignments
initial_clusters = [1, 1, 2, 2, 3, 3]  # subjects 1-2 in cluster 1, 3-4 in cluster 2, etc.
result = fit_mixture(PolynomialRegression(2), 3, t, y, ids; initial_assignments=initial_clusters)
```
"""
function fit_mixture(component::Component, n_components::Int, 
                     t::AbstractVector, y::AbstractMatrix, 
                     ids::AbstractVector;
                     n_repeats::Int=5,
                     error_model::ErrorModel=NormalError(),
                     inputs=nothing,
                     max_iter::Int=100,
                     tol::Float64=1e-6,
                     verbose::Bool=true,
                     initial_assignments::Union{Nothing, AbstractVector{Int}}=nothing)
    
    # Validate inputs
    n_obs, n_variables = size(y)
    @assert length(t) == n_obs "Length of t must match number of observations"
    @assert length(ids) == n_obs "Length of ids must match number of observations"
    
    data = MixtureData(t, y, ids)
    
    # Convert initial assignments to responsibilities if provided
    initial_responsibilities = nothing
    if initial_assignments !== nothing
        n_subjects = length(unique(ids))
        @assert length(initial_assignments) == n_subjects "initial_assignments must have one entry per unique subject (got $(length(initial_assignments)), expected $n_subjects)"
        initial_responsibilities = cluster_assignments_to_responsibilities(initial_assignments, n_components)
    end
    
    return _fit_mixtures(
        component, n_components, data, error_model, inputs, 
        max_iter, tol, verbose, n_repeats, initial_responsibilities
    )
end

function fit_mixture(component::Component, n_components::Int, 
                     t::AbstractVector, y::AbstractVector, 
                     ids::AbstractVector;
                     n_repeats::Int=5,
                     error_model::ErrorModel=NormalError(),
                     inputs=nothing,
                     max_iter::Int=100,
                     tol::Float64=1e-6,
                     verbose::Bool=true,
                     initial_assignments::Union{Nothing, AbstractVector{Int}}=nothing)
    # Convert y to matrix
    y_matrix = reshape(y, length(y), 1)
    return fit_mixture(component, n_components, t, y_matrix, ids;
                       n_repeats=n_repeats,
                       error_model=error_model,
                       inputs=inputs,
                       max_iter=max_iter,
                       tol=tol,
                       verbose=verbose,
                       initial_assignments=initial_assignments)
end