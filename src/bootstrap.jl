"""
    bootstrap(component::Component, n_components::Int, n_bootstrap::Int, 
    t::AbstractVector, y::AbstractMatrix, ids::AbstractVector;
    n_repeats::Int=5,
    error_model::ErrorModel=NormalError(),
    inputs=nothing,
    max_iter::Int=100,
    tol::Float64=1e-6,
    separation_threshold::Float64=0.01,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    show_progress_bar::Bool=true)

Run bootstrap resampling to estimate confidence intervals for the coefficients of each component in a mixture model. The function fits the mixture model multiple times on bootstrap samples drawn with replacement from the original data. After fitting, it matches the components to the original fit to prevent label switching and collects the parameter estimates. Separation scores are computed to detect ambiguities in component matching, which may indicate unreliable confidence intervals. This can happen if the components aren't well separated, for example when too many components are specified.

# Arguments
- `component::Component`: The component model to use (e.g., `PolynomialRegression`).
- `n_components::Int`: Number of mixture components (clusters).
- `n_bootstrap::Int`: Number of bootstrap samples to draw.
- `t::AbstractVector`: Time points vector.
- `y::AbstractMatrix`: Observations matrix (rows: time points, columns: measurements).
- `ids::AbstractVector`: Subject IDs vector.

# Optional keyword arguments
- `n_repeats::Int=5`: Number of random initializations for fitting.
- `error_model::ErrorModel=NormalError()`: Error model to use. Only `NormalError` is currently supported.
- `inputs=nothing`: Additional inputs for the component model (if applicable).
- `max_iter::Int=100`: Maximum number of EM iterations.
- `tol::Float64=1e-6`: Convergence tolerance for the EM algorithm.
- `separation_threshold::Float64=0.01`: Threshold for detecting ambiguities in component matching.
- `rng::AbstractRNG=Random.GLOBAL_RNG`: Random number generator to use.
- `show_progress_bar::Bool=true`: Whether to show a progress bar.

# Returns
- `bootstrap_results::Vector{MixtureResult}`: A vector of `MixtureResult` objects from each bootstrap sample.
- `ambiguities_detected::Int`: The number of ambiguities detected during component matching in the bootstrap resampling. A high number may indicate unreliable confidence intervals due to uncertain sample assignment to components.
"""
function bootstrap(component::Component, n_components::Int, n_bootstrap::Int, t::AbstractVector, y::AbstractMatrix, ids::AbstractVector;
    n_repeats::Int=5,
    error_model::ErrorModel=NormalError(),
    inputs=nothing,
    max_iter::Int=100,
    tol::Float64=1e-6,
    separation_threshold::Float64=0.01,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    show_progress_bar::Bool=true)

    # prepare data
    data = MixtureData(t, y, ids)
    n = length(unique(ids))

    # fit initial model
    reference = _fit_mixtures(
        component, n_components, data, error_model, inputs, 
        max_iter, tol, false, n_repeats
    )

    bootstrap_results = Vector{MixtureResult}(undef, n_bootstrap)
    ambiguities_detected = 0

    if show_progress_bar
        prog_bar = Progress(n_bootstrap; desc="Running bootstrap resampling...")
    end

    for b in 1:n_bootstrap

        # sample with replacement
        sampled_data = sample_subset_with_replacement(data, n; rng=rng)

        # fit mixture model
        result = _fit_mixtures(
        component, n_components, sampled_data, error_model, inputs, 
        max_iter, tol, false, n_repeats)

        # match components to prevent label switching
        matched_result, separation = match_components(result, reference, data)
        if any(separation .< separation_threshold)
            ambiguities_detected += 1
        end

        bootstrap_results[b] = matched_result

        if show_progress_bar
            next!(prog_bar)
        end

    end

    if ambiguities_detected > 0
        @warn "$(ambiguities_detected) ambiguities detected in component matching during bootstrap resampling. If this number is high, individual component CIs may be unreliable, as sample assignment to components is highly uncertain."
    end

    return bootstrap_results, ambiguities_detected
end

function bootstrap(component::Component, n_components::Int, n_bootstrap::Int, t::AbstractVector, y::AbstractVector, ids::AbstractVector;
    n_repeats::Int=5,
    error_model::ErrorModel=NormalError(),
    inputs=nothing,
    max_iter::Int=100,
    tol::Float64=1e-6,
    separation_threshold::Float64=0.01,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    show_progress_bar::Bool=true)

    # Convert y to matrix
    y_matrix = reshape(y, length(y), 1)
    return bootstrap(
        component, n_components, n_bootstrap, t, y_matrix, ids;
        n_repeats=n_repeats,
        error_model=error_model,
        inputs=inputs,
        max_iter=max_iter,
        tol=tol,
        separation_threshold=separation_threshold,
        rng=rng,
        show_progress_bar=show_progress_bar
    )
end

function match_components(result::MixtureResult, reference::MixtureResult, data::MixtureData)

    # compute posterior responsibilities
    R_model = posterior_responsibilities(result, data)
    R_ref = posterior_responsibilities(reference, data)
    K = size(R_model, 2)

    cost = zeros(eltype(R_model), K, K)
    for k in 1:K, l in 1:K
        cost[k,l] = -sum(R_ref[:,k] .* R_model[:,l])
    end
    assignment, _ = hungarian(cost)
    ordering = [assignment[i] for i in 1:K]
    
    return MixtureResult(result.component, result.n_clusters, result.parameters[ordering], 
                         result.variances[:, ordering], result.cluster_probs[ordering],
                         result.responsibilities[:, ordering], result.loglikelihood,
                         result.converged, result.n_iterations, result.error_model), match_separation(cost)
end

function match_separation(cost::Matrix{T}) where T
    scores = zeros(T, size(cost, 1))
    for i in axes(cost, 1)
        best, second = -1 .* partialsort(cost[i, :], 1:2)
        scores[i] = best / (second + eps(T))
    end
    return scores .- T(1.0)
end


