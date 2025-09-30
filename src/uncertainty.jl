
"""
Run bootstrap resampling to estimate confidence intervals for the coefficients of each component in a mixture model.

## Arguments
- `model::AbstractMixtureModel{T}`: The fitted mixture model
- `df::DataFrame`: The input DataFrame containing the data.

### Optional keyword arguments
- `n_bootstrap::Int=100`: Number of bootstrap samples to draw.
- `alpha::Float64=0.05`: Significance level for the confidence intervals (e.g., 0.05 for 95% CI).
- `rng::AbstractRNG=Random.GLOBAL_RNG`: Random number generator to use.
- `prog::Bool=true`: Whether to show a progress bar.

## Returns

### In case of a univariate mixture model
- `ci_results::Vector{Dict{Symbol, Any}}`: A vector where each element corresponds to a component and contains a dictionary with keys `:lower` and `:upper` for the confidence intervals of the coefficients.
- `component_samples::Vector{Vector{Vector{T}}}`: A vector where each element corresponds to a component and contains a vector of coefficient samples from the bootstrap resampling.
- `ambiguities_detected::Int`: The number of ambiguities detected during component matching in the bootstrap resampling. A high number may indicate unreliable confidence intervals due to uncertain sample assignment to components.

### In case of a multivariate mixture model
- `ci_results::Vector{Dict{Symbol, Any}}`: A vector where each element corresponds to a component and contains a dictionary. Each dictionary has variable names as keys and values are tuples with keys `:lower` and `:upper` for the confidence intervals of the coefficients for that variable.
- `component_samples::Vector{Dict{Symbol, Vector{Vector{T}}}}`: A vector where each element corresponds to a component and contains a dictionary. Each dictionary has variable names as keys and values are vectors of coefficient samples from the bootstrap resampling for that variable.
- `ambiguities_detected::Int`: The number of ambiguities detected during component matching in the bootstrap resampling. A high number may indicate unreliable confidence intervals due to uncertain sample assignment to components.
"""
function bootstrap_ci(model::AbstractMixtureModel{T}, df::DataFrame; n_bootstrap::Int=100, alpha::Float64=0.05, rng::AbstractRNG=Random.GLOBAL_RNG, prog::Bool = true) where T
    _run_bootstrap_ci(model, df; n_bootstrap=n_bootstrap, alpha=alpha, rng=rng, prog=prog)
end

function _run_bootstrap_ci(model::UnivariateMixtureModel{T}, df::DataFrame; n_bootstrap::Int=100, alpha::Float64=0.05, rng::AbstractRNG=Random.GLOBAL_RNG, prog::Bool = true, ambiguity_threshold = 0.01) where T
    X = _prepare_data(df)
    n = length(X.ids)
    n_components = length(model.components)
    component_samples = [Vector{Vector{T}}() for _ in 1:n_components]
    if prog
        p = Progress(n_bootstrap; desc="Running bootstrap resampling...")
    end
    ambiguities_detected = 0
    for b in 1:n_bootstrap
        sample_indices = rand(rng, 1:n, n)
        X_sample = get_subset_with_replacement(X, sample_indices)
        model_b = duplicate(model)
        reinit!(model_b)
        fit!(model_b, X_sample; verbose=false, max_iter=100, tol=1e-6, hard_assignment=false)
         # perform Hungarian algorithm to match components based on all coefficients of the original model to prevent label switching
        ambiguity = match_components!(model_b, model, X)
        if any(ambiguity .< ambiguity_threshold)
            ambiguities_detected += 1
        end
        for k in 1:n_components
            push!(component_samples[k], model_b.components[k].coefficients)
        end
        if prog
            next!(p)
        end

    end
    if ambiguities_detected > 0
        @warn "$(ambiguities_detected) ambiguities detected in component matching during bootstrap resampling. If this number is high, individual component CIs may be unreliable, as sample assignment to components is highly uncertain."
    end

    ci_results = []
    for k in 1:n_components
        coeffs_matrix = hcat(component_samples[k]...)'
        lower_bounds = mapslices(x -> quantile(x, alpha / 2), coeffs_matrix; dims=1)
        upper_bounds = mapslices(x -> quantile(x, 1 - alpha / 2), coeffs_matrix; dims=1)
        push!(ci_results, (lower=vec(lower_bounds), upper=vec(upper_bounds)))
    end
    return ci_results, component_samples, ambiguities_detected
end

function reorder!(model::UnivariateMixtureModel{T}, ordering::AbstractVector{Int}) where T
    model.components = model.components[ordering]
    model.weights = model.weights[ordering]
    model.variances = model.variances[ordering]
    return nothing
end

function reorder!(model::MultivariateMixtureModel{T}, ordering::AbstractVector{Int}) where T
    model.components = model.components[ordering]
    model.weights = model.weights[ordering]
    model.variances = model.variances[:, ordering]
    return nothing
end

function match_ambiguity(cost::Matrix{T}) where T
    scores = zeros(T, size(cost, 1))
    for i in axes(cost, 1)
        best, second = -1 .* partialsort(cost[i, :], 1:2)
        scores[i] = best / (second + eps(T))
    end
    return scores .- T(1.0)
end

function match_components!(model::AbstractMixtureModel{T}, reference::AbstractMixtureModel{T}, X::MixtureData) where T
    K = length(model.components)
    # compute posterior responsibilities
    R_model = posterior_responsibilities(model, X)
    R_ref = posterior_responsibilities(reference, X)

    cost = zeros(T, K, K)
    for k in 1:K, l in 1:K
        cost[k,l] = -sum(R_ref[:,k] .* R_model[:,l])
    end
    assignment, _ = hungarian(cost)
    col_ind = [assignment[i] for i in 1:K]
    reorder!(model, col_ind)
    return match_ambiguity(cost)
end

function _run_bootstrap_ci(model::MultivariateMixtureModel{T}, df::DataFrame; n_bootstrap::Int=100, alpha::Float64=0.05, rng::AbstractRNG=Random.GLOBAL_RNG, prog::Bool = true, ambiguity_threshold = 0.01) where T
    X = _prepare_data(df)
    n = length(X.ids)
    n_components = length(model.components)
    component_samples = [Dict{Symbol, Vector{Vector{T}}}() for _ in 1:n_components]
    if prog
        p = Progress(n_bootstrap; desc="Running bootstrap resampling...")
    end
    ambiguities_detected = 0
    for b in 1:n_bootstrap
        sample_indices = rand(rng, 1:n, n)
        X_sample = get_subset_with_replacement(X, sample_indices)
        model_b = duplicate(model)
        reinit!(model_b)
        fit!(model_b, X_sample; verbose=false, max_iter=100, tol=1e-6, hard_assignment=false)
        ambiguity = match_components!(model_b, model, X)
        if any(ambiguity .< ambiguity_threshold)
            ambiguities_detected += 1
        end
        # store coefficients for each variable separately to prevent mixup
        for k in 1:n_components
            comp_dict = model_b.components[k]
            # Assuming we want to store coefficients for each variable separately
            for (var, comp) in comp_dict
                if !haskey(component_samples[k], var)
                    component_samples[k][var] = Vector{Vector{T}}()
                end
                push!(component_samples[k][var], comp.coefficients)
            end
        end
        if prog
            next!(p)
        end
    end
    if ambiguities_detected > 0
        @warn "$(ambiguities_detected) ambiguities detected in component matching during bootstrap resampling. If this number is high, individual component CIs may be unreliable, as sample assignment to components is highly uncertain."
    end
    ci_results = []
    for k in 1:n_components
        var_ci = Dict{Symbol, Any}()
        for (var, samples) in component_samples[k]
            coeffs_matrix = hcat(samples...)'
            lower_bounds = mapslices(x -> quantile(x, alpha / 2), coeffs_matrix; dims=1)
            upper_bounds = mapslices(x -> quantile(x, 1 - alpha / 2), coeffs_matrix; dims=1)
            var_ci[var] = (lower=vec(lower_bounds), upper=vec(upper_bounds))
        end
        push!(ci_results, var_ci)
    end
    return ci_results, component_samples, ambiguities_detected
end

