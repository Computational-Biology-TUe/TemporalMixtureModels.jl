
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

### In case of a multivariate mixture model
- `ci_results::Vector{Dict{Symbol, Any}}`: A vector where each element corresponds to a component and contains a dictionary. Each dictionary has variable names as keys and values are tuples with keys `:lower` and `:upper` for the confidence intervals of the coefficients for that variable.
- `component_samples::Vector{Dict{Symbol, Vector{Vector{T}}}}`: A vector where each element corresponds to a component and contains a dictionary. Each dictionary has variable names as keys and values are vectors of coefficient samples from the bootstrap resampling for that variable.
"""
function bootstrap_ci(model::AbstractMixtureModel{T}, df::DataFrame; n_bootstrap::Int=100, alpha::Float64=0.05, rng::AbstractRNG=Random.GLOBAL_RNG, prog::Bool = true) where T
    _run_bootstrap_ci(model, df; n_bootstrap=n_bootstrap, alpha=alpha, rng=rng, prog=prog)
end

function _run_bootstrap_ci(model::UnivariateMixtureModel{T}, df::DataFrame; n_bootstrap::Int=100, alpha::Float64=0.05, rng::AbstractRNG=Random.GLOBAL_RNG, prog::Bool = true) where T
    X = _prepare_data(df)
    n = length(X.ids)
    n_components = length(model.components)
    component_samples = [Vector{Vector{T}}() for _ in 1:n_components]
    if prog
        p = Progress(n_bootstrap; desc="Running bootstrap resampling...")
    end
    for b in 1:n_bootstrap
        sample_indices = rand(rng, 1:n, n)
        X_sample = get_subset_with_replacement(X, sample_indices)
        model_b = duplicate(model)
        reinit!(model_b)
        fit!(model_b, X_sample; verbose=false, max_iter=100, tol=1e-6, hard_assignment=false)
         # perform Hungarian algorithm to match components based on all coefficients of the original model to prevent label switching
        match_components!(model_b, model)
        for k in 1:n_components
            push!(component_samples[k], model_b.components[k].coefficients)
        end
        if prog
            next!(p)
        end

    end
    ci_results = []
    for k in 1:n_components
        coeffs_matrix = hcat(component_samples[k]...)'
        lower_bounds = mapslices(x -> quantile(x, alpha / 2), coeffs_matrix; dims=1)
        upper_bounds = mapslices(x -> quantile(x, 1 - alpha / 2), coeffs_matrix; dims=1)
        push!(ci_results, (lower=vec(lower_bounds), upper=vec(upper_bounds)))
    end
    return ci_results, component_samples
end

function match_components!(model::UnivariateMixtureModel{T}, reference::UnivariateMixtureModel{T}) where T
    n = length(model.components)
    cost_matrix = zeros(T, n, n)
    for i in 1:n
        for j in 1:n
            cost_matrix[i, j] = sum((model.components[i].coefficients .- reference.components[j].coefficients).^2)
        end
    end
    assignment, cost = hungarian(cost_matrix)
    col_ind = [assignment[i] for i in 1:n]
    model.components = model.components[col_ind]
    model.weights = model.weights[col_ind]
    return model
end

function match_components!(model::MultivariateMixtureModel{T}, reference::MultivariateMixtureModel{T}) where T
    n = length(model.components)
    cost_matrix = zeros(T, n, n)
    for i in 1:n
        for j in 1:n
            total_cost = zero(T)
            for var in keys(model.components[i])
                if haskey(reference.components[j], var)
                    total_cost += sum((model.components[i][var].coefficients .- reference.components[j][var].coefficients).^2)
                else
                    total_cost += Inf
                end
            end
            cost_matrix[i, j] = total_cost
        end
    end
    assignment, cost = hungarian(cost_matrix)
    col_ind = [assignment[i] for i in 1:n]
    model.components = model.components[col_ind]
    model.weights = model.weights[col_ind]
    return model
end

function _run_bootstrap_ci(model::MultivariateMixtureModel{T}, df::DataFrame; n_bootstrap::Int=100, alpha::Float64=0.05, rng::AbstractRNG=Random.GLOBAL_RNG, prog::Bool = true) where T
    X = _prepare_data(df)
    n = length(X.ids)
    n_components = length(model.components)
    component_samples = [Dict{Symbol, Vector{Vector{T}}}() for _ in 1:n_components]
    if prog
        p = Progress(n_bootstrap; desc="Running bootstrap resampling...")
    end
    for b in 1:n_bootstrap
        sample_indices = rand(rng, 1:n, n)
        X_sample = get_subset_with_replacement(X, sample_indices)
        model_b = duplicate(model)
        reinit!(model_b)
        fit!(model_b, X_sample; verbose=false, max_iter=100, tol=1e-6, hard_assignment=false)
        match_components!(model_b, model)
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
    return ci_results, component_samples
end

