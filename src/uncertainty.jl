
function bootstrap_ci(model::UnivariateMixtureModel{T}, df::DataFrame; n_bootstrap::Int=100, alpha::Float64=0.05, rng::AbstractRNG=Random.GLOBAL_RNG) where T
    X = _prepare_data(df)
    n = length(X.ids)
    n_components = length(model.components)
    component_samples = [Vector{Vector{T}}() for _ in 1:n_components]
    for b in 1:n_bootstrap
        sample_indices = rand(rng, 1:n, n)
        X_sample = get_subset_with_replacement(X, sample_indices)
        model_b = duplicate(model)
        reinit!(model_b)
        fit!(model_b, X_sample; verbose=false, max_iter=100, tol=1e-6, hard_assignment=false)
        for k in 1:n_components
            push!(component_samples[k], model_b.components[k].coefficients)
        end
    end
    ci_results = []
    for k in 1:n_components
        coeffs_matrix = hcat(component_samples[k]...)'
        lower_bounds = mapslices(x -> quantile(x, alpha / 2), coeffs_matrix; dims=1)
        upper_bounds = mapslices(x -> quantile(x, 1 - alpha / 2), coeffs_matrix; dims=1)
        push!(ci_results, (lower=vec(lower_bounds), upper=vec(upper_bounds)))
    end
    return ci_results
end

function bootstrap_ci(model::MultivariateMixtureModel{T}, df::DataFrame; n_bootstrap::Int=100, alpha::Float64=0.05, rng::AbstractRNG=Random.GLOBAL_RNG) where T
    X = _prepare_data(df)
    n = length(X.ids)
    n_components = length(model.components)
    component_samples = [Dict{Symbol, Vector{Vector{T}}}() for _ in 1:n_components]
    for b in 1:n_bootstrap
        sample_indices = rand(rng, 1:n, n)
        X_sample = get_subset_with_replacement(X, sample_indices)
        model_b = duplicate(model)
        reinit!(model_b)
        fit!(model_b, X_sample; verbose=false, max_iter=100, tol=1e-6, hard_assignment=false)
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
    return ci_results
end

