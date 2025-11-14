# Using Custom Model Components
In addition to the built-in component models provided by the `TemporalMixtureModels.jl` package, you can also define your own custom component models to use within the temporal mixture modeling framework. This allows you to tailor the modeling approach to the specific characteristics of your data and research questions. In this section, an example custom component model is implemented to illustrate how to create and use custom components. In this example, we will create an `ExponentialDecay` component model that fits an exponential decay function to the time series data.

## Defining a Custom Component Model
To define a custom component model, you need to create a new struct that inherits from the `Component` abstract type. You also need to implement the required methods for the component model, including `initialize_parameters`, `n_parameters`, `predict`, and `fit!`.

### The ExponentialDecay Component
The `ExponentialDecay` component model fits an exponential decay function of the form:

```math
y(t) = A \cdot e^{-B \cdot t} + C
```

The parameters of the model are `A`, `B`, and `C`, which represent the initial value, decay rate, and offset, respectively.

```@example custom_components
using TemporalMixtureModels, LinearAlgebra, CairoMakie
import TemporalMixtureModels: Component, initialize_parameters, n_parameters, predict, fit!

struct ExponentialDecay <: Component
    # No additional fields needed for this simple model
end

function n_parameters(::ExponentialDecay)
    return 3  # A, B, C
end

function initialize_parameters(::ExponentialDecay)
    return randn(3) .* 0.1  # Small random initialization
end

function predict(::ExponentialDecay, params::AbstractVector, t::AbstractVector, inputs=nothing)
    A, B, C = params
    return A .* exp.(-B .* t) .+ C
end
```

These are quite straightforward implementations. The `n_parameters` function returns the number of parameters in the model, `initialize_parameters` provides an initial guess for the parameters, and `predict` computes the predicted values given the parameters and time points. 

### Model Fitting
We also need to implement the `fit!` method to estimate the parameters from the data. Here, we will use a linearization approach to fit the exponential decay model. We need both an unweighted and a weighted version of the `fit!` function.

```@example custom_components

function fit!(parameters::AbstractVector{T}, ::ExponentialDecay, t::AbstractVector{T}, y::AbstractVecOrMat{T}, ::Any) where T<:Real
    # Linearize the model: y - C = A * exp(-B * t)
    # Take logarithm: log(y - C) = log(A) - B * t
    # This requires y > C, so we need to estimate C first
    C_est = minimum(y) - 0.1  # Slightly below the minimum observed value
    y_adjusted = y .- C_est
    valid_mask = y_adjusted .> 0  # Only consider valid points for log
    
    if sum(valid_mask) < 2
        error("Not enough valid data points to fit ExponentialDecay model.")
    end
    
    log_y = log.(y_adjusted[valid_mask])
    t_valid = t[valid_mask]
    
    # Fit linear model: log_y = log(A) - B * t_valid
    X = hcat(ones(length(t_valid)), -t_valid)
    coeffs = X \ log_y
    
    A_est = exp(coeffs[1])
    B_est = coeffs[2]
    
    parameters[1] = A_est
    parameters[2] = B_est
    parameters[3] = C_est
end

function fit!(parameters::AbstractVector{T}, ::ExponentialDecay, t::AbstractVector{T}, y::AbstractVecOrMat{T}, w::AbstractVector{T}, ::Any) where T<:Real

    C_est = minimum(y) - 0.1  # Slightly below the minimum observed value
    y_adjusted = y .- C_est
    valid_mask = (y_adjusted .> 0) .& (w .> 0)  # Only consider valid points for log and positive weights
    if sum(valid_mask) < 2
        error("Not enough valid data points to fit ExponentialDecay model.")
    end

    log_y = log.(y_adjusted[valid_mask])
    t_valid = t[valid_mask]
    w_valid = w[valid_mask]

    # Fit weighted linear model: log_y = log(A) - B * t_valid
    X = hcat(ones(length(t_valid)), -t_valid)
    W = Diagonal(w_valid)
    coeffs = (X' * W * X) \ (X' * W * log_y)

    A_est = exp(coeffs[1])
    B_est = coeffs[2]
    parameters[1] = A_est
    parameters[2] = B_est
    parameters[3] = C_est

end
```

## Using the Custom Component Model
Once you have defined your custom component model, you can use it in the same way as the built-in component models. Here is an example of how to fit a temporal mixture model using the `ExponentialDecay` component:

```@example custom_components
using TemporalMixtureModels, Random
Random.seed!(27052023)  # For reproducibility

# Generate synthetic data
t, y, ids, class_labels = example_bp_data(;n_subjects_drug=21, n_subjects_placebo=22, n_timepoints=5)

# Define the custom component model
component_model = ExponentialDecay()
num_components = 2  # For example, we want to fit 2 clusters
model = fit_mixture(component_model, num_components, t, y[:, 1], ids)
```

This will fit a temporal mixture model with 2 `ExponentialDecay` components to the systolic blood pressure data. You can then make predictions and analyze the fitted model as usual.

We can use the fitted model to make predictions for new time points:

```@example custom_components
new_time_points = 0:0.1:5.0
predictions = predict(model, new_time_points)

figure_predictions = Figure(size=(350,250)) # hide
ax_predictions = Axis(figure_predictions[1, 1]; xlabel="Time (hours)", ylabel="Systolic BP (mmHg)") # hide
colors = Makie.wong_colors() # hide
scatter!(ax_predictions, t, y[:, 1]; color=:black, markersize=4, label="Data") # hide
for k in 1:num_components # hide
    lines!(ax_predictions, new_time_points, predictions[k]; color=colors[k], # hide
              linewidth=2, label="Component $k") # hide
end # hide
axislegend(ax_predictions; position=:lb, merge=true) # hide

save("systolic_bp_custom.png", figure_predictions) # hide
```

![systolic_bp_custom](systolic_bp_custom.png)