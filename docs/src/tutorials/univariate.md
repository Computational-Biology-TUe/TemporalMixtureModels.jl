# Univariate Model
This tutorial provides a step-by-step guide on how to use the `TemporalMixtureModels.jl` package to fit temporal mixture models to time series data. It provides a simple example using synthetic data to illustrate the main functionalities of the package, including data preparation, model fitting, prediction, and uncertainty estimation.

## Prerequisites
While the package uses an internal data structure for the time series data, the package by default implements automatic conversion from a `DataFrame` to its internal format. 
```@example univariate
using TemporalMixtureModels, DataFrames
```

## Generating Synthetic Data
For this tutorial, we will generate synthetic univariate time series data. The data will consist of three clusters, each generated from a different polynomial function with added Gaussian noise. The input DataFrame should be a so-called "long" format, with columns for the individual ID, time points, and observed values. 
```@example univariate
# Set random seed for reproducibility
using Random
Random.seed!(1234)

# Generate time points
individuals_per_group = [20, 30]
t_values = 0:0.1:10
n_groups = 2

group_coefficients = [ [2.0, -0.5, 0.05],  # Group 1: y = 2 - 0.5*t + 0.05*t^2
                       [1.0, 0.3, -0.02] ] # Group 2: y = 1 + 0.3*t - 0.02*t^2

ids = Int[]
timepoints = Float64[]
measurements = Float64[]
for group in 1:n_groups
    id_start = sum(individuals_per_group[1:group-1]) + 1
    id_end = sum(individuals_per_group[1:group])
    for individual in id_start:id_end
        for t in t_values
            y = group_coefficients[group][1] + group_coefficients[group][2]*t + group_coefficients[group][3]*t^2 + randn()*0.25
            push!(ids, individual)
            push!(timepoints, t)
            push!(measurements, y)
        end
    end
end

input_data = DataFrame(id = ids, time = timepoints, value = measurements)
first(input_data, 5)  # Display the first 5 rows of the DataFrame
```

Here you can see the first few rows of the generated DataFrame. Each row corresponds to a measurement for a specific individual at a specific time point. The internal structure of the model takes care of the bookkeeping during the fitting process.

## Fitting a Temporal Mixture Model
Now that we have our synthetic data, we can fit a temporal mixture model using the `fit!` function. We will use a polynomial regression model as the component model for this example. We will specify the number of components (clusters) we expect in the data.

We first define the model, specifying the number of components and the type of component model to use. In this case, we will use a polynomial regression model of degree 2.
```@example univariate
# Define the number of components and the component model
n_components = 2
component_model = PolynomialRegression(2)
model = UnivariateMixtureModel(n_components, component_model)
```

We can now fit the model to our data using the `fit!` function. This function takes the model and the input DataFrame as arguments and performs the fitting process.
```@example univariate
# Fit the model to the data
fit!(model, input_data)
```

We can also compute the confidence intervals for the model parameters using bootstrap resampling. This provides an estimate of the uncertainty in the parameter estimates.
```@example univariate
# Compute bootstrap confidence intervals
n_bootstrap = 100  # Number of bootstrap samples
confidence_intervals, _ = bootstrap_ci(model, input_data; n_bootstrap=n_bootstrap)
confidence_intervals
```

The output shows the confidence intervals for the parameters of each component model in the mixture. We can use these intervals and the fitted model to visualize the results.

```@example univariate
using CairoMakie

# Create a range of time points for prediction
t_pred = 0:0.1:10

# Prepare a figure
figure_mixture = let f = Figure()
    ax = Axis(f[1, 1], xlabel="Time", ylabel="Value", title="Temporal Mixture Model Fit")

    # Plot the original data points
    scatter!(ax, input_data.time, input_data.value; color=:gray, markersize=4, label="Data")

    # Plot the fitted component models
    predictions = predict(model, t_pred)

    COMPONENT_COLORS = [colorant"#c74300", colorant"#008aa1"]

    for k in axes(predictions, 2)
        lines!(ax, t_pred, predictions[:, k]; 
        label="Component $k", color=COMPONENT_COLORS[k], linewidth=2)

        # get the confidence bounds. This is still a bit clunky because there is no API for this yet
        lower_bound_parameters = confidence_intervals[k][:lower]
        upper_bound_parameters = confidence_intervals[k][:upper]

        model_lb = PolynomialRegression(2, lower_bound_parameters)
        model_ub = PolynomialRegression(2, upper_bound_parameters)

        y_lower = predict(model_lb, t_pred)
        y_upper = predict(model_ub, t_pred)

        # Plot the confidence intervals as shaded areas
        band!(ax, t_pred, y_lower, y_upper; 
        color=(COMPONENT_COLORS[k], 0.2), label="Component $k")
    end

    axislegend(ax; merge=true)

    f
end
figure_mixture
```
