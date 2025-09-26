# Multivariate Model
In this section, we will demonstrate how to fit a multivariate temporal mixture model using the `TemporalMixtureModels.jl` package. We will use synthetic data for this example.

## Prerequisites
While the package uses an internal data structure for the time series data, the package by default implements automatic conversion from a `DataFrame` to its internal format. 
```@example multivariate
using TemporalMixtureModels, DataFrames
```

## Generating Synthetic Data
We will generate some synthetic multivariate time series data. The data will consist of two variables, each generated from different polynomial functions with added Gaussian noise. The input DataFrame should be a so-called "long" format, with columns for the individual ID, time points, variable names, and observed values.
```@example multivariate
# Set random seed for reproducibility
using Random
Random.seed!(1234)

# Generate time points
individuals_per_group = [20, 30]
t_values = 0:0.1:10
n_groups = 2

group_coefficients_y = [ [2.0, -0.5, 0.05],  # Group 1: y = 2 - 0.5*t + 0.05*t^2
                       [1.0, 0.3, -0.02] ] # Group 2: y = 1 + 0.3*t - 0.02*t^2
group_coefficients_z = [ [0.5, 0.4, -0.03],  # Group 1: z = 0.5 + 0.4*t - 0.03*t^2
                       [3.0, -0.2, 0.04] ] # Group 2: z = 3 - 0.2*t + 0.04*t^2

ids = Int[]
timepoints = Float64[]
measurements_y = Float64[]
measurements_z = Float64[]
for group in 1:n_groups
    id_start = sum(individuals_per_group[1:group-1]) + 1
    id_end = sum(individuals_per_group[1:group])
    for individual in id_start:id_end
        for t in t_values
            y = group_coefficients_y[group][1] + group_coefficients_y[group][2]*t + group_coefficients_y[group][3]*t^2 + randn()*0.25
            z = group_coefficients_z[group][1] + group_coefficients_z[group][2]*t + group_coefficients_z[group][3]*t^2 + randn()*0.25
            push!(ids, individual)
            push!(timepoints, t)
            push!(measurements_y, y)
            push!(measurements_z, z)
        end
    end
end

input_data = DataFrame(id = [ids; ids], time = [timepoints; timepoints], value = [measurements_y; measurements_z], var_name = repeat(["y", "z"], inner=length(ids)))
first(input_data, 5)  # Display the first 5 rows of the DataFrame
```

Here you can see the first few rows of the generated DataFrame. Each row corresponds to a measurement for a specific individual at a specific time point for one of the two variables.

## Fitting a Multivariate Temporal Mixture Model
Now that we have our synthetic multivariate data, we can fit a multivariate temporal mixture model using the `fit!` function. We will use polynomial regression models as the component models for this example. We will specify the number of components (clusters) we expect in the data.

We first define the model, specifying the number of components and the type of component models to use for each variable. In this case, we will use polynomial regression models of degree 2 for both variables.
```@example multivariate
# Define the number of components and the component models for each variable
n_components = 2
component_models = Dict(:y => PolynomialRegression(2), :z => PolynomialRegression(2))
model = MultivariateMixtureModel(n_components, component_models)
```

We can now fit the model to our data using the `fit!` function. This function takes the model and the input DataFrame as arguments and performs the fitting process.
```@example multivariate
# Fit the model to the data
fit!(model, input_data)
```

As with the univariate case, we can also estimate the uncertainty of the fitted model parameters using the `bootstrap_ci` function. This function performs bootstrap resampling to compute confidence intervals for the model parameters.
```@example multivariate
# Compute bootstrap confidence intervals
n_bootstrap = 100  # Number of bootstrap samples
confidence_intervals, _ = bootstrap_ci(model, input_data; n_bootstrap=n_bootstrap)
confidence_intervals
```

The output shows the confidence intervals for the parameters of each component model and variable in the mixture. We can use these intervals and the fitted model to visualize the results.

```@example multivariate
using CairoMakie

# Create a range of time points for prediction
t_pred = 0:0.1:10

# Prepare a figure
figure_mixture = let f = Figure()

    # Plot the fitted component models
    predictions = predict(model, t_pred)

    COMPONENT_COLORS = [colorant"#c74300", colorant"#008aa1"]

    for (axi, variable) in enumerate([:y, :z])
        ax = Axis(f[1, axi], xlabel="Time", ylabel="Value", title="Temporal Mixture, variable: $variable")
        # Plot the original data points
        scatter!(ax, input_data[input_data[!,:var_name] .== String(variable),:time], input_data[input_data[!,:var_name] .== String(variable),:value]; color=:gray, markersize=2, label="Data - $variable")
        for k in axes(predictions[variable], 2)
            lines!(ax, t_pred, predictions[variable][:, k]; 
            label="Component $k - $variable", color=COMPONENT_COLORS[k], linewidth=2)

            # get the confidence bounds. This is still a bit clunky because there is no API for this yet
            lower_bound_parameters = confidence_intervals[k][variable][:lower]
            upper_bound_parameters = confidence_intervals[k][variable][:upper]

            model_lb = PolynomialRegression(2, lower_bound_parameters)
            model_ub = PolynomialRegression(2, upper_bound_parameters)

            y_lower = predict(model_lb, t_pred)
            y_upper = predict(model_ub, t_pred)

            # Plot the confidence intervals as shaded areas
            band!(ax, t_pred, y_lower, y_upper; 
            color=(COMPONENT_COLORS[k], 0.3), label="Component $k - $variable")
        end
    end

    f
end
figure_mixture
```
