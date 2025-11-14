# Basic Usage of TemporalMixtureModels.jl
The main functionality revolves around the `fit_mixture` function, which fits a temporal mixture model to the provided time series data, using the EM algorithm.

First, we need to import the package:
```@example usage
using TemporalMixtureModels, Random
Random.seed!(27052023)  # For reproducibility
```

## Data Preparation
First of all, you'll need some data that is in the right format. The basic format for the data consists of three elements:
- `t`: A vector of time points. This should be a one-dimensional array of real numbers representing the time axis for the time series data.
- `Y`: A vector or matrix of observed values. Each row corresponds to a time point in `t`, and each column corresponds to a different outcome variable (for multivariate time series). For univariate time series, this can be a one-dimensional array.
- `ids`: A vector of identifiers for each independent time series. The length of this vector should match the number of rows in `Y` and indicates which time series each observation belongs to.

Optionally, you can also provide:
- `inputs`: A matrix of input variables (covariates) that may influence the observed values. Each row corresponds to a time point, and each column corresponds to a different input variable. Inputs are currently not used in the built-in component models, but the support is there for custom models.

Say, we have a dataset with blood pressure measurements over time for multiple patients while they have been given either a placebo or a drug. Our table of data might look like this:

| id | time | systolic_pressure | diastolic_pressure | treatment |
|----|------|-------------------|--------------------|-----------|
| 1  | 0.0  | 120               | 80                 | drug      |
| 1  | 1.0  | 118               | 78                 | drug      |
| 1  | 2.0  | 115               | 75                 | drug      |
| 2  | 0.0  | 130               | 85                 | placebo   |
| 2  | 1.0  | 128               | 83                 | placebo   |
| 2  | 2.0  | 125               | 80                 | placebo   |
| ...| ...  | ...               | ...                | ...       |
| 43 | 2.0  | 110               | 70                 | placebo   |

For the examples in the manual, the package contains a function called `example_bp_data()` that can be used to generate a dataset like this.

```@docs
example_bp_data
```

Let's generate the example blood pressure dataset:
```@example usage
t, y, ids, class_labels = example_bp_data(;n_subjects_drug=21, n_subjects_placebo=22, n_timepoints=5)
```

We can quickly visualize the systolic blood pressure measurements over time for all subjects:

```@eval
using CairoMakie, Random
using TemporalMixtureModels: example_bp_data

Random.seed!(27052023)
t, y, ids, class_labels = example_bp_data(;n_subjects_drug=21, n_subjects_placebo=22, n_timepoints=5)
fig_bp = Figure(size=(350,250))
ax_bp = Axis(fig_bp[1, 1]; xlabel="Time (hours)", ylabel="Systolic BP (mmHg)")
for subject_id in unique(ids)
    mask = ids .== subject_id
    t_view = t[mask]
    y_view = y[mask, 1] # Systolic pressure is the first column
    label = class_labels[mask][1]
    descr = label == 1 ? "Drug" : "Placebo"
    lines!(ax_bp, t_view, y_view, label=descr, color=label == 1 ? Makie.wong_colors()[1] : Makie.wong_colors()[2], alpha=0.5)
end

axislegend(ax_bp; merge=true, position=:lb)

save("systolic_bp_plot.png", fig_bp)

nothing
```
![systolic_bp_plot](systolic_bp_plot.png)

!!! note "Missing Data"
    It may occur that some time series have missing values at certain time points. The current implementation of `fit_mixture` handles missing data by ignoring those specific observations during the fitting process. 

    This may for example also happen if you measure two variables that are not always both measured at the same time points. In that case, you can fill the missing values with `missing` and the fitting process will skip those values. `TemporalMixtureModels.jl` is designed to handle such cases gracefully.

    An example of such a dataset, where we may measure systolic pressure at the start and after 1 hour, while we diastolic pressure at the start and at 2 hours, would look like this:

    | id | time | systolic_pressure | diastolic_pressure | treatment |
    |----|------|-------------------|--------------------|-----------|
    | 1  | 0.0  | 120               | 80                 | drug      |
    | 1  | 1.0  | 118               | missing            | drug      |
    | 1  | 2.0  | missing           | 75                 | drug      |
    | 2  | 0.0  | 130               | 85                 | placebo   |
    | 2  | 1.0  | 128               | missing            | placebo   |
    | 2  | 2.0  | missing           | 80                 | placebo   |
    | ...| ...  | ...               | ...                | ...       |
    | 43 | 2.0  | missing           | 70                 | placebo   |

    As long as we have enough data points overall, the fitting process will still work.

## Fitting a Temporal Mixture Model
Once you have your data prepared, you can fit a temporal mixture model using the `fit_mixture` function. You need to specify the number of components (clusters) you want to fit, as well as the component model to use (e.g., `PolynomialRegression`). For now, we are interested in the systolic pressure only, so we will extract that column from `Y` for fitting.

```@example usage
component_model = PolynomialRegression(2)
num_components = 2  # For example, we want to fit 2 clusters
model = fit_mixture(component_model, num_components, t, y[:, 1], ids)
```

This will fit a temporal mixture model with 2 polynomial regression components of degree 2 to the systolic blood pressure data. 

### The Polynomial Regression Component
The `PolynomialRegression` component model fits a polynomial regression of a specified degree to the time series data. In this example, we used a polynomial of degree 2, which means that each component will fit a quadratic curve to the data. You can choose different degrees based on the complexity of the temporal patterns you expect in your data.
```@docs
PolynomialRegression
```

### Customizing the fitting process
The `fit_mixture` function provides several optional keyword arguments to customize the fitting process:
```@docs 
fit_mixture
```

## Making Predictions
After fitting the model, you can make predictions for new time points using the `predict` function. You can specify the time points at which you want predictions, and the function will return the predicted values for each component.

```julia
new_time_points = 0:0.1:2.0
predictions = predict(model, new_time_points)
```

An example of the resulting mixture model is shown below:

```@eval
using CairoMakie, Random
using TemporalMixtureModels

Random.seed!(27052023)
t, y, ids, class_labels = example_bp_data(;n_subjects_drug=21, n_subjects_placebo=22, n_timepoints=5)
component_model = PolynomialRegression(2)
num_components = 2
model = fit_mixture(component_model, num_components, t, y[:, 1], ids)

new_time_points = 0:0.1:5.0
predictions = predict(model, new_time_points)

figure_predictions = Figure(size=(350,250))
ax_predictions = Axis(figure_predictions[1, 1]; xlabel="Time (hours)", ylabel="Systolic BP (mmHg)")
colors = Makie.wong_colors()
scatter!(ax_predictions, t, y[:, 1]; color=:black, markersize=4, label="Data")
for k in 1:num_components
    lines!(ax_predictions, new_time_points, predictions[k]; color=colors[k],
              linewidth=2, label="Component $k")
end
axislegend(ax_predictions; position=:lb, merge=true)

save("systolic_bp_mixture_model.png", figure_predictions)
```

![systolic_bp_mixture_model](systolic_bp_mixture_model.png)
