# Multiple Output Variables
TemporalMixtureModels.jl supports fitting mixture models to multivariate time series data with multiple output variables. Each output variable is modeled independently, within the same mixture component. This means that for each component in the mixture model, there is a separate set of parameters for each output variable, allowing the model to capture different temporal patterns for each variable, but also to capture different combinations of patterns across variables in separate mixture components.

```@example multivariate
using TemporalMixtureModels, Random
Random.seed!(27052023)  # For reproducibility
```

## Data Preparation
When preparing your data for multivariate time series modeling, the format is similar to that of univariate time series, with the key difference being that the observed values `Y` should be a matrix where each column corresponds to a different output variable. Each row still corresponds to a time point in `t`, and the `ids` vector indicates which time series each observation belongs to. 

The example dataset provided by the `example_bp_data()` function includes two output variables: systolic and diastolic blood pressure measurements. You can use this function to generate a multivariate dataset as follows:

```@example multivariate
t, y, ids, class_labels = example_bp_data(;n_subjects_drug=21, n_subjects_placebo=22, n_timepoints=5)
```

## Defining a Composite Component Model
As the `PolynomialRegression` component only supports a single output variable, we need to define a composite component model that combines multiple `PolynomialRegression` models, one for each output variable. Creation of a composite component model is very straightforward using the `@component` macro.

```@example multivariate
component_model = @component begin
    y[1] ~ PolynomialRegression(2)
    y[2] ~ PolynomialRegression(2)
end
```

This composite model specifies that the first output variable (`y[1]`, systolic pressure) is modeled using a polynomial regression of degree 2, and the second output variable (`y[2]`, diastolic pressure) is also modeled using a polynomial regression of degree 2. The `@component` macro is sufficiently flexible to allow for different types of component models for each output variable if desired. 

```@docs
@component
```

## Fitting the Multivariate Temporal Mixture Model
Once you have defined your composite component model, you can fit the multivariate temporal mixture model using the `fit_mixture` function, just like in the univariate case. You need to specify the number of components (clusters) you want to fit, as well as the composite component model to use.

```@example multivariate
num_components = 2  # For example, we want to fit 2 clusters
model = fit_mixture(component_model, num_components, t, y, ids)
```

This will fit a temporal mixture model with 2 components, each consisting of two polynomial regression models (one for each output variable) of degree 2 to the multivariate blood pressure data. 

We can now use the fitted model to make predictions for new time points, just like in the univariate case. 

An example of the resulting mixture model is shown below:

```@eval
using CairoMakie, Random
using TemporalMixtureModels

Random.seed!(27052023)
t, y, ids, class_labels = example_bp_data(;n_subjects_drug=21, n_subjects_placebo=22, n_timepoints=5)
component_model = @component begin
    y[1] ~ PolynomialRegression(2)
    y[2] ~ PolynomialRegression(2)
end
num_components = 2
model = fit_mixture(component_model, num_components, t, y, ids)

new_time_points = 0:0.1:5.0
predictions = predict(model, new_time_points)

figure_predictions = Figure(size=(750, 350))
ax_predictions = Axis(figure_predictions[1, 1]; xlabel="Time (hours)", ylabel="Systolic BP (mmHg)")
ax_diastolic = Axis(figure_predictions[1, 2]; xlabel="Time (hours)", ylabel="Diastolic BP (mmHg)")
colors = Makie.wong_colors()
scatter!(ax_predictions, t, y[:, 1]; color=:black, markersize=4, label="Data")
scatter!(ax_diastolic, t, y[:, 2]; color=:black, markersize=4, label="Data")
for k in 1:num_components
    lines!(ax_predictions, new_time_points, predictions[k][:, 1]; color=colors[k],
              linewidth=2, label="Component $k")
    lines!(ax_diastolic, new_time_points, predictions[k][:, 2]; color=colors[k],
              linewidth=2, label="Component $k")
end
axislegend(ax_predictions; position=:lb, merge=true)

save("multivariate_mixture_model.png", figure_predictions)
```

![Multivariate Mixture Model](multivariate_mixture_model.png)