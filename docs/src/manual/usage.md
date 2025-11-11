# Basic Usage of TemporalMixtureModels.jl
The main functionality revolves around the `fit_mixture` function, which fits a temporal mixture model to the provided time series data, using the EM algorithm.

First, we need to import the package:
```@example usage
using TemporalMixtureModels
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

```@example usage
t, y, ids, class_labels = example_bp_data(;n_subjects_drug=21, n_subjects_placebo=22, n_timepoints=5)
```

!!! note "Missing Data"
    It may occur that some time series have missing values at certain time points. The current implementation of `fit_mixture` handles missing data by ignoring those specific observations during the fitting process. 

    This may for example also happen if you measure two variables that are not always both measured at the same time points. In that case, you can fill the missing values with `missing` and the fitting process will skip those values. `TemporalMixtureModels.jl` is designed to handle such cases gracefully.

## Fitting a Temporal Mixture Model
Once you have your data prepared, you can fit a temporal mixture model using the `fit_mixture` function. You need to specify the number of components (clusters) you want to fit, as well as the component model to use (e.g., `PolynomialRegression`). For now, we are interested in the systolic pressure only, so we will extract that column from `Y` for fitting.

```@example usage
component_model = PolynomialRegression(degree=2)
num_components = 2  # For example, we want to fit 2 clusters
model = fit_mixture(component_model, num_components, t, y[:, 1], ids)
```

This will fit a temporal mixture model with 2 polynomial regression components of degree 2 to the systolic blood pressure data. 

## Making Predictions
After fitting the model, you can make predictions for new time points using the `predict` function. You can specify the time points at which you want predictions, and the function will return the predicted values for each component.

```julia
new_time_points = 0:0.1:2.0
predictions = predict(model, new_time_points)
```

