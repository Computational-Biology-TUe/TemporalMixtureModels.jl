# Multiple Output Variables
TemporalMixtureModels.jl supports fitting mixture models to multivariate time series data with multiple output variables. Each output variable is modeled independently, within the same mixture component. This means that for each component in the mixture model, there is a separate set of parameters for each output variable, allowing the model to capture different temporal patterns for each variable, but also to capture different combinations of patterns across variables in separate mixture components.

```@example multivariate
using TemporalMixtureModels
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

## Fitting the Multivariate Temporal Mixture Model
Once you have defined your composite component model, you can fit the multivariate temporal mixture model using the `fit_mixture` function, just like in the univariate case. You need to specify the number of components (clusters) you want to fit, as well as the composite component model to use.

```@example multivariate
num_components = 2  # For example, we want to fit 2 clusters
model = fit_mixture(component_model, num_components, t, y, ids)
```

This will fit a temporal mixture model with 2 components, each consisting of two polynomial regression models (one for each output variable) of degree 2 to the multivariate blood pressure data. 
