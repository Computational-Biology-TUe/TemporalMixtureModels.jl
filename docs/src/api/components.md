# Model Components

## Polynomial Regression
The `PolynomialRegression` model fits a polynomial of a specified degree to the time series data.

```@docs
PolynomialRegression(degree::Int)
```

## Ridge Regression
The `Ridge` model fits a polynomial regression with L2 regularization (ridge regression)

```@docs
RidgePolynomialRegression(degree::Int, lambda)
```

## Lasso Regression
The `Lasso` model fits a polynomial regression with L1 regularization (lasso regression)
```@docs
LassoPolynomialRegression(degree::Int, lambda)
```

## Custom Models
Custom model components can be implemented by subtyping the `AbstractMixtureModelComponent` and implementing the required methods. This allows for flexibility in defining new model types and behaviors tailored to specific use cases. See the tutorial on [Implementing Custom Components](@ref) for how to create your own model components.