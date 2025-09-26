# Mixture Models
A mixture model is a probabilistic model that assumes that the observed data is generated from a mixture of several underlying distributions, each representing a different component of the data. In the context of temporal mixture models, each component is a temporal regression model that captures the underlying patterns in the time series data.

## Univariate Mixture Model
The `UnivariateMixtureModel` is designed for clustering univariate time series data. It consists of multiple component models, each representing a different cluster in the data.

```@docs
UnivariateMixtureModel
```

## Multivariate Mixture Model
The `MultivariateMixtureModel` is designed for clustering multivariate time series data. It consists of multiple component models, each representing a different cluster in the data. Each component model is a dictionary mapping variable names to their respective component models.

```@docs
MultivariateMixtureModel
```

## Predicting
Both `UnivariateMixtureModel` and `MultivariateMixtureModel` support making predictions at specified time points using the `predict` function.

```@docs
predict(model::UnivariateMixtureModel{T}, timepoints::AbstractVector{T}) where T
```

```@docs
predict(model::MultivariateMixtureModel{T}, timepoints::AbstractVector{T}) where T
```