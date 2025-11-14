# Home

!!! compat "Version 1.0 Notice"
    The upgrade from version 0.x to 1.x includes a complete redesign of the API. It is recommended to upgrade to the latest version to benefit from new features and improvements.

TemporalMixtureModels.jl is a small Julia package for fitting temporal mixture models to cluster time series data. The package supports both univariate and independent multivariate time series data. The package provides a simple API for fitting models, making predictions, and estimating uncertainty using bootstrap methods. 

## Installation
Installation is straightforward in Julia:

```julia
pkg> add TemporalMixtureModels
```

## Temporal Mixture Models
A temporal mixture model is a probabilistic model that assumes that the observed time series data is generated from a mixture of several underlying temporal processes. Each process is represented by a component model, and the overall model combines these components to explain the observed data. Temporal mixture models are particularly useful for clustering time series data, as they can capture the underlying patterns and variations in the data. The basic form of the temporal mixture model is

```math
y_i(t) \sim \sum_{k=1}^{K} \pi_k f_k(y_i(t) | x_t, \theta_k)
```

Here, ``f(y_i(t) | x_t, \theta_k)`` represents the component likelihood for the k-th component model, which describes the distribution of the observed value ``y_i(t)`` at time ``t`` given the input variable ``x_t`` and parameters ``\theta_k``. The mixture weights ``\pi_k`` indicate the contribution of each component to the overall model, with the constraint that they sum to 1 (``\sum_{k=1}^{K} \pi_k = 1``). The number of components ``K`` is a hyperparameter that can be specified by the user. 

In this package, the component likelihood is always based on a Gaussian zero-mean noise model, and the component models can be any regression model that fits within this framework. The main requirement for a component model is that it has a single explanatory variable (time) and produces a single output variable.

For multivariate time series, the model assumes independence between each variable, leading to the following formulation:

```math
y_{i}(t) \sim \sum_{k=1}^{K} \pi_k \prod_{j=1}^J f_{k,j}(y_{i,j}(t) | x_t, \theta_{k,j})
```

Temporal mixtures work exceptionally well when the time series are aligned, meaning that they share a common time axis and have similar lengths. This alignment allows the model to effectively learn the temporal patterns and relationships within the data. For unaligned time series, additional preprocessing steps such as dynamic time warping or interpolation may be necessary to align the data before applying temporal mixture models. 

While it is possible to implement custom component models, the package currently includes the following built-in models:
- Polynomial regression
- Ridge (L2) regression
- Lasso (L1) regression

## Notice
The package is still wildly in development. Breaking changes will come.

> *“Time, the devourer of all things,  
> and you, envious age, together you destroy all that is.”*  
> — Ovid, *Metamorphoses*
