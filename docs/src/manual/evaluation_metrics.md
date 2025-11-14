# Evaluation Metrics
Within the package, three primary evaluation metrics are implemented to assess the performance of fitted temporal mixture models: Akaike Information Criterion (AIC), Bayesian Information Criterion (BIC), and Log-Likelihood. These metrics provide insights into the goodness-of-fit of the models while accounting for model complexity.

```@docs
loglikelihood(::MixtureResult, ::AbstractVector{T}, ::AbstractMatrix{Y}, ::AbstractVector{Int}; ::Any) where {T<:Real, Y<:Union{Real, Missing}}
```

```@docs
aic
```

```@docs
bic
```