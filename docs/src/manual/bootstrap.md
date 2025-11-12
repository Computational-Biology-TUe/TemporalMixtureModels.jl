# Bootstrapping for Uncertainty Estimation
TemporalMixtureModels.jl provides functionality to perform bootstrapping to estimate the uncertainty of the fitted model parameters. Bootstrapping involves resampling the data with replacement and refitting the model multiple times to obtain a distribution of parameter estimates.

## Performing Bootstrapping
To perform bootstrapping with TemporalMixtureModels.jl, you can use the `bootstrap` function. This function works in the same way as the `fit_mixture` function, but it takes an additional argument specifying the number of bootstrap samples to generate.

```@docs 
bootstrap
```