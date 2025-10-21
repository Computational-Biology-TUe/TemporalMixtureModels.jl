# Fitting Models and Estimating Uncertainty
The primary function for fitting temporal mixture models is `fit!`. This function takes a mixture model and a dataset as input and performs the fitting process using the Expectation-Maximization (EM) algorithm. The function modifies the input model in place, updating its parameters to best fit the data.

```@docs
fit!
```

## Additional keyword arguments used by `fit!`

- `rng`: An optional random number generator for reproducibility. Default is `Random.default_rng()`.
- `verbose`: A boolean flag to control the verbosity of the fitting process. Default is `true`.
- `max_iter`: The maximum number of iterations for the EM algorithm. Default is `100`.
- `tol`: The tolerance for convergence. The fitting process stops when the change in log-likelihood is less than this value. Default is `1e-6`.
- `hard_assignment`: A boolean flag indicating whether to use hard assignments (True) or soft assignments (False) during the E-step of the EM algorithm. Default is `false`.

## Evaluating Model Fit
To evaluate the fit of a temporal mixture model, the package provides functions to compute the log-likelihoods and posterior responsibilities.

```@docs
log_likelihood
```
    
```@docs
posterior_responsibilities
```

## Bootstrap Confidence Intervals
To estimate the uncertainty of the fitted model parameters, the package provides a `bootstrap_ci` function. This function performs bootstrap resampling to compute confidence intervals for the model parameters.

```@docs
bootstrap_ci
```
