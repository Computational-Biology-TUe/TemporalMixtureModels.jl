# TemporalMixtureModels.jl
Temporal mixture models for aligned univariate and multivariate time series data. 

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://Computational-Biology-TUe.github.io/TemporalMixtureModels.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://Computational-Biology-TUe.github.io/TemporalMixtureModels.jl/dev)
 [![tests](https://github.com/Computational-Biology-TUe/TemporalMixtureModels.jl/actions/workflows/tests.yml/badge.svg)](https://github.com/Computational-Biology-TUe/TemporalMixtureModels.jl/actions/workflows/tests.yml)

## About
TemporalMixtureModels.jl is a small Julia package for fitting temporal mixture models to cluster time series data. The package supports both univariate and independent multivariate time series data. The package provides a simple API for fitting models, making predictions, and estimating uncertainty using bootstrap methods. A temporal mixture model is a probabilistic model that assumes that the observed time series data is generated from a mixture of several underlying temporal processes. Each process is represented by a component model, and the overall model combines these components to explain the observed data. Temporal mixture models are particularly useful for clustering time series data, as they can capture the underlying patterns and variations in the data.

## Notice
This package is still wildly in development. Breaking changes will come.

> *“Time, the devourer of all things,  
> and you, envious age, together you destroy all that is.”*  
> — Ovid, *Metamorphoses*

