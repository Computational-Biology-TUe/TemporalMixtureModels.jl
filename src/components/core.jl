"""
Abstract type for all component models.
Custom models should subtype this and implement:
- `n_parameters(component)`: return number of parameters
- `predict(component, params, t, inputs)`: return predicted values
- `initialize_parameters(component)`: return initial parameter values
"""
abstract type Component end

function n_parameters end
function predict end
function initialize_parameters end
function fit! end

@required Component begin
    n_parameters(::Component)
    initialize_parameters(::Component)
    predict(::Component, ::AbstractVector, ::AbstractVector, ::Any)
    fit!(::AbstractVector, ::Component, ::AbstractVector,::AbstractArray, ::Any)
    fit!(::AbstractVector, ::Component, ::AbstractVector, ::AbstractArray, ::AbstractVector, ::Any)
end