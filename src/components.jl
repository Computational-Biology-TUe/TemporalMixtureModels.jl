"""
Abstract type for all component models.
Custom models should subtype this and implement:
- `n_parameters(component)`: return number of parameters
- `predict(component, params, t, inputs)`: return predicted values
- `initialize_parameters(component)`: return initial parameter values
"""
abstract type Component end


"""
    PolynomialRegression(degree::Int)

Polynomial regression component model.

# Arguments
- `degree`: Polynomial degree (e.g., 2 for quadratic)
"""
struct PolynomialRegression <: Component
    degree::Int
    
    function PolynomialRegression(degree::Int)
        degree >= 0 || error("Degree must be non-negative")
        new(degree)
    end
end

n_parameters(m::PolynomialRegression) = m.degree + 1

function initialize_parameters(m::PolynomialRegression)
    return randn(n_parameters(m)) .* 0.1
end

"""
Predict using polynomial regression (optimized version)
"""
function predict(m::PolynomialRegression, params::AbstractVector, 
                t::AbstractVector, inputs=nothing)
    n = length(t)
    y_pred = zeros(n)
    
    # Horner's method for polynomial evaluation
    @inbounds for i in 1:n
        ti = t[i]
        result = params[end]
        for j in (m.degree):-1:1
            result = result * ti + params[j]
        end
        y_pred[i] = result
    end
    
    return y_pred
end

function basis(t::AbstractVector{T}, degree::Int) where T<:Real
    N = length(t)
    X = ones(T, N, degree + 1)
    for d in 1:degree
        X[:, d + 1] = t .^ d
    end
    return X
end

function fit!(parameters, model::PolynomialRegression, t::AbstractVector, y::AbstractArray, inputs=nothing)
    Ξ = basis(t, model.degree)
    parameters .= LS.solve(LS.LinearProblem(Ξ, y[:])).u
end

function fit!(parameters, model::PolynomialRegression, t::AbstractVector, y::AbstractArray, w::AbstractVector, inputs=nothing)
    # No need to build W as a matrix
    Wv = view(w, :)
    Ξ = basis(t, model.degree)
    XTW = transpose(Ξ) .* Wv'
    parameters .= LS.solve(LS.LinearProblem(XTW * Ξ, XTW * y[:])).u
end

"""
Build design matrix for polynomial regression (pre-allocated version)
"""
function design_matrix!(X::Matrix{Float64}, m::PolynomialRegression, t::AbstractVector)
    n = length(t)
    @inbounds for i in 1:n
        ti = t[i]
        X[i, 1] = 1.0
        for j in 2:(m.degree + 1)
            X[i, j] = X[i, j-1] * ti
        end
    end
    return X
end

function design_matrix(m::PolynomialRegression, t::AbstractVector)
    X = zeros(length(t), m.degree + 1)
    design_matrix!(X, m, t)
    return X
end