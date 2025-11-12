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

function basis(t::AbstractVector{T}, degree::Int) where T<:Real
    N = length(t)
    X = ones(T, N, degree + 1)
    for d in 1:degree
        X[:, d + 1] = t .^ d
    end
    return X
end

# Necessary methods for Component interface implementation
n_parameters(m::PolynomialRegression) = m.degree + 1

function initialize_parameters(m::PolynomialRegression)
    return randn(n_parameters(m)) .* 0.1
end

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

function fit!(parameters::AbstractVector{T}, model::PolynomialRegression, t::AbstractVector{T}, y::AbstractVecOrMat{T}, ::Any) where T<:Real
    Ξ = basis(t, model.degree)
    parameters .= LS.solve(LS.LinearProblem(Ξ, y[:])).u
end

function fit!(parameters::AbstractVector{T}, model::PolynomialRegression, t::AbstractVector{T}, y::AbstractVecOrMat{T}, w::AbstractVector{T}, ::Any) where T<:Real
    # No need to build W as a matrix
    Wv = view(w, :)
    Ξ = basis(t, model.degree)
    XTW = transpose(Ξ) .* Wv'
    parameters .= LS.solve(LS.LinearProblem(XTW * Ξ, XTW * y[:])).u
end