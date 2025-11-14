abstract type RegressionComponent <: Component end


"""
    PolynomialRegression(degree::Int)

Polynomial regression component model.

# Arguments
- `degree`: Polynomial degree (e.g., 2 for quadratic)
"""
struct PolynomialRegression <: RegressionComponent
    degree::Int
    
    function PolynomialRegression(degree::Int)
        degree >= 0 || error("Degree must be non-negative")
        new(degree)
    end
end

"""
    RidgeRegression(degree::Int, λ::Real)

Polynomial regression component model with L2 regularization penalization (Ridge regression).

# Arguments
- `degree`: Polynomial degree (e.g., 2 for quadratic)
- `λ`: Regularization strength (non-negative)
"""
struct RidgeRegression <: RegressionComponent
    degree::Int
    λ::Real
    
    function RidgeRegression(degree::Int, λ::Real)
        degree >= 0 || error("Degree must be non-negative")
        λ >= 0 || error("Regularization strength λ must be non-negative")
        new(degree, λ)
    end
end

"""
    LassoRegression(degree::Int, λ::Real)

Polynomial regression component model with L1 regularization penalization (Lasso regression).

# Arguments
- `degree`: Polynomial degree (e.g., 2 for quadratic)
- `λ`: Regularization strength (non-negative)
"""
struct LassoRegression <: RegressionComponent
    degree::Int
    λ::Real
    
    function LassoRegression(degree::Int, λ::Real)
        degree >= 0 || error("Degree must be non-negative")
        λ >= 0 || error("Regularization strength λ must be non-negative")
        new(degree, λ)
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
n_parameters(m::RegressionComponent) = m.degree + 1

function initialize_parameters(m::RegressionComponent)
    return randn(n_parameters(m)) .* 0.1
end

function predict(m::RegressionComponent, params::AbstractVector, 
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

# Polynomial Regression fitting
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

# Ridge Regression fitting
function fit!(parameters::AbstractVector{T}, model::RidgeRegression, t::AbstractVector{T}, y::AbstractVecOrMat{T}, ::Any) where T<:Real
    Ξ = basis(t, model.degree)
    n_params = n_parameters(model)
    A = transpose(Ξ) * Ξ + model.λ * I(n_params)
    b = transpose(Ξ) * y[:]
    parameters .= LS.solve(LS.LinearProblem(A, b)).u
end

function fit!(parameters::AbstractVector{T}, model::RidgeRegression, t::AbstractVector{T}, y::AbstractVecOrMat{T}, w::AbstractVector{T}, ::Any) where T<:Real
    # No need to build W as a matrix
    Wv = view(w, :)
    Ξ = basis(t, model.degree)
    XTW = transpose(Ξ) .* Wv'
    n_params = n_parameters(model)
    A = XTW * Ξ + model.λ * I(n_params)
    b = XTW * y[:]
    parameters .= LS.solve(LS.LinearProblem(A, b)).u
end

# Lasso Regression fitting
function _fit_lasso_model(Ξ, y, lambda)
    # using Convex.jl and SCS solver
    T, K = size(Ξ)
    Q = Ξ'Ξ / T
    c = Ξ'y / T

    b = Convex.Variable(K)
    obj = Convex.quadform(b, Q) - 2 * Convex.dot(c, b) + lambda * Convex.norm(b, 1)
    problem = Convex.minimize(obj)
    Convex.solve!(problem, SCS.Optimizer; silent=true)
    return vec(Convex.evaluate(b))
end

function _fit_lasso_model(Ξ, y, lambda, w)
    # using Convex.jl and SCS solver
    T, K = size(Ξ)
    Xw = Ξ .* w                   
    Q = (Ξ' * Xw) / T
    c = (Ξ' * (w .* y)) / T

    # preserve symmetry of Q
    Q = (Q + Q') / 2

    b = Convex.Variable(K)
    obj = Convex.quadform(b, Q) - 2 * Convex.dot(c, b) + lambda * Convex.norm(b, 1)
    problem = Convex.minimize(obj)
    Convex.solve!(problem, SCS.Optimizer; silent=true)
    return vec(Convex.evaluate(b))
end

function fit!(parameters::AbstractVector{T}, model::LassoRegression, t::AbstractVector{T}, y::AbstractVecOrMat{T}, ::Any) where T<:Real
    Ξ = basis(t, model.degree)
    parameters .= _fit_lasso_model(Ξ, y[:], model.lambda)
end

function fit!(parameters::AbstractVector{T}, model::LassoRegression, t::AbstractVector{T}, y::AbstractVecOrMat{T}, w::AbstractVector{T}, ::Any) where T<:Real
    # No need to build W as a matrix
    Ξ = basis(t, model.degree)
    parameters .= _fit_lasso_model(Ξ, y[:], model.lambda, w)
end