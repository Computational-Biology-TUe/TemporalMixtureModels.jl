# polynomials
abstract type AbstractMixtureModelComponent{T<:Real} end
abstract type AbstractRegressionModel{T<:Real} <: AbstractMixtureModelComponent{T} end

function log_likelihood(model::AbstractMixtureModelComponent{T}, t::AbstractVector{T}, y::AbstractVector{T}, variance::T) where T<:Real
    N = length(y)
    residuals = y .- predict(model, t)
    return - (N/2) * log(2*pi) - (N/2) * log(variance) - sum(abs2, residuals) / (2 * variance)
end

function variance(model::AbstractMixtureModelComponent{T}, t::AbstractVector{T}, y::AbstractVector{T}) where T<:Real
    N = length(y)
    residuals = y .- predict(model, t)
    return sum(abs2, residuals) ./ (N - n_parameters(model))
end

function variance(model::AbstractMixtureModelComponent{T}, t::AbstractVector{T}, y::AbstractVector{T}, w::AbstractVector{T}) where T<:Real
    residuals = y .- predict(model, t)
    return sum(w .* abs2.(residuals)) ./ (sum(w) - n_parameters(model))
end

function predict(model::AbstractRegressionModel, t::AbstractVector)
    X = basis(t, model.degree)
    return X * model.coefficients
end

function randinit!(model::AbstractRegressionModel, rng=Random.default_rng())
    randn!(rng, model.coefficients)
end

function n_parameters(model::AbstractRegressionModel)
    return length(model.coefficients)
end

# ============================================================
# Polynomial Regression Model
# ============================================================

mutable struct PolynomialRegression{T<:Real} <: AbstractRegressionModel{T}
    degree::Int
    coefficients::Vector{T}
end

"""
Create a polynomial regression mixture model component of specified degree.

## Arguments
- `degree::Int`: Degree of the polynomial.

## Example
Creating a polynomial regression model of degree 2:
```julia
model = PolynomialRegression(2)
```
"""
function PolynomialRegression(degree::Int)
    return PolynomialRegression{Float64}(degree, zeros(Float64, degree + 1))
end

function basis(t::AbstractVector{T}, degree::Int) where T<:Real
    N = length(t)
    X = ones(T, N, degree + 1)
    for d in 1:degree
        X[:, d + 1] = t .^ d
    end
    return X
end

function fit!(model::PolynomialRegression, t::AbstractVector, y::AbstractVector)
    Ξ = basis(t, model.degree)
    model.coefficients = LS.solve(LS.LinearProblem(Ξ, y)).u
end

function fit!(model::PolynomialRegression, t::AbstractVector, y::AbstractVector, w::AbstractVector)
    # No need to build W as a matrix
    Wv = view(w, :)
    Ξ = basis(t, model.degree)
    XTW = transpose(Ξ) .* Wv'
    model.coefficients = LS.solve(LS.LinearProblem(XTW * Ξ, XTW * y)).u
end

# ============================================================
# Ridge (L2) Polynomial Regression Model
# ============================================================
mutable struct RidgePolynomialRegression{T<:Real} <: AbstractRegressionModel{T}
    degree::Int
    coefficients::Vector{T}
    lambda::T

end


"""
Create a ridge (L2) polynomial regression mixture model component of specified degree and regularization parameter.

## Arguments
- `degree::Int`: Degree of the polynomial.
- `lambda::T`: Regularization parameter (L2 penalty).

## Example
Creating a ridge polynomial regression model of degree 2 with lambda = 0.1:
```julia
model = RidgePolynomialRegression(2, 0.1)
```
"""
function RidgePolynomialRegression(degree::Int, lambda)
    lambda = convert(Float64, lambda)
    return RidgePolynomialRegression{Float64}(degree, zeros(Float64, degree + 1), lambda)
end

function fit!(model::RidgePolynomialRegression, t::AbstractVector, y::AbstractVector)
    Ξ = basis(t, model.degree)
    model.coefficients = LS.solve(LS.LinearProblem(Ξ' * Ξ + model.lambda * I, Ξ' * y)).u
end

function fit!(model::RidgePolynomialRegression, t::AbstractVector, y::AbstractVector, w::AbstractVector)
    # No need to build W as a matrix
    Wv = view(w, :)
    Ξ = basis(t, model.degree)
    XTW = transpose(Ξ) .* Wv'
    model.coefficients = LS.solve(LS.LinearProblem(XTW * Ξ + model.lambda * I, XTW * y)).u
end

# ============================================================
# Lasso (L1) Polynomial Regression Model
# ============================================================
mutable struct LassoPolynomialRegression{T<:Real} <: AbstractRegressionModel{T}
    degree::Int
    coefficients::Vector{T}
    lambda::T
end


"""
Create a lasso (L1) polynomial regression mixture model component of specified degree and regularization parameter.

## Arguments
- `degree::Int`: Degree of the polynomial.
- `lambda::T`: Regularization parameter (L1 penalty).

## Example
Creating a lasso polynomial regression model of degree 2 with lambda = 0.1:
```julia
model = LassoPolynomialRegression(2, 0.1)
```
"""
function LassoPolynomialRegression(degree::Int, lambda)
    lambda = convert(Float64, lambda)
    return LassoPolynomialRegression{Float64}(degree, zeros(Float64, degree + 1), lambda)
end

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

function fit!(model::LassoPolynomialRegression, t::AbstractVector, y::AbstractVector)
    Ξ = basis(t, model.degree)
    model.coefficients = _fit_lasso_model(Ξ, y, model.lambda)
end

function fit!(model::LassoPolynomialRegression, t::AbstractVector, y::AbstractVector, w::AbstractVector)
    # No need to build W as a matrix
    Ξ = basis(t, model.degree)
    model.coefficients = _fit_lasso_model(Ξ, y, model.lambda, w)
end