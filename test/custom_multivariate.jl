# https://github.com/Computational-Biology-TUe/TemporalMixtureModels.jl/issues/12#issue-5118887252
# https://github.com/Computational-Biology-TUe/TemporalMixtureModels.jl/issues/13#issue-5119807403

Random.seed!(1234)
import TemporalMixtureModels: Component, initialize_parameters, n_parameters, predict, fit!
using LinearAlgebra
# Generate time points
function generate_3_measurement_input_data(individuals_per_group, t_values, n_groups, group_coefficients_y1, group_coefficients_y2, group_coefficients_y3)
    ids = Int[]
    timepoints = Float64[]
    measurements_y1 = Float64[]
    measurements_y2 = Float64[]
    measurements_y3 = Float64[]
    for group in 1:n_groups
        id_start = sum(individuals_per_group[1:group-1]) + 1
        id_end = sum(individuals_per_group[1:group])
        for individual in id_start:id_end
            for t in t_values
                y1 = group_coefficients_y1[group][1] + group_coefficients_y1[group][2]*t + group_coefficients_y1[group][3]*t^2 + randn()*0.25
                y2 = group_coefficients_y2[group][1] + group_coefficients_y2[group][2]*t + group_coefficients_y2[group][3]*t^2 + randn()*0.25
                y3 = group_coefficients_y3[group][1] + group_coefficients_y3[group][2]*t + group_coefficients_y3[group][3]*t^2 + randn()*0.25
                push!(ids, individual)
                push!(timepoints, t)
                push!(measurements_y1, y1)
                push!(measurements_y2, y2)
                push!(measurements_y3, y3)
            end
        end
    end

    return ids, timepoints, [measurements_y1 measurements_y2 measurements_y3]
end

function generate_4_measurement_input_data(individuals_per_group, t_values, n_groups, group_coefficients_y1, group_coefficients_y2, group_coefficients_y3, group_coefficients_y4)
    ids = Int[]
    timepoints = Float64[]
    measurements_y1 = Float64[]
    measurements_y2 = Float64[]
    measurements_y3 = Float64[]
    measurements_y4 = Float64[]
    for group in 1:n_groups
        id_start = sum(individuals_per_group[1:group-1]) + 1
        id_end = sum(individuals_per_group[1:group])
        for individual in id_start:id_end
            for t in t_values
                y1 = group_coefficients_y1[group][1] + group_coefficients_y1[group][2]*t + group_coefficients_y1[group][3]*t^2 + randn()*0.25
                y2 = group_coefficients_y2[group][1] + group_coefficients_y2[group][2]*t + group_coefficients_y2[group][3]*t^2 + randn()*0.25
                y3 = group_coefficients_y3[group][1] + group_coefficients_y3[group][2]*t + group_coefficients_y3[group][3]*t^2 + randn()*0.25
                y4 = group_coefficients_y4[group][1] + group_coefficients_y4[group][2]*t + group_coefficients_y4[group][3]*t^2 + randn()*0.25
                push!(ids, individual)
                push!(timepoints, t)
                push!(measurements_y1, y1)
                push!(measurements_y2, y2)
                push!(measurements_y3, y3)
                push!(measurements_y4, y4)
            end
        end
    end

    return ids, timepoints, [measurements_y1 measurements_y2 measurements_y3 measurements_y4]
end

# define a custom multivariate component (just triple polynomials)
struct TriplePolynomial <: TemporalMixtureModels.Component
  order::Int
end

function TemporalMixtureModels.n_parameters(m::TriplePolynomial)
    return 3*(m.order+1)
end

function TemporalMixtureModels.initialize_parameters(m::TriplePolynomial)
    return randn(TemporalMixtureModels.n_parameters(m)) .* 0.1  # Small random initialization
end

function TemporalMixtureModels.predict(m::TriplePolynomial, params::AbstractVector, t::AbstractVector, inputs=nothing)
    n = length(t)
    y_pred = zeros(n, 3)  # 3 measurements
    for i in 1:n
        for j in 0:m.order
            y_pred[i, 1] += params[j+1] * t[i]^j
            y_pred[i, 2] += params[m.order+1 + j+1] * t[i]^j
            y_pred[i, 3] += params[2*(m.order+1) + j+1] * t[i]^j
        end
    end
    return y_pred
end

function TemporalMixtureModels.fit!(parameters::AbstractVector{T}, m::TriplePolynomial, t::AbstractVector{T}, y::AbstractVecOrMat{T}, ::Any) where T<:Real
    # Fit each measurement separately using polynomial regression
    for j in 1:3
        # Extract the relevant parameters for this measurement
        start_idx = (j-1)*(m.order+1) + 1
        end_idx = j*(m.order+1)

        # Fit polynomial regression for this measurement
        X = hcat([t.^i for i in 0:m.order]...)
        coeffs = X \ y[:, j]  # Least squares fit
        parameters[start_idx:end_idx] = coeffs
    end
end

function TemporalMixtureModels.fit!(parameters::AbstractVector{T}, m::TriplePolynomial, t::AbstractVector{T}, y::AbstractVecOrMat{T}, w::AbstractVector{T}, ::Any) where T<:Real
    # Fit each measurement separately using weighted polynomial regression
    for j in 1:3
        # Extract the relevant parameters for this measurement
        start_idx = (j-1)*(m.order+1) + 1
        end_idx = j*(m.order+1)

        # Fit polynomial regression for this measurement with weights
        X = hcat([t.^i for i in 0:m.order]...)
        W = Diagonal(w)
        coeffs = (X' * W * X) \ (X' * W * y[:, j])  # Weighted least squares fit
        parameters[start_idx:end_idx] = coeffs
    end
end

@testset "custom multivariate mixture model" begin

    individuals_per_group = [20, 30]
    t_values = 0:0.1:10
    n_groups = 2

    group_coefficients_y1 = [ [2.0, -0.5, 0.05],  # Group 1: y1 = 2 - 0.5*t + 0.05*t^2
                        [1.0, 0.3, -0.02] ] # Group 2: y1 = 1 + 0.3*t - 0.02*t^2
    group_coefficients_y2 = [ [0.5, 0.4, -0.03],  # Group 1: y2 = 0.5 + 0.4*t - 0.03*t^2
                        [3.0, -0.2, 0.04] ] # Group 2: y2 = 3 - 0.2*t + 0.04*t^2
    group_coefficients_y3 = [ [1.0, 0.1, -0.01],  # Group 1: y3 = 1 + 0.1*t - 0.01*t^2
                        [2.0, -0.3, 0.03] ] # Group 2: y3 = 2 - 0.3*t + 0.03*t^2

    ids, timepoints, y = generate_3_measurement_input_data(individuals_per_group, t_values, n_groups, group_coefficients_y1, group_coefficients_y2, group_coefficients_y3)

    model = TriplePolynomial(2)
    result = fit_mixture(model, 2, timepoints, y, ids)

    @test sum(result.cluster_probs) ≈ 1.0 atol=1e-8
    @test result.converged == true

    bic_value = bic(result, timepoints, y, ids)
    @test isa(bic_value, Float64)

    aic_value = aic(result, timepoints, y, ids)
    @test isa(aic_value, Float64)

    ll_value = loglikelihood(result, timepoints, y, ids)
    @test isa(ll_value, Float64)
    @test ll_value ≈ result.loglikelihood atol=1e-8

end

@testset "custom composition with multivariate mixture model" begin

    individuals_per_group = [20, 30]
    t_values = 0:0.1:10
    n_groups = 2

    group_coefficients_y1 = [ [2.0, -0.5, 0.05],  # Group 1: y1 = 2 - 0.5*t + 0.05*t^2
                        [1.0, 0.3, -0.02] ] # Group 2: y1 = 1 + 0.3*t - 0.02*t^2
    group_coefficients_y2 = [ [0.5, 0.4, -0.03],  # Group 1: y2 = 0.5 + 0.4*t - 0.03*t^2
                        [3.0, -0.2, 0.04] ] # Group 2: y2 = 3 - 0.2*t + 0.04*t^2
    group_coefficients_y3 = [ [1.0, 0.1, -0.01],  # Group 1: y3 = 1 + 0.1*t - 0.01*t^2
                        [2.0, -0.3, 0.03] ] # Group 2: y3 = 2 - 0.3*t + 0.03*t^2
    group_coefficients_y4 = [ [0.8, 0.2, -0.02],  # Group 1: y4 = 0.8 + 0.2*t - 0.02*t^2
                        [1.5, -0.1, 0.03] ] # Group 2: y4 = 1.5 - 0.1*t + 0.03*t^2

    ids, timepoints, y = generate_4_measurement_input_data(individuals_per_group, t_values, n_groups, group_coefficients_y1,
    group_coefficients_y2, group_coefficients_y3, group_coefficients_y4)

    model =     model = @component begin
        y[1:3] ~ TriplePolynomial(2)  # Quadratic for measurement 1:3
        y[4] ~ PolynomialRegression(2)  # Quadratic for measurement 4
    end
    result = fit_mixture(model, 2, timepoints, y, ids)

    @test sum(result.cluster_probs) ≈ 1.0 atol=1e-8
    @test result.converged == true

    bic_value = bic(result, timepoints, y, ids)
    @test isa(bic_value, Float64)

    aic_value = aic(result, timepoints, y, ids)
    @test isa(aic_value, Float64)

    ll_value = loglikelihood(result, timepoints, y, ids)
    @test isa(ll_value, Float64)
    @test ll_value ≈ result.loglikelihood atol=1e-8

end



