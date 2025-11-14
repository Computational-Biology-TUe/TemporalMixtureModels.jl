Random.seed!(1234)

# Generate time points
function generate_univariate_input_data(individuals_per_group, t_values, n_groups, group_coefficients)
    ids = Int[]
    timepoints = Float64[]
    measurements = Float64[]
    for group in 1:n_groups
        id_start = sum(individuals_per_group[1:group-1]) + 1
        id_end = sum(individuals_per_group[1:group])
        for individual in id_start:id_end
            for t in t_values
                y = group_coefficients[group][1] + group_coefficients[group][2]*t + group_coefficients[group][3]*t^2 + randn()*0.25
                push!(ids, individual)
                push!(timepoints, t)
                push!(measurements, y)
            end
        end
    end

    ids, timepoints, measurements
end

@testset "univariate mixture model" begin

    individuals_per_group = [20, 30]
    t_values = 0:0.1:10
    n_groups = 2

    group_coefficients = [ [2.0, -0.5, 0.05],  # Group 1: y = 2 - 0.5*t + 0.05*t^2
                        [1.0, 0.3, -0.02] ] # Group 2: y = 1 + 0.3*t - 0.02*t^2


    ids, timepoints, measurements = generate_univariate_input_data(individuals_per_group, t_values, n_groups, group_coefficients)

    model = PolynomialRegression(2)
    result = fit_mixture(model, 2, timepoints, measurements, ids)

    @test sum(result.cluster_probs) ≈ 1.0 atol=1e-8
    @test result.converged == true

    result = fit_mixture(model, 2, timepoints, measurements, ids; max_iter=1)
    @test result.converged == false
    @test result.n_iterations == 1

    bic_value = bic(result, timepoints, measurements, ids)
    @test isa(bic_value, Float64)

    aic_value = aic(result, timepoints, measurements, ids)
    @test isa(aic_value, Float64)

    ll_value = loglikelihood(result, timepoints, measurements, ids)
    @test isa(ll_value, Float64)
    @test ll_value ≈ result.loglikelihood atol=1e-8

end




