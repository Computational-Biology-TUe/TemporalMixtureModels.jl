Random.seed!(1234)

# Generate time points
function generate_multivariate_input_data(individuals_per_group, t_values, n_groups, group_coefficients_y, group_coefficients_z)
    ids = Int[]
    timepoints = Float64[]
    measurements_y = Float64[]
    measurements_z = Float64[]
    for group in 1:n_groups
        id_start = sum(individuals_per_group[1:group-1]) + 1
        id_end = sum(individuals_per_group[1:group])
        for individual in id_start:id_end
            for t in t_values
                y = group_coefficients_y[group][1] + group_coefficients_y[group][2]*t + group_coefficients_y[group][3]*t^2 + randn()*0.25
                z = group_coefficients_z[group][1] + group_coefficients_z[group][2]*t + group_coefficients_z[group][3]*t^2 + randn()*0.25
                push!(ids, individual)
                push!(timepoints, t)
                push!(measurements_y, y)
                push!(measurements_z, z)
            end
        end
    end

    return ids, timepoints, [measurements_y measurements_z]
end

@testset "univariate mixture model" begin

    individuals_per_group = [20, 30]
    t_values = 0:0.1:10
    n_groups = 2

    group_coefficients_y = [ [2.0, -0.5, 0.05],  # Group 1: y = 2 - 0.5*t + 0.05*t^2
                        [1.0, 0.3, -0.02] ] # Group 2: y = 1 + 0.3*t - 0.02*t^2
    group_coefficients_z = [ [0.5, 0.4, -0.03],  # Group 1: z = 0.5 + 0.4*t - 0.03*t^2
                        [3.0, -0.2, 0.04] ] # Group 2: z = 3 - 0.2*t + 0.04*t^2


    ids, timepoints, y = generate_multivariate_input_data(individuals_per_group, t_values, n_groups, group_coefficients_y, group_coefficients_z)

    model = @component begin
        y[1] ~ PolynomialRegression(2)  # Quadratic for measurement 1
        y[2] ~ PolynomialRegression(2)  # Quadratic for measurement 2
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




