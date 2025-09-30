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
    input_data = DataFrame(id = [ids; ids], time = [timepoints; timepoints], value = [measurements_y; measurements_z], var_name = repeat(["y", "z"], inner=length(ids)))

    return input_data
end

@testset "univariate mixture model" begin

    individuals_per_group = [20, 30]
    t_values = 0:0.1:10
    n_groups = 2

    group_coefficients_y = [ [2.0, -0.5, 0.05],  # Group 1: y = 2 - 0.5*t + 0.05*t^2
                        [1.0, 0.3, -0.02] ] # Group 2: y = 1 + 0.3*t - 0.02*t^2
    group_coefficients_z = [ [0.5, 0.4, -0.03],  # Group 1: z = 0.5 + 0.4*t - 0.03*t^2
                        [3.0, -0.2, 0.04] ] # Group 2: z = 3 - 0.2*t + 0.04*t^2


    input_data = generate_multivariate_input_data(individuals_per_group, t_values, n_groups, group_coefficients_y, group_coefficients_z)

    model = MultivariateMixtureModel(2, Dict(:y => PolynomialRegression(2), :z => PolynomialRegression(2)))
    fit!(model, input_data)

    @test sum(model.weights) ≈ 1.0 atol=1e-8
    @test model.converged == true

    fit!(model, input_data, hard_assignment=true)
    @test sum(model.weights) ≈ 1.0 atol=1e-8
    @test model.converged == true

    fit!(model, input_data, max_iter=1)
    @test model.converged == false
    @test model.iterations == 1

    @test length(model.components) == 2

    # run bootstrap confidence intervals
    ci_results, component_samples = bootstrap_ci(model, input_data, n_bootstrap=10)
    @test length(ci_results) == 2

end




