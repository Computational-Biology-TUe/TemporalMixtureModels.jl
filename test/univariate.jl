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

@testset "initial cluster assignments" begin
    individuals_per_group = [10, 15]
    t_values = 0:0.2:5
    n_groups = 2

    group_coefficients = [ [3.0, -0.3, 0.02],  # Group 1
                          [1.0, 0.5, -0.01] ]  # Group 2

    ids, timepoints, measurements = generate_univariate_input_data(individuals_per_group, t_values, n_groups, group_coefficients)

    # Create initial assignments based on true groups
    n_subjects = individuals_per_group[1] + individuals_per_group[2]
    initial_assignments = vcat(fill(1, individuals_per_group[1]), fill(2, individuals_per_group[2]))

    model = PolynomialRegression(2)
    
    # Test with initial assignments
    result_with_init = fit_mixture(model, 2, timepoints, measurements, ids; 
                                    initial_assignments=initial_assignments, 
                                    n_repeats=1, verbose=false)
    
    @test result_with_init.converged == true
    @test sum(result_with_init.cluster_probs) ≈ 1.0 atol=1e-8
    @test size(result_with_init.responsibilities) == (n_subjects, 2)
    
    # Test without initial assignments for comparison
    result_random = fit_mixture(model, 2, timepoints, measurements, ids; 
                                n_repeats=1, verbose=false)
    
    @test result_random.converged == true
    
    # Both should produce valid results (though potentially different)
    @test isa(result_with_init.loglikelihood, Float64)
    @test isa(result_random.loglikelihood, Float64)
    
    # Test error handling: wrong number of assignments
    @test_throws AssertionError fit_mixture(model, 2, timepoints, measurements, ids; 
                                             initial_assignments=[1, 2, 1], 
                                             n_repeats=1, verbose=false)
    
    # Test error handling: invalid cluster number
    bad_assignments = vcat(fill(1, individuals_per_group[1]), fill(3, individuals_per_group[2]))
    @test_throws AssertionError fit_mixture(model, 2, timepoints, measurements, ids; 
                                             initial_assignments=bad_assignments, 
                                             n_repeats=1, verbose=false)

end




