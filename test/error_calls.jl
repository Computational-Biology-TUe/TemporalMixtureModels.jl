@testset "Error calls" begin


    # generate data where ids are not compatible with y
    t = collect(0.0:0.1:10.0)
    y = randn(length(t), 2)  # 2D measurements
    ids = vcat(fill(1, length(t)÷2), fill(2, length(t)÷2 - 1))  # incompatible length

    # try to fit mixture model and expect an error
    model = PolynomialRegression(2)
    @test_throws AssertionError fit_mixture(model, 2, t, y, ids)

    # generate data where one subject has all missing entries in one variable
    t = repeat(collect(0.0:0.1:5.0), inner=2)
    y = Matrix{Union{Missing, Float64}}(randn(length(t), 2))  # 2D measurements
    ids = repeat([1, 2], inner=length(t)÷2)  # compatible length
    # Introduce missing data for subject 2 in variable 1
    for i in eachindex(ids)
        if ids[i] == 2
            y[i, 1] = missing
        end
    end
    
    # try to fit mixture model and expect an ArgumentError
    @test_throws ArgumentError fit_mixture(model, 2, t, y, ids)

end