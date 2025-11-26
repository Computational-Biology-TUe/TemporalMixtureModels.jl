using TemporalMixtureModels, Random, Test

@testset "Error calls" begin
    include("error_calls.jl")
end

@testset "Univariate Mixture Models" begin
    include("univariate.jl")
end

@testset "Multivariate Mixture Models" begin
    include("multivariate.jl")
end