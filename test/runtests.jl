using TemporalMixtureModels, Random, Test

@testset "Univariate Mixture Models" begin
    include("univariate.jl")
end

@testset "Multivariate Mixture Models" begin
    include("multivariate.jl")
end