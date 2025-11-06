using DataFrames
using Random
using Distributions
using LinearAlgebra
using Statistics
import LinearSolve as LS
using SpecialFunctions

include("data.jl")
include("components.jl")
include("composition.jl")
include("errormodels.jl")
include("solve.jl")

# test composite component

println("="^70)
println("Example 1: Normal Error Model")
println("="^70)

Random.seed!(123)

function generate_normal_data(n_subjects=50, n_timepoints=20)
    df = DataFrame(
        id = Int[],
        time = Float64[],
        glucose = Union{Float64, Missing}[]
    )
    
    for i in 1:n_subjects
        if i <= n_subjects ÷ 2
            # Cluster 1: Rising glucose
            for t in 0:n_timepoints-1
                glucose = 12 + 5*t + 0.3*t^2 + randn()*5
                if rand() > 0.95
                    glucose = missing
                end
                push!(df, (id=i, time=Float64(t), glucose=glucose))
            end
        else
            # Cluster 2: Stable glucose
            for t in 0:n_timepoints-1
                glucose = 85 + 1*t + randn()*4
                if rand() > 0.95
                    glucose = missing
                end
                push!(df, (id=i, time=Float64(t), glucose=glucose))
            end
        end
    end
    
    return df
end

df_normal = generate_normal_data()

y = df_normal[!, :glucose]
t = df_normal[!, :time]
ids = df_normal[!, :id]

result = fit_mixture(
    PolynomialRegression(2), 2, t, y, ids
)

# ============================================================================
# Example 4: Multiple Measurements with Composite Component
# ============================================================================

println("\n\n" * "="^70)
println("Example 4: Multiple Measurements")
println("="^70)

function generate_multi_measurement_data(n_subjects=50, n_timepoints=20)
    df = DataFrame(
        id = Int[],
        time = Float64[],
        glucose = Union{Float64, Missing}[],
        insulin = Union{Float64, Missing}[]
    )
    
    for i in 1:n_subjects
        if i <= n_subjects ÷ 2
            # Cluster 1
            for t in 0:n_timepoints-1
                glucose = 90 + 5*t + 0.3*t^2 + randn()*4
                insulin = 15 + 2*t + randn()*2
                
                if rand() > 0.95
                    glucose = missing
                end
                if rand() > 0.95
                    insulin = missing
                end
                
                push!(df, (id=i, time=Float64(t), glucose=glucose, insulin=insulin))
            end
        else
            # Cluster 2
            for t in 0:n_timepoints-1
                glucose = 85 + 1*t + randn()*4
                insulin = 20 + 3*t + 0.5*t^2 + randn()*2
                
                if rand() > 0.95
                    glucose = missing
                end
                if rand() > 0.95
                    insulin = missing
                end
                
                push!(df, (id=i, time=Float64(t), glucose=glucose, insulin=insulin))
            end
        end
    end
    
    return df
end

df_multi = generate_multi_measurement_data()

y_multi = hcat(df_multi[!, :glucose], df_multi[!, :insulin])
t_multi = df_multi[!, :time]
ids_multi = df_multi[!, :id]

model = @component begin
    y[1] ~ PolynomialRegression(2)
    y[2] ~ PolynomialRegression(2)
end

println("\nFitting composite model...")
result_multi = fit_mixture(
    model,
    2,
    t_multi,
    y_multi,
    ids_multi;
    error_model=NormalError(),
    verbose=true
)

result_multi.parameters