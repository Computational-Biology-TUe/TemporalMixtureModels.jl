# Implementing Custom Components
This tutorial demonstrates how to create custom model components by subtyping the `AbstractMixtureModelComponent` and implementing the required methods. This allows for flexibility in defining new model types and behaviors tailored to specific use cases. In this example, we will create an ODE model component using the `OrdinaryDiffEq.jl` package and optimize its parameters using `Optim.jl`.

Load the necessary packages:
```@example custom
using OrdinaryDiffEq, CairoMakie, TemporalMixtureModels, DataFrames, Optim, Random
```

## Generating some synthetic data
In this tutorial, the ODE model that we will use is based on a simple decay process defined by the differential equation:

```math
\frac{\mathrm{d}u}{\mathrm{d}t} = -k u
```

where ``k`` is the decay constant, and we have an initial condition ``u(0) = u_0``. While we can analytically solve this ODE, we will use a numerical solver to demonstrate how to integrate ODE solving into a custom model component.

We will generate synthetic data by simulating this decay process with added Gaussian noise. The input DataFrame should be a so-called "long" format, with columns for the individual ID, time points, and observed values. First, we define functions to simulate the decay process and generate synthetic data for multiple individuals.
```@example custom
# Set random seed for reproducibility
Random.seed!(1234)

function decay_ode!(du, u, p, t)
    k = p[1]
    du[1] = -k * u[1]
end

function simulate_decay(p, u0, t; noise_std=0.1)
    prob = ODEProblem(decay_ode!, u0, (t[1], t[end]), p)
    sol = solve(prob, Tsit5(), saveat=t)
    noisy_data = sol .+ noise_std * randn(length(t))
    return noisy_data
end

function simulate_group(p_mean, p_std, u0_mean, u0_std, t, n_individuals, id_start=1; noise_std=0.1)
    p = p_mean .+ randn(n_individuals) .* p_std
    u0 = u0_mean .+ randn(n_individuals) .* u0_std

    values = Float64[]
    time = Float64[]
    ids = Int[]
    for (i, id) in enumerate(id_start:(id_start + n_individuals - 1))
        data = Array(simulate_decay([p[i]], [u0[i]], t; noise_std=noise_std))[1,:]
        append!(values, data)
        append!(time, t)
        append!(ids, fill(id, length(t)))
    end
    return DataFrame(id=ids, time=time, value=values)
end
```

We can now generate synthetic data for three groups of individuals, each with different decay constants and initial conditions.
```@example custom
# simulate three groups
time_points = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
group1 = simulate_group(0.5, 0.05, 5.0, 0.5, time_points, 30)
group2 = simulate_group(1.0, 0.1, 3.0, 0.3, time_points, 15, 31)
group3 = simulate_group(1.5, 0.15, 1.0, 0.2, time_points, 45, 46)
data = vcat(group1, group2, group3)
first(data, 5)  # Display the first 5 rows of the DataFrame
```

Here you can see the first few rows of the generated DataFrame. Each row corresponds to a measurement for a specific individual at a specific time point.

## Defining the Custom ODE Model Component
We will now define a custom model component by subtyping the `AbstractMixtureModelComponent`.

```@example custom
mutable struct DecayODE{T<:Real} <: TemporalMixtureModels.AbstractMixtureModelComponent{T}
    coefficients::Vector{T}  # [k, u0]
    function DecayODE(p::T, u0::T) where T<:Real
        new{T}([p, u0])
    end
end

function DecayODE()
    DecayODE(1.0, 1.0)
end
```

The `DecayODE` struct holds the parameters of the ODE model, specifically the decay constant ``k`` and the initial condition ``u_0``. The coefficients field is _mandatory_ for all model components, as it is used in the fitting process.

Next, we need to implement four essential methods for our custom model component: `predict(::DecayODE, t)`, `randinit!(::DecayODE)`, `fit!(::DecayODE, t, y)`, and `fit!(::DecayODE, t, y, w)`.
```@example custom
function TemporalMixtureModels.predict(m::DecayODE, t)
    p, u0 = m.coefficients
    prob = ODEProblem(decay_ode!, [u0], (minimum(t), maximum(t)), [p])

    # argsort t
    sorted_indices = sortperm(t)

    sol = Array(solve(prob, Tsit5(), saveat=t))[:]

    # count the number of time points where t == minimum(t), and add those to the start of sol (n - 1) times
    n = count(x -> x == minimum(t), t)
    if n > 1
        sol = vcat(fill(sol[1], n - 1), sol)
    end

    try
        return sol[invperm(sorted_indices)]
    catch
        return sol
    end
end

function TemporalMixtureModels.randinit!(m::DecayODE)
    m.coefficients = [rand(0.1:0.1:2.0), rand(0.5:0.5:6.0)]
end

function TemporalMixtureModels.fit!(m::DecayODE, t, y)
    function loss(p)
        y_hat = TemporalMixtureModels.predict(DecayODE(p[1], p[2]), t)
        if length(y_hat) != length(y)
            return Inf
        end
        return sum(abs2, y_hat .- y)
    end
    result = Optim.minimizer(optimize(loss, [0.0, 0.0], [10.0,10.0], m.coefficients, Fminbox(BFGS()), autodiff=:forward))
    m.coefficients = result
end

function TemporalMixtureModels.fit!(m::DecayODE, t, y, weights)
    function loss(p)
        y_hat = TemporalMixtureModels.predict(DecayODE(p[1], p[2]), t)
        if length(y_hat) != length(y)
            return Inf
        end
        return sum(weights .* (y_hat .- y).^2)
    end
    result = Optim.minimizer(optimize(loss, [0.0, 0.0], [10.0,10.0], m.coefficients, Fminbox(BFGS()), autodiff=:forward))
    m.coefficients = result
end
```

The `predict` function uses the `OrdinaryDiffEq.jl` package to solve the ODE numerically for given time points. The `randinit!` function initializes the model parameters randomly within specified ranges. The `fit!` functions optimize the model parameters to minimize the squared error between the predicted and observed values, with and without weights.

## Fitting the Custom Model Component
After defining the custom model component, we can now fit a temporal mixture model using our `DecayODE` component.
```@example custom
mm = UnivariateMixtureModel(3, DecayODE())
TemporalMixtureModels.fit!(mm, data)
```

Similarly, we can also use the bootstrap method to estimate confidence intervals for the model parameters.
```@example custom
n_bootstrap = 50  # Number of bootstrap samples
confidence_intervals, _, _ = bootstrap_ci(mm, data; n_bootstrap=n_bootstrap)
confidence_intervals
```

The output shows the confidence intervals for the parameters of each component model in the mixture. We can use these intervals and the fitted model to visualize the results.

```@example custom
# Create a range of time points for prediction
t_pred = 0:0.1:3.0

# Prepare a figure
figure_mixture = let f = Figure()
    ax = Axis(f[1, 1], xlabel="Time", ylabel="Value", title="Temporal Mixture Model Fit")

    # Plot the original data points
    scatter!(ax, data.time, data.value; color=:gray, markersize=4, label="Data")

    # Plot the fitted component models
    predictions = predict(mm, t_pred)

    COMPONENT_COLORS = [colorant"#c74300", colorant"#008aa1", colorant"#ffc300"]

    for k in axes(predictions, 2)
        lines!(ax, t_pred, predictions[:, k]; 
        label="Component $k", color=COMPONENT_COLORS[k], linewidth=2)

        # get the confidence bounds. This is still a bit clunky because there is no API for this yet
        lower_bound_parameters = confidence_intervals[k][:lower]
        upper_bound_parameters = confidence_intervals[k][:upper]

        # sample 5000 random parameter sets within the confidence bounds
        random_parameters = [[rand() * (upper_bound_parameters[i] - lower_bound_parameters[i]) + lower_bound_parameters[i] for i in eachindex(lower_bound_parameters)] for _ in 1:5000]

        models = [DecayODE(p[1], p[2]) for p in random_parameters]
        predictions_ci = [predict(m, t_pred) for m in models]
        y_lower = map(t -> minimum([pred[t] for pred in predictions_ci]), 1:length(t_pred))
        y_upper = map(t -> maximum([pred[t] for pred in predictions_ci]), 1:length(t_pred))

        # Plot the confidence intervals as shaded areas
        band!(ax, t_pred, y_lower, y_upper; 
        color=(COMPONENT_COLORS[k], 0.2), label="Component $k")
    end

    axislegend(ax; merge=true)

    f
end
```

