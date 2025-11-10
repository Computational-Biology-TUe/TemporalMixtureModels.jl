# ============================================================================
# Composite Component
# ============================================================================

"""
    CompositeComponent

A component that combines multiple sub-components for different measurements.
"""
struct CompositeComponent <: Component
    components::Vector{Component}
    param_ranges::Vector{UnitRange{Int}}  # Pre-computed parameter ranges
    y_ranges::Vector{UnitRange{Int}}  # Ranges for which measurement corresponds to which component
    
    function CompositeComponent(components::Vector{<:Component}, y_ranges::Vector{UnitRange{Int}})
        @assert length(components) == length(y_ranges) "Number of components must match number of y_ranges"
        # Pre-compute parameter ranges for fast indexing
        param_ranges = UnitRange{Int}[]
        start_idx = 1
        for comp in components
            n_params = n_parameters(comp)
            push!(param_ranges, start_idx:(start_idx + n_params - 1))
            start_idx += n_params
        end
        new(components, param_ranges, y_ranges)
    end
end

function n_parameters(m::CompositeComponent)
    return sum(n_parameters(c) for c in m.components)
end

function initialize_parameters(m::CompositeComponent)
    params = Float64[]
    for comp in m.components
        append!(params, initialize_parameters(comp))
    end
    return params
end

"""
Predict using composite component (optimized version)
"""
function predict(m::CompositeComponent, params::AbstractVector, 
                t::AbstractVector, inputs=nothing)
    n_obs = length(t)
    n_measurements = length(m.components)
    y_pred = zeros(n_obs, n_measurements)
    
    for (y_range, param_range, comp) in zip(m.y_ranges, m.param_ranges, m.components)
        comp_params = view(params, param_range)
        y_pred[:, y_range] = predict(comp, comp_params, t, inputs)
    end
    
    return y_pred
end

# ============================================================================
# Macro for Composite Component
# ============================================================================

"""
    @component begin ... end

Macro for specifying a composite component with multiple measurements.
"""
macro component(expr)
    components = []
    y_ranges = []
    
    if expr.head != :block
        error("@component requires a begin...end block")
    end
    
    for line in expr.args
        if line isa LineNumberNode || line === nothing
            continue
        end
        
        if line.head == :call && line.args[1] == :~
            lhs = line.args[2]
            rhs = line.args[3]
            
            if !(lhs isa Expr && lhs.head == :ref)
                error("Left-hand side must be of form y[index]")
            end
            
            if lhs.args[1] != :y
                error("Left-hand side must use variable 'y'")
            end
            
            idx = lhs.args[2]
            push!(components, esc(rhs))
            if idx isa Int
                push!(y_ranges, idx:idx)
            
            # if idx is an expression of the form a:b (this is now a bit crude but works, could be improved)
            elseif idx isa Expr
                start_idx = idx.args[2]
                end_idx = idx.args[3]
                push!(y_ranges, start_idx:end_idx)
            else
                println(typeof(idx))
                println(typeof(idx.head))
                error("Invalid index format in y[index]")
            end
        end
    end
    
    if isempty(components)
        error("Must specify at least one component")
    end
    
    return quote
        CompositeComponent(Component[$(components...)], UnitRange{Int}[$(y_ranges...)])
    end
end