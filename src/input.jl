# Specifies helpful functions for handling input data, to make sure the model runs fast but is also easy to use.

# internally, we use a MixtureData struct to organise the data
# this struct is not exported, and will be built internally from a dataframe
abstract type MixtureData end

# Sample struct for one (time, value) point
struct TimedValue{T<:Real}
    id::Int
    t::T
    y::T
end

TimedVector = StructArray{TimedValue}

struct UnivariateGroupedTimeSeries
    data::Dict{Int, SubArray}
    time::Dict{Int, SubArray}
end

struct MultivariateGroupedTimeSeries
    data::Dict{Tuple{Int,Symbol}, SubArray}
    time::Dict{Tuple{Int,Symbol}, SubArray} # Time points for each variable
end

struct UnivariateMixtureData <: MixtureData
    data::TimedVector
    ids::Vector{Int}
    grouped_view::UnivariateGroupedTimeSeries
end

struct MultivariateMixtureData <: MixtureData
    data::Dict{Symbol, TimedVector}
    ids::Vector{Int}
    variables::Vector{Symbol}
    grouped_view::MultivariateGroupedTimeSeries
end

function UnivariateGroupedTimeSeries(data)
    grouped = Dict{Int, SubArray}()
    timepoints = Dict{Int, SubArray}()
    id_indices = Dict{Int, Vector{Int}}()
    for (i, row) in enumerate(data)
        push!(get!(id_indices, row.id, Int[]), i)
    end
    for (id, idxs) in id_indices
        grouped[id] = view(data.y, idxs)
        timepoints[id] = view(data.t, idxs)
    end
    return UnivariateGroupedTimeSeries(grouped, timepoints)
end

function MultivariateGroupedTimeSeries(data::Dict{Symbol, <:TimedVector}, variables::Vector{Symbol})
    grouped = Dict{Tuple{Int,Symbol}, SubArray}()
    timepoints = Dict{Tuple{Int,Symbol}, SubArray}()
    for var in variables
        id_indices = Dict{Int, Vector{Int}}()
        for (i, row) in enumerate(data[var])
            push!(get!(id_indices, row.id, Int[]), i)
        end
        for (id, idxs) in id_indices
            grouped[(id, var)] = view(data[var].y, idxs)
            timepoints[(id, var)] = view(data[var].t, idxs)
        end
    end
    return MultivariateGroupedTimeSeries(grouped, timepoints)
end

function UnivariateMixtureData(data::TimedVector)
    ids = unique(data.id)
    grouped_view = UnivariateGroupedTimeSeries(data)
    return UnivariateMixtureData(data, ids, grouped_view)
end

function MultivariateMixtureData(data::Dict{Symbol, <:TimedVector})
    variables = keys(data)
    ids = unique(vcat([data[var].id for var in variables]...))
    grouped_view = MultivariateGroupedTimeSeries(data, collect(variables))
    return MultivariateMixtureData(data, ids, collect(variables), grouped_view)
end

function get_subset(X::UnivariateMixtureData, ids::Vector{Int})
    subset_data = X.data[X.data.id .∈ Ref(ids)]
    return UnivariateMixtureData(subset_data)
end

function get_subset_with_replacement(X::UnivariateMixtureData, ids::Vector{Int})
    subset_data = TimedVector(id = Int[], t = Float64[], y = Float64[])
    # assign new ids to the resampled data
    new_id = 1
    for id in ids
        subset_id_data = X.data[X.data.id .== id]
        append!(subset_data.id, fill(new_id, length(subset_id_data.id)))
        append!(subset_data.t, subset_id_data.t)
        append!(subset_data.y, subset_id_data.y)
        new_id += 1
    end
    return UnivariateMixtureData(subset_data)
end

function get_subset(X::MultivariateMixtureData, ids::Vector{Int})
    subset_data = Dict{Symbol, TimedVector}()
    for var in X.variables
        subset_data[var] = X.data[var][X.data[var].id .∈ Ref(ids)]
    end
    return MultivariateMixtureData(subset_data)
end

function get_subset_with_replacement(X::MultivariateMixtureData, ids::Vector{Int})
    subset_data = Dict{Symbol, TimedVector}()
    # assign new ids to the resampled data
    for var in X.variables
        new_id = 1
        subset_data[var] = TimedVector(id = Int[], t = Float64[], y = Float64[])
        for id in ids
            subset_id_data = X.data[var][X.data[var].id .== id]
            append!(subset_data[var].id, fill(new_id, length(subset_id_data.id)))
            append!(subset_data[var].t, subset_id_data.t)
            append!(subset_data[var].y, subset_id_data.y)
            new_id += 1
        end
    end
    return MultivariateMixtureData(subset_data)
end

function _prepare_data(df::DataFrame; 
    id_col = "id", time_col = "time", value_col = "value", var_name_col = "var_name",
    ignore_cols::Vector{Symbol}=Symbol[])

    # check how many columns we have besides the columns to ignore
    value_cols = setdiff(names(df), ignore_cols)
    if length(value_cols) == 3
        return _df_to_univariate(df, id_col, time_col, value_col)
    elseif length(value_cols) == 4
        return _df_to_multivariate(df, id_col, time_col, var_name_col, value_col)
    else
        throw(ArgumentError("The input dataframe contains an unexpected number of columns. It should contain exactly 3 columns (id, time, value) for univariate data, or 4 columns (id, time, var_name, value) for multivariate data. This dataframe contains $(length(value_cols)) columns: $(value_cols), after ignoring the columns: $(ignore_cols)."))
    end

end

function _df_to_univariate(df::DataFrame, id_col, time_col, value_col)
    if !(id_col in names(df)) || !(time_col in names(df)) || !(value_col in names(df))
        throw(ArgumentError("DataFrame must contain specified id, time and value columns"))
    end

    idtype = eltype(promote(df[!,id_col]...))
    valtype = eltype(promote(df[!,value_col]...))

    ids = convert(Vector{idtype}, df[!,id_col])
    t = convert(Vector{valtype}, df[!,time_col])
    y = convert(Vector{valtype}, df[!,value_col])

    meas = TimedVector(id = ids, t = t, y = y)
    return UnivariateMixtureData(meas)
end

function _df_to_multivariate(df::DataFrame, id_col, time_col, var_name_col, value_col)
    if !(id_col in names(df)) || !(time_col in names(df)) || !(var_name_col in names(df)) || !(value_col in names(df))
        throw(ArgumentError("DataFrame must contain specified id, time, var_name and value columns"))
    end

    idtype = eltype(promote(df[!,id_col]...))
    valtype = eltype(promote(df[!,value_col]...))

    grouped = Dict{Symbol, TimedVector}()
    for var in unique(df[!, var_name_col])
        subset = df[df[!, var_name_col] .== var, :]
        grouped[Symbol(var)] = TimedVector(id = convert(Vector{idtype}, subset[!,id_col]), t = convert(Vector{valtype}, subset[!,time_col]), y = convert(Vector{valtype}, subset[!,value_col]))
    end
    return MultivariateMixtureData(grouped)
end
