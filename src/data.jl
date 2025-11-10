"""
    MixtureData

Pre-processed data structure optimized for EM algorithm.
All missing data is handled during construction.

# Fields
- `subjects`: Vector of SubjectData, one per subject
- `n_subjects`: Number of subjects
- `n_measurements`: Number of measurements
- `n_total_obs`: Total number of observations
- `id_to_idx`: Dictionary mapping subject ID to index
"""
struct MixtureData{T<:Union{Real, Missing}, Y<:Union{Real, Missing}} 
    t::Vector{T}
    y::Matrix{Y}
    ids::Vector{Int}

    function MixtureData(t::AbstractVector{T}, y::AbstractMatrix{Y}, ids::AbstractVector{Int}) where {T<:Union{Real, Missing}, Y<:Union{Real, Missing}}
    length(t) == size(y, 1) || error("Length of t must match number of rows in y")
    length(t) == length(ids) || error("Length of t must match length of ids")

    # Convert ids to numbered IDs starting from 1
    id_map = Dict{Int, Int}(current_id => new_id for (new_id, current_id) in enumerate(unique(ids)))
    mapped_ids = [id_map[id] for id in ids]


    return new{T, Y}(collect(t), collect(y), collect(mapped_ids))
end

end



"""
    subset_view(data::MixtureData, subject_ids)

Get views of `t` and `y` for a subset of subject IDs.

# Arguments
- `data::MixtureData`: The mixture data
- `subject_ids`: Vector of subject IDs to extract

# Returns
- `t_view`: View of time points for selected subjects
- `y_view`: View of observations for selected subjects
"""
function subset_view(data::MixtureData, subject_ids::Vector{Int})
    mask = in.(data.ids, Ref{Vector{Int}}(subject_ids))
    return view(data.t, mask), view(data.y, :, mask)
end

function subset_view(data::MixtureData, subject_id::Int)
    mask = data.ids .== subject_id
    return view(data.t, mask), view(data.y, mask, :)
end

function subject_data(data::MixtureData, subject_id::Int)
    mask = data.ids .== subject_id
    return copy(data.t[mask]), copy(data.y[mask, :]), copy(data.ids[mask])
end

function sample_subset_with_replacement(data::MixtureData, n_ids::Int; rng::AbstractRNG=Random.GLOBAL_RNG)

    # sample subject IDs with replacement
    unique_ids = unique(data.ids)
    sampled_ids = rand(rng, unique_ids, n_ids)

    t_sample, y_sample, ids_sample = subject_data(data, sampled_ids[1])

    for i in 2:n_ids
        t_i, y_i, ids_i = subject_data(data, sampled_ids[i])
        t_sample = vcat(t_sample, t_i)
        y_sample = vcat(y_sample, y_i)
        ids_sample = vcat(ids_sample, ids_i)
    end
    return MixtureData(t_sample, y_sample, ids_sample)
end