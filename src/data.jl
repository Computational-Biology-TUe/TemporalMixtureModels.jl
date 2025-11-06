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

