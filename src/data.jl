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

    function MixtureData(t::AbstractVector{T}, y::AbstractMatrix{Y}, ids::AbstractVector{Int}) where {T<:Real, Y<:Union{Real, Missing}}
    length(t) == size(y, 1) || error("Length of t must match number of rows in y")
    length(t) == length(ids) || error("Length of t must match length of ids")
    check_compatible(ids, y)
    # Convert ids to numbered IDs starting from 1
    id_map = Dict{Int, Int}(current_id => new_id for (new_id, current_id) in enumerate(unique(ids)))
    mapped_ids = [id_map[id] for id in ids]


    return new{T, Y}(collect(t), collect(y), collect(mapped_ids))
end

end

function check_compatible(ids::Vector{Int}, y::Matrix{Y}) where {Y}

    # find number of missing entries per id in each column of y
    unique_ids = unique(ids)
    n_ids = length(unique_ids)
    n_cols = size(y, 2)
    for (i, uid) in enumerate(unique_ids)
        mask = ids .== uid
        total_entries = sum(mask)
        for j in 1:n_cols
            if total_entries == sum(ismissing.(y[mask, j]))
                throw(ArgumentError("Subject ID $uid has all missing entries in column $j of y. Make sure each subject has at least one observation per variable."))
            end
        end
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


"""
    example_bp_data(;n_subjects_drug=50, n_subjects_placebo=50, n_timepoints=5, rng::AbstractRNG=Random.GLOBAL_RNG)

Generate example blood pressure data for testing and examples.

# Arguments
- `n_subjects_drug`: Number of subjects in the drug group (default: 50)
- `n_subjects_placebo`: Number of subjects in the placebo group (default: 50)
- `n_timepoints`: Number of time points per subject (default: 5)
- `rng`: Random number generator (default: `Random.GLOBAL_RNG`)

# Returns
- `t`: Vector of time points
- `y`: Matrix of blood pressure measurements (systolic and diastolic)
- `ids`: Vector of subject IDs
- `class_labels`: Vector of class labels (1 for drug, 0 for placebo)

# Example
```julia
using TemporalMixtureModels: example_bp_data
t, y, ids, class_labels = example_bp_data(n_subjects_drug=30, n_subjects_placebo=30, n_timepoints=4)
```
"""
function example_bp_data(;n_subjects_drug=50, n_subjects_placebo=50, n_timepoints=5, rng::AbstractRNG=Random.GLOBAL_RNG)

    ids = Int[]
    t = Float64[]
    bp_sys = Float64[]
    bp_dia = Float64[]
    class_labels = Int[]

    tp = LinRange(0.0, 5.0, n_timepoints)

    for i in 1:n_subjects_drug

        base_sys = 120 + randn(rng)*5
        base_dia = 80 + randn(rng)*3

        for j in tp

            random_noise_sys = randn(rng)*5.1
            random_noise_dia = randn(rng)*3.2

            effect_noise_sys = randn(rng)*0.3
            effect_noise_dia = randn(rng)*0.14
            push!(ids, i)
            push!(t, Float64(j))
            systolic = base_sys - 8.8*j*(1+effect_noise_sys) + 0.8*j^2 + random_noise_sys
            diastolic = base_dia - 6.43*j*(1+effect_noise_dia) + 0.64*j^2 + random_noise_dia
            push!(bp_sys, systolic)
            push!(bp_dia, diastolic)
            push!(class_labels, 1)
        end
    end

    for i in n_subjects_drug+1:n_subjects_drug+n_subjects_placebo

        base_sys = 120 + randn(rng)*5
        base_dia = 80 + randn(rng)*3

        for j in tp

            random_noise_sys = randn(rng)*5.1
            random_noise_dia = randn(rng)*3.2

            effect_noise_sys = randn(rng)*0.3
            effect_noise_dia = randn(rng)*0.14
            push!(ids, i)
            push!(t, Float64(j))
            systolic = base_sys - j*effect_noise_sys + random_noise_sys
            diastolic = base_dia - j*effect_noise_dia + random_noise_dia
            push!(bp_sys, systolic)
            push!(bp_dia, diastolic)
            push!(class_labels, 0)
        end
    end

    return t, hcat(bp_sys, bp_dia), ids, class_labels
end