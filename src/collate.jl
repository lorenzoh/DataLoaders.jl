
function collate(samples::AbstractVector{<:Tuple})
    !isempty(samples) || return samples
    return tuple([collate([s[key] for s in samples]) for key in keys(samples[1])]...)
end


function collate(samples::AbstractVector{<:Dict})
    !isempty(samples) || return samples
    return Dict(key => collate([s[key] for s in samples]) for key in keys(samples[1]))
end

function collate(samples::AbstractVector{<:NamedTuple})
    !isempty(samples) || return samples
    return (;(key => collate([s[key] for s in samples]) for key in keys(samples[1]))...)
end

collate(samples::AbstractVector{<:AbstractArray{T, N}}) where {T, N} = cat(samples...; dims = N + 1)

collate(samples::AbstractVector) = samples
