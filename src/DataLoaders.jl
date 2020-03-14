module DataLoaders

using MLDataUtils
using ThreadPools: QueuePool, ResultIterator, results
using LearnBase
using Random: shuffle!
import Base: length, show

include("./collate.jl")


mutable struct DataLoader
    dataset
    batchsize::Integer
    shuffle::Bool
    numworkers::Integer
    transformfn
    collatefn
    droplast::Bool
    _results::Union{Nothing, ResultIterator}
end

"""
    DataLoader(dataset, batchsize; kwargs...)

# Arguments

- `dataset`: A data container supporting the `LearnBase` data access pattern
- `batchsize::Integer = 1`: Number of samples to batch together

# Keyword arguments

- `shuffle::Bool = true`: Whether to shuffle the observations before iterating
- `numworkers::Integer = max(1, Threads.nthreads() - 1)`: Number of workers to
  spawn to load data in parallel
- `transformfn`: Function that is applied to individual samples before batching
- `collatefn`: Function that collates multiple samples into a batch. For default
  behavior, see [`collate`](@ref)
- `droplast::Bool = false`: Whether to drop the last batch when `nobs(dataset)` is
  not divisible by `batchsize`. `true` ensures all batches have the same size, but
  some samples might be dropped

"""
function DataLoader(
        dataset,
        batchsize = 1;
        shuffle = true,
        numworkers = max(1, Threads.nthreads() - 1),
        transformfn = identity,
        collatefn = collate,
        droplast = false)
    return DataLoader(
        dataset,
        batchsize,
        shuffle,
        numworkers,
        transformfn,
        collatefn,
        droplast,
        nothing)
end

Base.length(dl::DataLoader) = (dl.droplast ? fld : cld)(nobs(dl.dataset), dl.batchsize)

function Base.iterate(dl::DataLoader)
    batchindices = getbatchindices(nobs(dl.dataset), dl.batchsize, dl.shuffle, dl.droplast)
    dl._results = results(createworkerpool(dl, batchindices))
    return Base.iterate(dl, nothing)
end

function Base.iterate(dl::DataLoader, state)
    iter = Base.iterate(dl._results, state)
    !isnothing(iter) || return nothing

    if iter[1] isa Exception
        close(dl._results.pool.outq)
        close(dl._results.pool.inq)
        throw(iter[1])
    else
        return iter
    end
end

function createworkerpool(dl::DataLoader, batchindices)
    workerpool = QueuePool(2, dl.numworkers)

    @async begin
        for idxs in batchindices
            put!(
                workerpool,
                loadbatch,
                dl.dataset, idxs, dl.collatefn, dl.transformfn)
        end
        close(workerpool)
    end
    return workerpool
end

show(io::IO, dl::DataLoader) = print(io, "DataLoader (", length(dl), " batches, ", dl.numworkers, " threads)")


# Utils

function getbatchindices(n, batchsize, shuffle = true, droplast = false)
    indices = collect(1:n)
    if shuffle
        shuffle!(indices)
    end

    batchindices = []
    for i in collect(1:batchsize:n)
        to = min(i+batchsize-1, n)
        if droplast && ((to - i) < batchsize - 1)
            break
        end
        push!(batchindices, indices[i:to])
    end
    return batchindices
end


function loadbatch(dataset, idxs, collatefn, transformfn)
    try
        batch = collatefn([transformfn(getobs(dataset, idx)) for idx in idxs])
        return batch
    catch e
        return e
    end
end

export DataLoader, collate

end # module
