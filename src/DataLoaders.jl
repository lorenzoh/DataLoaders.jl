module DataLoaders

using MLDataPattern
using ThreadPools: QueuePool, ResultIterator, results
using LearnBase
using Random: shuffle!
import Base: length, show

include("./collate.jl")


mutable struct DataLoader
    dataset
    batchsize::Int
    shuffle::Bool
    numworkers::Union{Nothing, Int}
    transformfn
    collatefn
    droplast::Bool
    splitxyfn
    _results::Union{Nothing, ResultIterator}
end

"""
    DataLoader(dataset, batchsize; kwargs...)

# Arguments

- `dataset`: A data container supporting the `LearnBase` data access pattern
- `batchsize::Int = 1`: Number of samples to batch together

# Keyword arguments

- `shuffle::Bool = true`: Whether to shuffle the observations before iterating
- `numworkers::Int = max(1, Threads.nthreads() - 1)`: Number of workers to
  spawn to load data in parallel
- `transformfn`: Function that is applied to individual samples before batching
- `collatefn`: Function that collates multiple samples into a batch. For default
  behavior, see [`collate`](@ref)
- `droplast::Bool = false`: Whether to drop the last batch when `nobs(dataset)` is
  not divisible by `batchsize`. `true` ensures all batches have the same size, but
  some samples might be dropped
- `splitxyfn = splitxy`: Function that splits a batch into input and target. For
  default behavior, see [`splitxy`](@ref)
"""
function DataLoader(
        dataset,
        batchsize = 1;
        shuffle = true,
        numworkers = nothing,
        transformfn = identity,
        collatefn = collate,
        splitxyfn = identity,
        droplast = false)
    return DataLoader(
        dataset,
        batchsize,
        shuffle,
        numworkers,
        transformfn,
        collatefn,
        droplast,
        splitxyfn,
        nothing)
end

Base.length(dl::DataLoader) = (dl.droplast ? fld : cld)(nobs(dl.dataset), dl.batchsize)

function Base.iterate(dl::DataLoader)
    batchindices = getbatchindices(dl)
    dl._results = results(createworkerpool(dl, batchindices))
    return Base.iterate(dl, nothing)
end

function Base.iterate(dl::DataLoader, state)
    iter = Base.iterate(dl._results, state)
    !isnothing(iter) || return nothing

    if isnothing(iter)
        return
    end

    res, state = iter

    if res isa Tuple && res[2] isa Exception
        tid, e = res
        @error "Error on worker thread $tid, closing pool..." error = e
        closeinout(dl._results.pool)
        throw(e)
    else
        return iter
    end
end

function createworkerpool(dl::DataLoader, batchindices)
    if isnothing(dl.numworkers)
        workerpool = QueuePool(1, 1)
    else
        workerpool = QueuePool(2, min(Threads.nthreads() - 1, dl.numworkers))
    end

    @async begin
        try
            for idxs in batchindices
                put!(
                    workerpool,
                    loadbatch,
                    dl.dataset, idxs, dl.collatefn, dl.transformfn, dl.splitxyfn)
            end
        catch e
            println(e)
            rethrow(e)
        finally
            close(workerpool)
        end
    end
    return workerpool
end

Base.show(io::IO, dl::DataLoader) = print(io, "DataLoader (", length(dl), " batches, ", dl.numworkers, " threads)")


# Utils

function getbatchindices(n, batchsize, shuffle = true, droplast = false)
    indices = collect(1:n)
    if shuffle
        shuffle!(indices)
    end

    batchindices = Vector{Int}[]
    for i in collect(1:batchsize:n)
        to = min(i+batchsize-1, n)
        if droplast && ((to - i) < batchsize - 1)
            break
        end
        push!(batchindices, indices[i:to])
    end
    return batchindices
end

getbatchindices(dl::DataLoader) =
    getbatchindices(nobs(dl.dataset), dl.batchsize, dl.shuffle, dl.droplast)

"""
    loadbatch(dataset, idxs, collatefn, transformfn, splitxyfn)

Must not throw an error or Julia will crash! Errors are caught and returned
to the iterator to be rethrown on the main thread.
"""
function loadbatch(dataset, idxs, collatefn, transformfn, splitxyfn)
    try
        batch = collatefn([splitxyfn(transformfn(getobs(dataset, idx))) for idx in idxs])
        return batch
    catch e
        println(e)
        return (Threads.threadid(), e)
    end
end


splitxy(sample::NTuple{2}) = sample
splitxy(sample::Dict) = (sample[:x], sample[:y])


function closeinout(pool::QueuePool)
    @async begin
        close(pool.inq)
        close(pool.outq)
    end
end

function getsample(dl::DataLoader, idx; transform = true, split = false)
    sample = getobs(dl.dataset, idx)
    if transform
        sample = dl.transformfn(sample)
        if split
            sample = dl.splitxyfn(sample)
        end
    end
    return sample
end


export DataLoader, collate, getsample

end # module
