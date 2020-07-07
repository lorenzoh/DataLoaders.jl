module DataLoaders

using MLDataPattern
using ThreadPools: QueuePool, ResultIterator, results
using LearnBase
using Random: shuffle!
using Parameters
using DocStringExtensions
import Base: length, show

include("./collate.jl")


"""
$(TYPEDEF)

## Fields

$(TYPEDFIELDS)

"""
@with_kw mutable struct DataLoader
    "A data container supporting the `LearnBase` data access pattern"
    dataset
    "Number of samples to batch together"
    batchsize::Int = 1
    "Whether to shuffle the observations before iterating"
    shuffle::Bool = true
    numworkers::Int = Threads.nthreads() - 1
    "Function that is applied to individual samples before batching"
    transformfn = identity
    "Function that collates multiple samples into a batch. For default
    behavior, see [`collate`](@ref)"
    collatefn = collate
    """
    Whether to drop the last batch when `nobs(dataset)` is
    not divisible by `batchsize`. `true` ensures all batches have the same size, but
    some samples might be dropped
    """
    droplast::Bool = false
end
DataLoader(dataset; kwargs...) = DataLoader(; dataset = dataset, kwargs...)

Base.length(dl::DataLoader) = (dl.droplast ? fld : cld)(nobs(dl.dataset), dl.batchsize)

function Base.iterate(dl::DataLoader)
    batchindices = getbatchindices(dl)
    resultpool = results(createworkerpool(dl, batchindices))
    return Base.iterate(dl, (resultpool, nothing))
end

function Base.iterate(dl::DataLoader, state)
    resultpool, poolstate = state

    iter = Base.iterate(resultpool, poolstate)
    !isnothing(iter) || return nothing

    if isnothing(iter)
        return
    end

    res, poolstate_ = iter

    if res isa Tuple && res[2] isa Exception

        tid, e = res
        @error "Error on worker thread $tid, closing pool..." error = e
        closeinout(resultpool.pool)
        return e
    else
        return (res, (resultpool, poolstate_))
    end
end

function createworkerpool(dl::DataLoader, batchindices)
    if dl.numworkers == 1
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
                    dl.dataset, idxs, dl.collatefn, dl.transformfn)
            end
            close(workerpool)
        catch e
            @error "Error while filling worker pool with tasks" error=e
            closeinout(workerpool)
            rethrow(e)
        end
    end
    return workerpool
end


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
function loadbatch(dataset, idxs, collatefn, transformfn)
    try
        batch = collatefn([transformfn(getobs(dataset, idx)) for idx in idxs])
        return batch
    catch e
        println(e)
        rethrow(e)
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

function getsample(dl::DataLoader, idx; transform = true)
    sample = getobs(dl.dataset, idx)
    if transform
        sample = dl.transformfn(sample)
    end
    return sample
end


export DataLoader, collate, getsample

end # module
