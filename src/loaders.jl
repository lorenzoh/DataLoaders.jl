

struct GetObsParallel{TData}
    data::TData
    useprimary::Bool
    function GetObsParallel(data::TData; useprimary = false) where {TData}
        (useprimary || Threads.nthreads() > 1) ||
            error("Cannot load data off main thread with only one thread available. Pass `useprimary = true` or start Julia with > 1 threads.")
        return new{TData}(data, useprimary)
    end
end


Base.length(iterparallel::GetObsParallel) = nobs(iterparallel.data)

function Base.iterate(iterparallel::GetObsParallel)
    resultschannel = Channel(Threads.nthreads() - Int(!iterparallel.useprimary))

    workerpool =
        WorkerPool(1:nobs(iterparallel.data), useprimary = iterparallel.useprimary) do idx
            put!(resultschannel, getobs(iterparallel.data, idx))
        end
    @async run(workerpool)

    return iterate(iterparallel, (resultschannel, workerpool, 0))
end


function Base.iterate(iterparallel::GetObsParallel, state)
    resultschannel, workerpool, index = state

    # Worker pool failed
    if workerpool.state === Failed
        error("Worker pool failed.")
        # Iteration complete
    elseif index >= nobs(iterparallel.data)
        return nothing
    else
        return take!(resultschannel), (resultschannel, workerpool, index + 1)
    end
end


# Buffered version

"""
    BufferGetObsParallel(data; useprimary = false)

Like `MLDataPattern.BufferGetObs` but preloads observations into a
buffer ring with multi-threaded workers.
"""
struct BufferGetObsParallel{TElem,TData}
    data::TData
    buffers::Vector{TElem}
    useprimary::Bool
end

function BufferGetObsParallel(data; useprimary = false)
    nthreads = Threads.nthreads() - Int(!useprimary)
    nthreads > 0 ||
        error("Cannot load data off main thread with only one thread available. Pass `useprimary = true` or start Julia with > 1 threads.")

    buffer = getobs(data, 1)
    buffers = [buffer]
    for _ ∈ 1:nthreads
        push!(buffers, deepcopy(buffer))
    end

    return BufferGetObsParallel(data, buffers, useprimary)
end


Base.length(iterparallel::BufferGetObsParallel) = nobs(iterparallel.data)


function Base.iterate(iterparallel::BufferGetObsParallel)
    ringbuffer = RingBuffer(iterparallel.buffers)

    workerpool =
        WorkerPool(1:nobs(iterparallel.data), useprimary = iterparallel.useprimary) do idx
            put!(ringbuffer) do buf
                getobs!(buf, iterparallel.data, idx)
            end
        end
    @async run(workerpool)

    return iterate(iterparallel, (ringbuffer, workerpool, 0))
end


function Base.iterate(iterparallel::BufferGetObsParallel, state)
    ringbuffer, workerpool, index = state

    # Worker pool failed
    if workerpool.state === Failed
        error("Worker pool failed.")
        # Iteration complete
    elseif index >= nobs(iterparallel.data)
        return nothing
    else
        return take!(ringbuffer), (ringbuffer, workerpool, index + 1)
    end
end


# functional interface

"""
    eachobsparallel(data; useprimary = false, buffered = true)

Parallel data iterator for data container `data`. Loads data on all
available threads (except the first if `useprimary` is `false`).

If `buffered` is `true`, uses `getobs!` to load samples inplace.

See also `MLDataPattern.eachobs`


!!! warning "Order"

    `eachobsparallel` does not guarantee that the samples
    are returned in the correct order.

"""
eachobsparallel(data; useprimary = false, buffered = true) =
    buffered ? BufferGetObsParallel(data, useprimary = useprimary) :
    GetObsParallel(data, useprimary = useprimary)
