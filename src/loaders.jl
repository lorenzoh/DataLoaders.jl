

"""
    GetObsAsync(data, useprimary = false)
"""
struct GetObsAsync{TData}
    data::TData
    useprimary::Bool
    function GetObsAsync(data::TData, useprimary = false) where TData
        (useprimary || Threads.nthreads() > 1) ||
            error("Cannot load data off main thread with only one thread available. Pass `useprimary = true` or start Julia with > 1 threads.")
        return new{TData}(data, useprimary)
    end
end


Base.length(iterasync::GetObsAsync) = nobs(iterasync.data)

function Base.iterate(iterasync::GetObsAsync)
    resultschannel = Channel(Threads.nthreads() - Int(!iterasync.useprimary))

    workerpool = WorkerPool(1:nobs(iterasync.data), useprimary = iterasync.useprimary) do idx
        put!(resultschannel, getobs(iterasync.data, idx))
    end
    @async run(workerpool)

    return iterate(iterasync, (resultschannel, workerpool, 0))
end


function Base.iterate(iterasync::GetObsAsync, state)
    resultschannel, workerpool, index = state

    # Worker pool failed
    if workerpool.state === Failed
        error("Worker pool failed.")
    # Iteration complete
    elseif index >= nobs(iterasync.data)
        return nothing
    else
        return take!(resultschannel), (resultschannel, workerpool, index + 1)
    end
end


# Buffered version

"""
    BufferGetObsAsync(data; useprimary, nthreads)

Like `MLDataPattern.BufferGetObs` but preloads observations into a
buffer ring with multi-threaded workers.
"""
struct BufferGetObsAsync{TElem,TData}
    data::TData
    buffers::Vector{TElem}
    useprimary::Bool
end

function BufferGetObsAsync(data; useprimary = false)
    nthreads = Threads.nthreads() - Int(!useprimary)
    nthreads > 0 ||
        error("Cannot load data off main thread with only one thread available. Pass `useprimary = true` or start Julia with > 1 threads.")

    buffer = getobs(data, 1)
    buffers = [buffer]
    for _ ∈ 1:nthreads
        push!(buffers, deepcopy(buffer))
    end

    return BufferGetObsAsync(data, buffers, useprimary)
end


Base.length(iterasync::BufferGetObsAsync) = nobs(iterasync.data)


function Base.iterate(iterasync::BufferGetObsAsync)
    ringbuffer = RingBuffer(iterasync.buffers)

    workerpool = WorkerPool(1:nobs(iterasync.data), useprimary = iterasync.useprimary) do idx
        put!(ringbuffer) do buf
            getobs!(buf, iterasync.data, idx)
        end
    end
    @async run(workerpool)

    return iterate(iterasync, (ringbuffer, workerpool, 0))
end


function Base.iterate(iterasync::BufferGetObsAsync, state)
    ringbuffer, workerpool, index = state

    # Worker pool failed
    if workerpool.state === Failed
        error("Worker pool failed.")
    # Iteration complete
    elseif index >= nobs(iterasync.data)
        return nothing
    else
        return take!(ringbuffer), (ringbuffer, workerpool, index + 1)
    end
end
