

struct GetObsParallel{TData}
    data::TData
    usethreads::Bool
    useprimary::Bool
    maxquesize::Int
    function GetObsParallel(data::TData; usethreads = true, useprimary = false, maxquesize = nothing) where {TData}
        if usethreads
            (useprimary || nthreads() > 1) ||
                error("Cannot load data off main thread with only one thread available. Pass `useprimary = true` or start Julia with > 1 threads.")
        else
            (useprimary || nworkers() > 1) ||
                error("Cannot load data off main thread with only one process available. Pass `useprimary = true` or start Julia with > 1 processes.")
        end
        maxquesize = something(maxquesize, usethreads ? nthreads() : nworkers())
        return new{TData}(data, usethreads, useprimary, maxquesize)
    end
end


Base.length(iterparallel::GetObsParallel) = nobs(iterparallel.data)

function Base.iterate(iterparallel::GetObsParallel)
    resultschannel = if iterparallel.usethreads
        Channel(iterparallel.maxquesize)
    else
        RemoteChannel(() -> Channel(iterparallel.maxquesize))
    end

    workerpool =
        WorkerPool(1:nobs(iterparallel.data), usethreads = iterparallel.usethreads, useprimary = iterparallel.useprimary) do idx
            put!(resultschannel, getobs(iterparallel.data, idx))
        end
    task = @async run(workerpool)

    return iterate(iterparallel, (task, resultschannel, workerpool, 0))
end


function Base.iterate(iterparallel::GetObsParallel, state)
    task, resultschannel, workerpool, index = state

    # Worker pool failed
    if fetch(workerpool.state) === Failed
        error("Worker pool failed.")
        # Iteration complete
    elseif index >= nobs(iterparallel.data)
        close(resultschannel)
        wait(task)
        return nothing
    else
        return take!(resultschannel), (task, resultschannel, workerpool, index + 1)
    end
end


# Buffered version

"""
    BufferGetObsParallel(data; usethreads = true, useprimary = false, maxquesize = nothing)

Like `MLDataPattern.BufferGetObs` but preloads observations into a
buffer ring with multi-threaded workers.
"""
struct BufferGetObsParallel{TElem,TData}
    data::TData
    buffers::Vector{TElem}
    usethreads::Bool
    useprimary::Bool
    maxquesize::Int
end

Base.show(io::IO, bufparallel::BufferGetObsParallel) = print(io, "eachobsparallel($(bufparallel.data))")

function BufferGetObsParallel(data; usethreads = true, useprimary = false, maxquesize = nothing)
    if usethreads
        (useprimary || nthreads() > 1) ||
            error("Cannot load data off main thread with only one thread available. Pass `useprimary = true` or start Julia with > 1 threads.")
    else
        (useprimary || nworkers() > 1) ||
            error("Cannot load data off main thread with only one process available. Pass `useprimary = true` or start Julia with > 1 processes.")
    end

    buffer = getobs(data, 1)
    buffers = [buffer]
    maxquesize = something(maxquesize, usethreads ? nthreads() : nworkers())
    for _ ∈ 1:maxquesize
        push!(buffers, deepcopy(buffer))
    end

    return BufferGetObsParallel(data, buffers, usethreads, useprimary, maxquesize)
end


Base.length(iterparallel::BufferGetObsParallel) = nobs(iterparallel.data)


function Base.iterate(iterparallel::BufferGetObsParallel)
    if iterparallel.usethreads
        resultschannel = RingBuffer(iterparallel.buffers)
        workerpool =
            WorkerPool(1:nobs(iterparallel.data), useprimary = iterparallel.useprimary) do idx
                put!(resultschannel) do buf
                    getobs!(buf, iterparallel.data, idx)
                end
            end
    else
        resultschannel = RemoteChannel(() -> Channel(iterparallel.maxquesize))
        workerpool =
            WorkerPool(1:nobs(iterparallel.data), usethreads=iterparallel.usethreads, useprimary = iterparallel.useprimary) do idx
                put!(resultschannel, getobs(iterparallel.data, idx))
            end
    end
    task = @async run(workerpool)

    return iterate(iterparallel, (task, resultschannel, workerpool, 0))
end


function Base.iterate(iterparallel::BufferGetObsParallel, state)
    task, resultschannel, workerpool, index = state

    # Worker pool failed
    if fetch(workerpool.state) === Failed
        error("Worker pool failed.")
        # Iteration complete
    elseif index >= nobs(iterparallel.data)
        close(resultschannel)
        wait(task)
        return nothing
    else
        return take!(resultschannel), (task, resultschannel, workerpool, index + 1)
    end
end


# functional interface

"""
    eachobsparallel(data; usethreads = true, useprimary = false, buffered = true, maxquesize = nothing)

Parallel data iterator for data container `data`. Loads data on all
available threads (except the first if `useprimary` is `false`).

If `buffered` is `true`, uses `getobs!` to load samples inplace.

See also `MLDataPattern.eachobs`


!!! warning "Order"

    `eachobsparallel` does not guarantee that the samples
    are returned in the correct order.

"""
eachobsparallel(data; usethreads = true, useprimary = false, buffered = true, maxquesize = nothing) =
    buffered ? BufferGetObsParallel(data,  usethreads = usethreads, useprimary = useprimary, maxquesize = maxquesize) :
    GetObsParallel(data, usethreads = usethreads, useprimary = useprimary, maxquesize = maxquesize)
