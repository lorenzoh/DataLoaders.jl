
#=

on creation: Done
on iterate: Running
on finish iterate: Done
on error or interrupt: Failed

error can occur
- in `ìterate` under `@qthreads`
-

on error
- @qthreads is stopped
- workers are stopped
- results channel must be cleared and buffers filled back


=#

@enum LoaderState begin
    Done
    Running
    Failed
end
# Buffered

mutable struct DataLoaderBuffered{TData, TElem}
    data::TData
    buffers::Vector{TElem}
    ch_results::Channel{TElem}
    ch_buffers::Channel{TElem}
    current::TElem
    useprimary::Bool
    state::LoaderState
end

function DataLoaderBuffered(
    data;
    useprimary = false,
    )
    # get one observation as base for the buffers and to find out the element type
    obs = getobs(data, 1)
    T = typeof(obs)
    nthreads = Threads.nthreads() - Int(!useprimary)

    buffers = [obs]
    for _ in 1:nthreads
        push!(buffers, deepcopy(obs))
    end

    # Create a channel for the buffers and fill it with `nthreads` buffers
    ch_buffers = Channel{T}(nthreads + 1)
    @async begin
        for buf in buffers[2:end]
            put!(ch_buffers, buf)
        end
    end

    ch_results = Channel{T}(nthreads)

    return DataLoaderBuffered(data, ch_results, ch_buffers, obs, useprimary, Done)
end


Base.length(dl::DataLoaderBuffered) = nobs(dl.data)

function workerfn(dl, i)
    if dl.state === Failed
        # an error occured somewhere else
        @info "Shutting down worker $(Threads.threadid())"
        error("Shutting down worker")
    else
        try
            buf = take!(dl.ch_buffers)
            buf = getobs!(buf, dl.data, i)
            put!(dl.ch_results, buf)
        catch e
            dl.state = Failed
            @error "Error on worker $(Threads.threadid())" error=e
            error("Shutting down worker")
        end
    end
end

function Base.iterate(dl::DataLoaderBuffered)
    dl.state = Running
    @async begin


    end
    @async begin
        try
            if dl.useprimary
                @qthreads for i in 1:nobs(dl.data)
                    workerfn(dl, i)
                end
            else
                @qbthreads for i in 1:nobs(dl.data)
                    workerfn(dl, i)
                end
            end
        catch e
            @error "Error while filling task queue" error=e
            dl.state = Failed
            rethrow()
        end
    end
    return Base.iterate(dl, 0)
end

function Base.iterate(dl::DataLoaderBuffered, state)
    try
        if state < nobs(dl.data)
            # Put previously in use buffer back into channel
            put!(dl.ch_buffers, dl.current)
            # Take the latest result
            dl.current = take!(dl.ch_results)
            return dl.current, state + 1
        else
            dl.state = Done
            return nothing
        end
    catch e
        dl.state = Failed
        @error "Iterate failed" error=e
        rethrow()
    end
end


"""
handleerror(dl)

Cleans up the buffers channel and results channel
when an error occurs.
"""
function handleerror(dl)


end
