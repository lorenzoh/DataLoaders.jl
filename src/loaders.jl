
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

mutable struct DataLoaderBuffered{TData,TElem}
    data::TData
    buffers::Vector{TElem}
    current::TElem
    nthreads::Int
    useprimary::Bool
    state::LoaderState
end

function DataLoaderBuffered(
    data;
    useprimary = false,
    nthreads = Threads.nthreads() - Int(!useprimary),
)

    # create buffers
    obs = getobs(data, 1)
    buffers = [obs]
    for _ = 1:nthreads
        push!(buffers, deepcopy(obs))
    end

    return DataLoaderBuffered(data, buffers, obs, nthreads, useprimary, Done)
end


Base.length(dl::DataLoaderBuffered) = nobs(dl.data)

function workerfn(dl, ch_buffers, ch_results, i, t)
    try
        buf = take!(ch_buffers)
        buf = getobs!(buf, dl.data, i)
        put!(ch_results, buf)
    catch e
        dl.state = Failed
        close(ch_buffers)
        close(ch_results)
        @error "Error on worker $(Threads.threadid()), shutting workers down" error = e
        rethrow()
    end
end


function Base.iterate(dl::DataLoaderBuffered{TData,TElem}) where {TData,TElem}
    dl.state = Running
    dl.current = dl.buffers[1]

    ch_buffers = Channel{TElem}(dl.nthreads + 1)
    ch_results = Channel{TElem}(dl.nthreads)
    index = 0
    maintask = current_task()
    state = (ch_buffers, ch_results, index)

    # fill buffer channel
    @async begin
        for buf in dl.buffers[2:end]
            put!(ch_buffers, buf)
        end
    end
    # start tasks
    @async begin
        try
            if dl.useprimary
                @qthreads for i = 1:nobs(dl.data)
                    if dl.state !== Failed
                        workerfn(dl, ch_buffers, ch_results, i, maintask)
                    else
                        @info "Shutting down worker $(Thread.threadid())"
                        error("Shutting down worker $(Thread.threadid())")
                    end
                end
            else
                @qbthreads for i = 1:nobs(dl.data)
                    if dl.state !== Failed
                        workerfn(dl, ch_buffers, ch_results, i, maintask)
                    else
                        error("Shutting down worker $(Thread.threadid())")
                    end
                end
            end
        catch e
            @error "Error while filling task queue" error = e
            dl.state = Failed
            rethrow()
        end
    end
    return Base.iterate(dl, state)
end


function Base.iterate(dl::DataLoaderBuffered, state)
    ch_buffers, ch_results, index = state
    try
        if index < nobs(dl.data)
            # Put previously in use buffer back into channel
            put!(ch_buffers, dl.current)
            # Take the latest result
            dl.current = take!(ch_results)
            return dl.current, (ch_buffers, ch_results, index + 1)
        else
            dl.state = Done
            return nothing
        end
    catch e
        dl.state = Failed
        if e isa InvalidStateException
            error("Worker task failed")
        end
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
