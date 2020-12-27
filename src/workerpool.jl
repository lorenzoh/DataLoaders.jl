import Base: put!, wait, isready, take!, fetch

mutable struct ValueChannel{T} <: AbstractChannel{T}
    v::T
    cond_take::Condition    # waiting for data to become available
    ValueChannel(v) = new{typeof(v)}(v, Condition())
end

function put!(c::ValueChannel, v)
    c.v = v
    notify(c.cond_take)
    return c
end

take!(c::ValueChannel) = fetch(c)

isready(c::ValueChannel) = true

fetch(c::ValueChannel) = c.v

wait(c::ValueChannel) = wait(c.cond_take)

@enum PoolState begin
    Done
    Running
    Failed
end

struct PoolFailedException <: Exception
    s::Any
end

mutable struct WorkerPool{TArgs}
    workerfn::Any
    args::Vector{TArgs}
    usethreads::Bool
    useprimary::Bool
    state::ValueChannel{PoolState}
end


function WorkerPool(workerfn, args::AbstractVector{TArgs}; usethreads = true, useprimary = false) where {TArgs}
    if usethreads
        (useprimary || nthreads() > 1) ||
            error("Cannot load data off main thread with only one thread available. Pass `useprimary = true` or start Julia with > 1 threads.")
    else
        (useprimary || nworkers() > 1) ||
            error("Cannot load data off main thread with only one process available. Pass `useprimary = true` or start Julia with > 1 processes.")
    end
    return WorkerPool{TArgs}(workerfn, collect(args), usethreads, useprimary, ValueChannel(Done))
end


function run(pool::WorkerPool{TArgs}) where TArgs
    @unpack workerfn, usethreads, useprimary, state = pool
    put!(state, Running)

    # watchdog that sends exception to main thread if a worker fails
    maintask = current_task()
    watchdog = @async begin
        while fetch(state) !== Done
            if fetch(state) === Failed
                Base.throwto(
                    maintask,
                    PoolFailedException("Failed to process all tasks. $(length(tasks)) unfinished tasks remaining"),
                )
            end
            sleep(0.1)
        end
    end
    
    if usethreads
        (useprimary ? qforeach : qbforeach)(pool.args) do args
            inloop(state, workerfn, threadid(), args)
        end
    else
        tasks = Channel{TArgs}(Inf)
        foreach(a -> put!(tasks, a), pool.args)
        close(tasks)
        remote_state = RemoteChannel(() -> state)
        remote_tasks = RemoteChannel(() -> tasks)
        @sync for id in (useprimary ? procs() : workers())
            @spawnat id on_worker(remote_tasks, remote_state, workerfn, usethreads, useprimary)
        end
    end

    # Tasks completed successfully
    put!(state, Done)
    wait(watchdog)
end

function inloop(state, workerfn, id, args)
    try
        # execute task
        workerfn(args...)
    catch e
        display(stacktrace())
        @error "Exception while executing task on worker $id. Shutting down WorkerPool." e =
            e stacktrace = stacktrace() args = args
        put!(state, Failed)
        rethrow()
    end
end

function on_worker(tasks, state, workerfn, usethreads, useprimary)
    # task error handling
    id = usethreads ? threadid() : myid()
    !useprimary && id == 1 && return
    while isready(tasks)
        args = try take!(tasks) catch e break end
        fetch(state) !== Failed || error("Shutting down worker $id")
        inloop(state, workerfn, id, args)
    end
end