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
    useprimary::Bool
    state::PoolState
    ntasks::Threads.Atomic{Int}
end


function WorkerPool(workerfn, args::AbstractVector{TArgs}; useprimary = false) where {TArgs}
    (useprimary || Threads.nthreads() > 1) ||
        error("Cannot load data off main thread with only one thread available. Pass `useprimary = true` or start Julia with >1 threads.")
    return WorkerPool{TArgs}(workerfn, collect(args), useprimary, Done, Threads.Atomic{Int}(0))
end


function run(pool::WorkerPool)
    pool.state = Running
    # set remaining tasks counter.
    pool.ntasks[] = length(pool.args)

    # watchdog that sends exception to main thread if a worker fails
    maintask = current_task()
    @async begin
        while pool.state !== Done
            if pool.state === Failed
                Base.throwto(
                    maintask,
                    PoolFailedException("Failed to process all tasks. $(pool.ntasks[]) unfinished tasks remaining"),
                )
            end
            sleep(0.1)
        end
    end

    @qbthreads for args in pool.args
        #for args in pool.args  # uncomment for debugging
        # task error handling
        pool.state !== Failed || error("Shutting down worker $(Threads.threadid())")
        try
            # execute task
            pool.workerfn(args...)
            Threads.atomic_add!(pool.ntasks, -1)
        catch e
            @error "Exception while executing task on worker $(Threads.threadid()). Shutting down WorkerPool." e =
                e
            pool.state = Failed
            rethrow()
        end
    end

    # Tasks completed successfully
    pool.state = Done
end
