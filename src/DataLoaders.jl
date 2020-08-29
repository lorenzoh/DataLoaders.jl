module DataLoaders

using MLDataPattern
using ThreadPools
using LearnBase
using Parameters


include("./collate.jl")
include("./batchview.jl")
include("./loaders.jl")



function DataLoader(dataset; buffered = true, useprimary = false)
    if buffered
        return DataLoaderBuffered(dataset; useprimary = useprimary)
    else
        return DataLoaderUnbuffered(dataset; useprimary = useprimary)
    end
end

"""
    BatchLoader(dataset, batchsize; kwargs...)

Iterates over collated batches of `batchsize`.

## Arguments

- `dataset`: A data container supporting the `LearnBase` data access pattern
- `batchsize::Integer`: Number of samples to batch together

## Keyword arguments

- `shuffle::Bool = true`: Whether to shuffle the observations before iterating
- `numworkers::Integer = max(1, Threads.nthreads() - 1)`: Number of workers to
  spawn to load data in parallel. The primary thread is kept free.
- `droplast::Bool = false`: Whether to drop the last batch when `nobs(dataset)` is
  not divisible by `batchsize`. `true` ensures all batches have the same size, but
  some samples might be dropped
"""
function BatchLoader(
        dataset, batchsize;
        buffered = true, collate = true, useprimary = false, droplast = true)
    data = BatchViewCollated(dataset, batchsize, droplast = droplast)
    return DataLoader(data; buffered = buffered, useprimary = useprimary)
end


export DataLoader, BatchLoader

end  # module
