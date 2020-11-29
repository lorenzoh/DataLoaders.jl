module DataLoaders

using MLDataPattern
using ThreadPools
using LearnBase
using Parameters


include("./ringbuffer.jl")
include("./workerpool.jl")
include("./collate.jl")
include("./batchview.jl")
include("./loaders.jl")


"""
    DataLoader(
        data, batchsize = 1;
        partial = true,
        collate = true,
        buffered = collate,
        parallel = Threads.nthreads() > 1,
        useprimary = false,
    )

Utility for creating iterators of container `data` with a familiar interface
for PyTorch users.

## Arguments

- `data`: A data container supporting the `LearnBase` data access pattern
- `batchsize = 1`: Number of samples to batch together. Disable batching
  by setting to `nothing`.

## Keyword arguments

- `partial::Bool = true`: Whether to include the last batch when `nobs(dataset)` is
  not divisible by `batchsize`. `true` ensures all batches have the same size, but
  some samples might be dropped.
- `buffered::Bool = collate`: If `buffered` is `true`, loads data inplace
  using `getobs!`. See [Data containers](../docs/datacontainers.md) for details
  on buffered loading.
- `parallel::Bool = Threads.nthreads() > 1)`: Whether to load data
  in parallel, keeping the primary thread is. Default is `true` if
  more than one thread is available.
- `useprimary::Bool = false`: If `false`, keep the main thread free when loading
  data in parallel. Is ignored if `parallel` is `false`.

## Examples

`DataLoader(data, 16) === `[`BufferGetObsParallel`](#)`(`[`batchviewcollated`](#)`(data, 16))`

"""
function DataLoader(
        data,
        batchsize = 1;
        collate = !isnothing(batchsize),
        buffered = collate,
        partial = true,
        useprimary = Threads.nthreads() == 1,
    )

    Threads.nthreads() > 1 || useprimary || error(
        "Julia is running with one thread only, either pass `useprimary = true` or " *
        "start Julia with multiple threads by passing " *
        "the `-t n` option or setting the `JULIA_NUM_THREADS` " *
        "environment variable before starting Julia.")

    batchwrapper = if isnothing(batchsize)
        identity
    elseif collate
        data -> batchviewcollated(data, batchsize; partial = partial)
    else
        partial == false || error("Partial batches not yet supported for non-collated batches")
        data -> batchview(data, size = batchsize)
    end

    loadwrapper = data -> eachobsparallel(data; useprimary = useprimary, buffered = buffered)

    return loadwrapper(batchwrapper(data))
end


export DataLoader, eachobsparallel, batchviewcollated

end  # module
