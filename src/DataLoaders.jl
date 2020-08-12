#=
TODO:
- implement unbuffered version
- document
- check how errors are handled
- format structs repr
- propagate errors to workers
    - interrupt main thread when worker fails. currently stuck in take! or put!
    - clean up channels on error. (all buffers should be returned to ch_buffers)

# Ideas

Multiple DataIterators:
- DataIterParallelBuffered
- DataIterParallel

Data Views:
- BatchViewCollated

DataLoader interface:
- DataLoader(ds, batchsize, [TDataIter])
=#
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


function BatchLoader(
        dataset, batchsize;
        buffered = true, collate = true, useprimary = false, droplast = true)
    data = BatchViewCollated(dataset, batchsize, droplast = droplast)
    return DataLoader(data; buffered = buffered, useprimary = useprimary)
end


export DataLoader, BatchLoader

end  # module
