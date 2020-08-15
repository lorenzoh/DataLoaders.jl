# DataLoaders

[![Build Status](https://github.com/lorenzoh/DataLoaders.jl/workflows/CI/badge.svg)](https://github.com/lorenzoh/DataLoaders.jl/actions)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://lorenzoh.github.io/DataLoaders.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://lorenzoh.github.io/DataLoaders.jl/dev)

A parallel iterator for large machine learning datasets that don't fit into memory inspired by PyTorch's `DataLoader` class.

It uses [`ThreadPools.jl`](https://github.com/tro3/ThreadPools.jl) to load data in parallel while keeping the primary thread free. It also supports loading data fully inplace to avoid unnecessary allocations.

Utilizing `LearnBase.jl`'s [Data Access Pattern](https://mldatautilsjl.readthedocs.io/en/latest/data/pattern.html), many data containers work out of the box and custom containers are easily supported by implementing `LearnBase.getobs` and `LearnBase.nobs`, (and optionally `LearnBase.getobs!` for inplace data loading).

## Usage


### BatchLoader(dataset, batchsize)

Iterates over observations 

#### Arguments

- `dataset`: A data container supporting the `LearnBase` data access pattern
- `batchsize::Integer`: Number of samples to batch together

##### Keyword arguments

- `shuffle::Bool = true`: Whether to shuffle the observations before iterating
- `numworkers::Integer = max(1, Threads.nthreads() - 1)`: Number of workers to
  spawn to load data in parallel. The primary thread is kept free.
- `transformfn`: Function that is applied to individual samples before batching
- `collatefn`: Function that collates multiple samples into a batch. For default
  behavior, see [`collate`](@ref)
- `droplast::Bool = false`: Whether to drop the last batch when `nobs(dataset)` is
  not divisible by `batchsize`. `true` ensures all batches have the same size, but
  some samples might be dropped


### Simple example

```julia
dataset = ([1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12])
dataloader = DataLoader(dataset, 2, numworkers = 1)

for batch in dataloader
    # do your stuff
end
```

Note: if your dataset fits into memory like in this toy example, you don't need parallel workers

### Custom data container example

```julia
using Images: load
import LearnBase

# Custom data container
struct ImageDataset
    files::AbstractVector{AbstractString}
end

# Implementing `LearnBase.jl` interface
LearnBase.nobs(ds::ImageDataset) = length(ds.files)
LearnBase.getobs(ds::ImageDataset, idx::Integer) =
    load(ds.files[idx], view = true)
# Optionally implement inplace getter for performance
# Note this only works if the observations all have the
# same shape
LearnBase.getobs!(buf, ds, idx) = ...


# Now you can use your custom container

dataset = ImageDataset(["image1.jpg", "image2.jpg", ...])

# To iterate over observations
dataloader = DataLoader(dataset)

for obs in dataloader
    # do your stuff
end

# To iterate over batches
batchloader = BatchLoader(dataset, 16)

for obs in batchloader
    # do your stuff
end
```

Note: To use multiple workers (default behavior), you have to set the `JULIA_NUM_THREADS` environment variable before starting your session.