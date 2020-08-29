# Quickstart for PyTorch users

Like Pytorch's [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader),
this package provides an iterator over your dataset that loads data in parallel in the
background.

The basic interface is the same: `DataLoader(data, batchsize)`.

See [`DataLoader`](#) for all supported options.

## PyTorch vs. `DataLoaders.jl`

### Dataset interface 

The dataset interface for map-style datasets is similar:

PyTorch:

- `mydataset.__getindex__(idx)`
- `mydataset.__len__()`

DataLoaders.jl:

- `LearnBase.getobs(mydataset, idx)`
- `LearnBase.nobs(mydataset)`

See [Data containers](datacontainers.md) for specifics.

### Sampling and shuffling

Unlike PyTorch's DataLoader, `DataLoaders.jl` delegates [shuffling, subsetting and sampling](shuffling.md) to existing packages. Consequently there are no `shuffle`, `sampler` and `batch_sampler` arguments.

### Automatic batching

Automatic batching is controlled with the `collate` keyword argument (default `true`). A custom `collate_fn` is not supported.
