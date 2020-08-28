# Quickstart for PyTorch users

Like Pytorch's [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader),
this package provides an iterator over your dataset that loads data in parallel in the
background.

The basic interface is the same. Create a DataLoader with `DataLoader(data, batchsize)`.

## Commonalities

- **Dataset interface**: Instead of implementing `mydataset.__getindex__(idx)`
  and `mydataset.__len__()` you implement `getobs(mydataset, idx)` and `nobs(mydataset)`.
  Same, same, different name. See [Data containers](docs/datacontainers.md) for specifics.

- **Collating**: `DataLoader` has the `collate = true` keyword argument.