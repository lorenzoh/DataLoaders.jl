# Comparison to PyTorch

This package is inspired by Pytorch's [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) and works a lot like it. The basic usage for both is `DataLoader(dataset, batchsize)`, but for other use cases there are some differences.

The most important things are:

- DataLoaders.jl supports only map-style datasets at the moment
- It uses thread-based parallelism instead of process-based parallelism

## Detailed comparison

Let's go through every argument to `torch.utils.data.DataLoader` and have a look at similarities and differences. See [`DataLoader`](#) for a full list of its arguments.

- `dataset`: This package currently only supports map-style datasets which work similarly to Python's, but instead of implementing `__getindex__` and `__len__`, you'd implement [`LearnBase.getobs`](#) and [`nobs`](#). [More info here](datacontainers.md).
- `batch_size = 1`: If not specified otherwise, the default batch size is 1 for both packages. In DataLoaders.jl, you can additionally pass in `nothing` to turn off batching.
- `shuffle = false`: This package's `DataLoader` does **not** support this argument. Shuffling should be applied to the dataset beforehand. See [working with data containers](howto/workingwith.md).
- `collate_fn`: DataLoaders.jl collates batches by default unless `collate = false` is passed. A custom collate function is not supported, but you can extend [`DataLoaders.collate`](#) for custom data types for the same effect.
- `drop_last = False`. DataLoaders.jl has the same behavior of returning a partial batch by default, but the keyword argument is `partial = false` with `partial = not drop_last`.
- `prefetch_factor`: This cannot be customized currently. The default behavior for DataLoaders.jl is for every thread to be preloading one batch.
- `pin_memory`: DataLoaders.jl does not interact with the GPU, but you can do this in your data container.
- `num_workers`, `persistent_workers`, `worker_init_fn`, `timeout`: Unlike PyTorch, this package does not use multiprocessing, but multithreading which is not practical in Python due to the GIL. As such these arguments do not apply. Currently, DataLoaders.jl will use either all threads except the primary one or all threads based on the keyword argument `useprimary = false`.
- `sampler`, `batch_sampler`, `generator`: This package does not currently support these arguments for customizing the randomness.


