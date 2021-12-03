# Inplace loading

{.subtitle}
Background on inplace loading of data


When loading an observation of a [data container](datacontainers.md) requires allocating a lot of memory, it is sometimes possible to reuse a previous observation as a buffer to load into. To do so, the data container you're using, must [implement `getobs!`](interface.md). To use the buffered loading with this package, pass `buffered = true` to [`DataLoader`](#). This also works for collated batches.

