## Data containers

`DataLoaders` supports some data containers out of the box, like arrays
and tuples of arrays. For large datasets that don't fit into memory, however,
we need some custom logic that loads and preprocesses our data.

We can make any data container compatible with `DataLoaders` by implementing
the two methods `nobs` and `getobs` from the interface package [`LearnBase`](https://github.com/JuliaML/LearnBase.jl).

`nobs(ds::MyDataset)` returns the number of observations in your data container
and `getobs(ds::MyDataset, idx)` loads a single observation.

For performance reasons, you may want to implement `getobs!(buf, ds::MyDataset, idx)`, a buffered version.

### At last, a realistic example

Image datasets are a good use case for `DataLoaders` because

- 20GB (or more) of images will likely not fit into your memory, so we need an
  out-of-memory solution; and
- decoding images is CPU-bottlenecked (provided your secondary storage can keep up),
  so we benefit from using multiple threads.

A simple data container might simply be a struct that contains the paths to
a lot of image files, like so:

```julia
struct ImageDataset
    files::Vector{String}
end
```

Since we're only storing the file paths and not the actual images, `ImageDataset`
barely takes up memory.

Implementing the data container interface is straightforward:

```julia
import LearnBase: nobs, getobs
using Images: load

nobs(ds::ImageDataset) = length(ds.files)
getobs(ds::ImageDataset, idx::Int) = load(ds.files[idx])
```

And now we can use it with `DataLoaders`:

```julia
data = ImageDataset(readdir("./IMAGENET_IMAGES"))

dataloader = DataLoader(data, 32; collate = false)

for images in dataloader
    # do your thing
end
```

!!! warning "Threads"

    To use `DataLoaders`' multi-threading, you need to start Julia with multiple
    threads. Check the number of available threads with `Threads.nthreads()`.
