# Data containers

{.subtitle}
Introduction to data containers giving an overview over the kinds of datasets you can use

DataLoaders.jl is built to integrate with the further ecosystem and builds on a common interface to support datasets. We call such a dataset a **data container** and it needs to support the following operations:

- `getobs(data, i)` loads the `i`-th observation from a dataset
- `nobs(data)` gives the number of observations in a dataset.

## Basic data containers

The simplest data container is a vector of values:

{cell=main}
```julia
using DataLoaders
@show v = rand(1:10, 10)
@show nobs(v)
getobs(v, 1)
```

Multi-dimensional arrays also work, with the last dimension treated as the observation dimension:

{cell=main}
```julia
a = rand(50, 50, 10)
summary(getobs(a, 1))
```

You can also group multiple data containers with the same length together by putting them into a `Tuple`:

{cell=main}
```julia
data = (v, a)
getobs(data, 1)
```

You can pass any data container to [`DataLoader`](#) to create an iterator over batches:

```julia
for batch in DataLoader(v, 2)
    @assert size(batch) == (2,)
end

for batch in DataLoader(a, 2) 
    @assert size(batch) == (50, 50, 2)
end

for (vs, as) in DataLoader((v, a), 2) 
    @assert size(vs) == (2,)
    @assert size(as) == (50, 50, 2)
end

```

## Out-of-memory data containers

Arrays, of course, are kept in memory, so we (1) cannot use them to store larger-than-memory datasets (2) don't need to use multithreading since loading an observation just involves indexing an array which is generally fast.

One way to quickly get into the territory of too-large-to-fit in memory is to work with image datasets. So instead of loading every image of a dataset into an array, we'll implement a data container that stores only the file names of each image. It will load the image itself only when `getobs` is called. To do that we'll implement a `struct` that stores a vector of file names, and implement `getobs` and `nobs` for that type.

```julia
import DataLoaders.LearnBase: getobs, nobs
using Images

struct ImageDataset
    files::Vector{String}
end
ImageDataset(folder::String) = ImageDataset(readdir(folder))

nobs(data::ImageDataset) = length(data.files)
getobs(data::ImageDataset, i::Int) = Images.load(data.files[i])
```
Now, if we have a folder full of images, we can create a data container and load them quickly into batches as follows:

```julia
data = ImageDataset("path/to/my/images")
for images in DataLoader(data, 16, collate = false)
    # Do something
end
```

!!! note "Preprocessing"

    Above we pass the `collate = false` argument because images may be of different sizes that cannot be collated. See [`collate`](#). In practice, it is common to apply some cropping and resizing to images so that they all have the same size.


!!! warning "Threads"

    To use `DataLoaders`' multi-threading, you need to start Julia with multiple
    threads. Check the number of available threads with `Threads.nthreads()`.
