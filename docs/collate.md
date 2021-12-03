# Collating

Collating refers to combining a batch of observations so that the arrays in individual observations are stacked together. As an example, consider a dataset with 100 observations, each with 2 features.

{cell=main result=false}
```julia
data = rand(2, 100)  # observation dimension is last by default
```

We can collect a batch of 4 observations into a vector as follows:
{cell=main}
```julia
using DataLoaders
batch = [getobs(data, i) for i in 1:4]
```

Many machine learning models, however, expect an input in a _collated_ format: instead of a nested vector of vectors, we need a single ND-array. DataLoaders.jl provides the [`collate`](#) function for this:

{cell=main}
```julia
DataLoaders.collate(batch)
```

As you can see, the batch dimension is the last one by default.


## Nested observations

The above case only shows how to collate observations that each consist of a single array. In practice, however, observations will often consist of multiple variables like input features and a target variable. For example, we could have an integer indicating the class of an input sample.

{cell=main}
```julia
inputs = rand(2, 100)
targets = rand(1:10, 100)
data = (inputs, targets)
batch = [getobs(data, i) for i in 1:4]
```

Collating here also works, by keeping the tuple structure and collating each element separately:


{cell=main}
```julia
DataLoaders.collate(batch)
```

This is also implemented for `NamedTuple`s and `Dict`s. You can also collate nested structures, e.g. a `Tuple` of `Dict`s and the structure is preserved. This also works when using [inplace loading](inplaceloading.md).