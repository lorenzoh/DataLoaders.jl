# Data container interface

{class="subtitle"}
Reference for implementing the data container interface. See [data containers](datacontainers.md) for an introduction.

To implement the data container interface for a custom type `T`, you must implement two functions:

- `LearnBase.getobs(data::T, i::Int)` loads the `i`-th observation
- `LearnBase.nobs(data::T, i::Int)::Int` gives the number of observations in a data container

You can _optionally_ also implement:

- `LearnBase.getobs!(buf, data::T, i::Int)`: loads the `i`-th observation into the preallocated buffer `buf`.


See [the MLDataPattern.jl documentation](https://mldatapatternjl.readthedocs.io/en/latest/documentation/container.html) for a comprehensive discussion of and reference for data containers.

!!! note "Extending functions"

    To define a method for the above functions, you need to import the functions explicitly. You can do this without installing `LearnBase` by running:

    ```julia
    import DataLoaders.LearnBase: getobs, nobs, getobs!
    ````
