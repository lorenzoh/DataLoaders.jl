# Installation

### Julia

DataLoaders.jl is a package for the [Julia Programming Language](https://julialang.org/). To use the package you need to install Julia, which you can download [here](https://julialang.org/downloads/).

### DataLoaders.jl

Julia has a built-in package manager which is used to install packages. Running the installed `julia` command launches an interactive session. To install DataLoaders.jl, run the following command:

```julia-repl
using Pkg; Pkg.add("DataLoaders")
```

### Enabling multi-threading

To make use of multi-threaded data loading, you need to start Julia with multiple threads. If starting the `julia` executable yourself, you can pass a `-t <nthreads>` argument or set the environment variable `JULIA_NUM_THREADS` beforehand. To check that you have multiple threads available to you, run:

```julia-repl
julia> Threads.nthreads()
12
```

If you're running Julia in a Jupyter notebook, see [IJulia.jl's documentation](https://julialang.github.io/IJulia.jl/dev/manual/installation/#Installing-additional-Julia-kernels).