# DataLoaders

[Documentation (latest)](https://lorenzoh.github.io/DataLoaders.jl/dev)

A threaded data iterator for machine learning on out-of-memory datasets. Inspired by PyTorch's DataLoader.

It uses  to load data **in parallel** while keeping the primary thread free. It can also load data **inplace** to avoid allocations.

Many data containers work out of the box and it is easy to [extend with your own](docs/datacontainers.md).

`DataLoaders` is built on top of and fully compatible with `MLDataPattern.jl`'s [Data Access Pattern](https://mldatautilsjl.readthedocs.io/en/latest/data/pattern.html), a functional interface for machine learning datasets.

## Usage

```julia
x = rand(128, 10000)  #  10000 observations of size 128
y = rand(1, 10000)

dataloader = DataLoader((x, y), 16)

for (xs, ys) in dataloader
    @assert size(xs) == (128, 16)
    @assert size(ys) == (1, 16)
end
```

Of course, in the above example, we can keep the dataset in memory and don't need parallel workers. See [Custom data containers](docs/datacontainers.md) for a more realistic example.

## Getting Started

If you get the idea and know it from PyTorch, see [Quickstart for PyTorch users](docs/quickstartpytorch.md).

Otherwise, read on [here](docs/motivation.md).

Available methods are documented [here](docstrings.md).

## Acknowledgements

- [`MLDataPattern.jl`](https://github.com/JuliaML/MLDataPattern.jl)
- [`ThreadPools.jl`](https://github.com/tro3/ThreadPools.jl)
- [PyTorch `DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)