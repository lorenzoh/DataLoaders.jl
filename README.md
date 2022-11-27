# DataLoaders.jl


**This package is deprecated and its functionality has been moved into [MLUtils.jl](https://github.com/JuliaML/MLUtils.jl) as [`MLUtils.DataLoader`](https://juliaml.github.io/MLUtils.jl/stable/api/#MLUtils.DataLoader). This package is still functional but development continues in MLUtils.jl. There are slight differences in the API; see the documentation linked above for more details.**

[Documentation (latest)](https://lorenzoh.github.io/DataLoaders.jl/dev)

A Julia package implementing performant data loading for deep learning on out-of-memory datasets that. Works like PyTorch's `DataLoader`.

### What does it do?

- Uses multi-threading to load data in parallel while keeping the primary thread free for the training loop
- Handles batching and [collating](docs/collate.md)
- Is simple to [extend](docs/interface.md) for custom datasets
- Integrates well with other packages in the [ecosystem](docs/ecosystem.md)
- Allows for [inplace loading](docs/inplaceloading.md) to reduce memory load

### When should you use it?

- You have a dataset that does not fit into memory
- You want to reduce the time your training loop is waiting for the next batch of data

### How do you use it?

Install like any other Julia package using the package manager (see [setup](docs/setup.md)):

```julia-repl
]add DataLoaders
```

After installation, import it, create a `DataLoader` from a dataset and batch size, and iterate over it:

```julia
using DataLoaders
# 10.000 observations of inputs with 128 features and one target feature
data = (rand(128, 10000), rand(1, 10000))
dataloader = DataLoader(data, 16)

for (xs, ys) in dataloader
    @assert size(xs) == (128, 16)
    @assert size(ys) == (1, 16)
end
```

### Next, you may want to read

- [What datasets you can use it with](docs/datacontainers.md)
- [How it compares to PyTorch's data loader](docs/quickstartpytorch.md)
