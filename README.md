# DataLoaders.jl

A parallel iterator for large machine learning datasets that don't fit into memory inspired by PyTorch's `DataLoader` class.

Uses `MLDataUtils.jl`'s [Data Access Pattern](https://mldatautilsjl.readthedocs.io/en/latest/data/pattern.html), so many data containers work out of the box and custom containers are easily supported by implementing `LearnBase.getobs` and `LearnBase.nobs`.

## Usage

```julia
dataset = ([1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12])

```

