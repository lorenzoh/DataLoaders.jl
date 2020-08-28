# ## A simple example
# Let's have another look at the example from the [introduction](README.md).
#
# First we create some dummy data of observations and targets:
x = rand(128, 4096)  # 4096 samples of size (128,)
y = rand(1, 4096)    # 4096 samples of size (1,)

# We then create a `DataLoader` with batch size 16:
dataloader = BatchLoader((x, y), 16)

# Et voilá, a training loop:
for (xs, ys) in dataloader  # size(xs) == (128, 16)
    ## optimization step
end


# `DataLoaders` supports many different data containers by building off the interface of
# [`MLDataPattern.jl`](https://mldatautilsjl.readthedocs.io/en/latest/data/pattern.html).
# Above, we pass in a tuple of datasets, hence the batch is also tuple.
#
# ---
#
# Let's [next](docs/datacontainers.md) look at a realistic use case
# and show how to support custom data containers.
