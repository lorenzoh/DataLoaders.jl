
# Shuffling, subsetting, sampling

Shuffling your training data every epoch and splitting a dataset into training and validation splits are common practices.
While `DataLoaders` itself only provides tools to load your data effectively, using the underlying `MLDataPattern` package makes these things easy.

## Examples

### Shuffling

```julia
using MLDataPattern: shuffleobs

data = ...
dataloader = DataLoader(shuffleobs(data), batchsize)
```

### Subsetting

```julia
using MLDataPattern: datasubset

data = ...
idxs = 1:1000  # indices to select from dataset
  
dataloader = DataLoader(datasubset(data, idxs)), batchsize)
```

### Train/validation split

```julia
using MLDataPattern: splitobs
    
data = ...
traindata, valdata = splitobs(data, 0.8)  # 80/20 split
dataloader = DataLoader(shuffleobs(data), batchsize)
```

## But wait, there's more

For other dataset operations like weighted sampling, see [this section](https://mldatapatternjl.readthedocs.io/en/latest/documentation/datasubset.html) in MLDataPattern's documentation.
