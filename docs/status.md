# TODO before v0.1.0

- unify interface


```julia
# Default options
DataLoader(
    data, batchsize = 1;
    numworkers = Threads.nthreads() - 1,
    partial = false,
    shuffle = false,
    collate = true,
    buffered = collate,
)

# Training example
DataLoader(data, 16, partial = true)
  -> BatchViewCollated

# Validation example
DataLoader(data, 16, shuffle = false, partial = false)

# Visualization example
DataLoader(data, nothing, shuffle = true, collate = false)
```

- implement unbuffered version
- document
- format structs repr
- make sure datasets wrapped in `shuffleobs` and `datasubset`
  support `getobs!`
- provide
  - `batchview(droplast = false)`
  - `batchviewcollated(droplast = false)`
  - `asyncloader()`
  - `bufferedasyncloader`

## Requirements

- `droplast` works with `BatchViewBuffered`
- works when no threads available; print error message
- `shuffle` and `datasubset` work inplace
- errors on workers or main thread lead to interrupt of both
- compatible with `ObsDim`s
