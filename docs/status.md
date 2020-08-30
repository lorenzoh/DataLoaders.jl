# TODO before v0.1.0

- unify interface

```julia
# Default options
DataLoader(
    data, batchsize = 1;
    numworkers = Threads.nthreads() - 1,
    partial = false,
    collate = true,
    buffered = collate,
) = AsyncIterBuffered(batchviewcollated(data, batchsize), numworkers)

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
  - `batchview(partial = false)`
  - `batchviewcollated(partial = false)`
  - `AsyncIter(data, numworkers)`
  - `AsyncIterBuffered(data, numworkers)`

## Requirements

- `droplast` works with `BatchViewBuffered`
- works when no threads available; print error message
- `shuffle` and `datasubset` work inplace
- errors on workers or main thread lead to interrupt of both
- compatible with `ObsDim`s

## ToDo

- fix collation for data containers that don't support `getobs!`
