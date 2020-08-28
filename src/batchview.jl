# TODO: abstract BatchDims to allow for different schemes
# TODO: handle `droplast` == false


abstract type BatchDim end

struct BatchDimFirst <: BatchDim end
struct BatchDimLast <: BatchDim end


@with_kw struct BatchViewCollated{TData}
    data::TData
    size::Int
    count::Int
    droplast::Bool
    batchdim::BatchDim = BatchDimLast()
end

"""
    BatchViewCollated(data, size; droplast = true)

A batch view of container `data` with collated batches of
size `size`.
"""
function BatchViewCollated(data, size; droplast = true, batchdim = BatchDimLast())
    obs = getobs(data, 1)
    count = nobs(data) ÷ size + (1 - Int(droplast))
    return BatchViewCollated(data, size, count, droplast, batchdim)
end

Base.length(A::BatchViewCollated) = A.count
LearnBase.nobs(bv::BatchViewCollated) = length(bv)

function LearnBase.getobs(bv::BatchViewCollated, batchindex::Int)
    idxs = MLDataPattern._batchrange(bv.size, batchindex)
    collate([getobs(bv.data, idx) for idx in idxs])
end

function LearnBase.getobs!(buf, bv::BatchViewCollated, batchindex::Int)
    indices = MLDataPattern._batchrange(bv.size, batchindex)
    # TODO: Fix for partial batches
    for (idx, obs) in zip(indices, obsslices(buf, bv.batchdim))
        getobs!(obs, bv.data, idx)
    end
    return buf
end


"""
    obsslices(batch, batchdim = BatchDimLast())

Iterate over views of all observations in a `batch`.
`batch` can be a batched array, a tuple of batches, or a
dict of batches.

```julia
batch = rand(10, 10, 4)  # batch size is 4
iter = obsslices(batch)
size(first(iter)) == (10, 10)
```
"""
obsslices(batch, batchdim = BatchDimLast()) =
    (obsslice(batch, i, batchdim) for i in 1:_batchsize(batch))

function obsslice(batch::AbstractArray{T, N}, i, ::BatchDimLast) where {T, N}
    return view(batch, [(:) for _ in 1:N-1]..., i)
end

function obsslice(batch::AbstractArray{T, N}, i, ::BatchDimFirst) where {T, N}
    return view(batch, i, [(:) for _ in 2:N]...)
end

function obsslice(batch::Tuple, i, batchdim)
    return Tuple(obsslice(batch[j], i, batchdim) for j in 1:length(batch))
end

function obsslice(batch::Dict, i, batchdim)
    return Dict(k => obsslice(v, i, batchdim) for (k, v) in batch)
end


_batchsize(batch::Tuple) = _batchsize(batch[1])
_batchsize(batch::Dict) = _batchsize(batch[first(keys(batch))])
_batchsize(batch::AbstractArray{T, N}) where {T, N} = size(batch, N)
