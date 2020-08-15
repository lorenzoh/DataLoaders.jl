# TODO: abstract BatchDims to allow for different schemes
# TODO: handle `droplast` == false


@with_kw struct BatchViewCollated{TData}
    data::TData
    size::Int
    droplast::Bool
    count::Int
end

"""
    BatchViewCollated(data, size; droplast = true)

A batch view of container `data` with collated batches of
size `size`.
"""
function BatchViewCollated(data, size; droplast = true)
    obs = getobs(data, 1)
    count = nobs(data) ÷ size + (1 - Int(droplast))
    return BatchViewCollated(data, size, droplast, count)
end

Base.length(A::BatchViewCollated) = A.count
LearnBase.nobs(bv::BatchViewCollated) = length(bv)

LearnBase.getobs(bv::BatchViewCollated, batchindex::Int) =
    collate(getobs(bv.data, MLDataPattern._batchrange(bv.size, batchindex)))

function LearnBase.getobs!(buf, bv::BatchViewCollated, batchindex::Int)
    indices = MLDataPattern._batchrange(bv.size, batchindex)
    for (idx, obs) in zip(indices, obsslices(buf))
        getobs!(obs, bv.data, idx)
    end
    return buf
end


"""
    obsslices(batch)

Iterate over views of all observations in a `batch`.
`batch` can be a batched array, a tuple of batches, or a
dict of batches.

```julia
batch = rand(10, 10, 4)  # batch size is 4
iter = obsslices(batch)
size(first(iter)) == (10, 10)
```
"""
obsslices(batch) = (obsslice(batch, i) for i in 1:_batchsize(batch))

function obsslice(batch::AbstractArray{T, N}, i) where {T, N}
    return view(batch, [(:) for _ in 1:N-1]..., i)
end

function obsslice(batch::Tuple, i)
    return Tuple(obsslice(batch[j], i) for j in 1:length(batch))
end

function obsslice(batch::Dict, i)
    return Dict(k => obsslice(v, i) for (k, v) in batch)
end


_batchsize(batch::Tuple) = _batchsize(batch[1])
_batchsize(batch::Dict) = _batchsize(batch[first(keys(batch))])
_batchsize(batch::AbstractArray{T, N}) where {T, N} = size(batch, N)

x, y = zeros(10, 10, 4), zeros(5, 4)
