# TODO: abstract BatchDims to allow for different schemes


@with_kw struct BatchViewCollated{TData}
    data::TData
    size::Int
    droplast::Bool
    count::Int
end

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
    batchdim = length(size(buf))
    @assert length(indices) == size(buf, batchdim)
    for (idx, A) in zip(indices, eachslice(buf; dims = batchdim))
        getobs!(A, bv.data, idx)
    end

    return buf
end
