using Test
using TestSetExtensions
using DataLoaders
using MLDataPattern
using LearnBase


struct MockDataset
    n
    sz
end

LearnBase.getobs(ds::MockDataset, idx::Int) = (randn(ds.sz...))
LearnBase.getobs(ds::MockDataset, idxs) = [getobs(ds, idx) for idx in idxs]
LearnBase.getobs!(buf, ds::MockDataset, idx::Int) = fill!(buf, 0.3)
LearnBase.getobs!(bufs, ds::MockDataset, idxs) = map(idx -> LearnBase.getobs!(buf, ds, idx), idxs)
LearnBase.nobs(ds::MockDataset) = ds.n

ds = MockDataset(128, (16, 16))

bv = DataLoaders.BatchViewBuffered(ds, 4)

@time buf = getobs(shuffleobs(bv), 1)


buf[end] = 0
@time getobs!(buf, bv, 1)
