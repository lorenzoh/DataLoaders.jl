using Test
using TestSetExtensions
using DataLoaders
using DataLoaders: collate
using MLDataPattern
using LearnBase


@testset ExtendedTestSet "collate" begin
    @test collate([([1, 2], 3), ([4, 5], 6)]) == ([1 4; 2 5], [3, 6])
    @test collate([(x = [1, 2], y = 3), (x = [4, 5], y = 6)]) == (x = [1 4; 2 5], y = [3, 6])
    @test collate([Dict("x" => [1, 2], "y" => 3), Dict("x" => [4, 5], "y" => 6)]) == Dict("x" => [1 4; 2 5], "y" => [3, 6])
    @test collate([(1, 2), (3, 4)]) == ([1, 3], [2, 4])
end

@testset ExtendedTestSet "obsslices" begin

end

dataset = rand(128, 10000)  # 1024 observations of size 512
getobs(dataset, 1)

dl = BatchLoader(dataset, 8)
@time buf = getobs(dl.data, 1)
@time buf = getobs!(buf, dl.data, 1)

@time first(dl)
@time for obs in dl
end

@testset ExtendedTestSet "BatchLoader" begin
    dataset = rand(128, 10000)
    dataloader = BatchLoader(dataset, 16)
    @test_nowarn for batch in dataloader
        @assert size(batch) == (128, 16)
    end
end


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


@testset ExtendedTestSet "" begin
    @test_nowarn bv = DataLoaders.BatchView(ds, 4)
    bv = DataLoaders.BatchView(ds, 4)
    @test_nowarn buf = getobs(bv, 1)
end


x = rand(128, 10000)  #  10000 observations of size 128
y = rand(1, 10000)

dataloader = BatchLoader((x, y), 16)

getobs!(getobs(dataloader.data, 1), dataloader.data, 1)

for batch in dataloader
    @assert size(batch) == (128, 16)
end
