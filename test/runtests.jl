using Test
using TestSetExtensions
using DataLoaders
using DataLoaders: BatchViewCollated, BatchDimLast, BatchDimFirst, collate, obsslices
using MLDataPattern
using LearnBase


struct MockDataset
    n::Int
    sz
    inplace::Bool
end


LearnBase.getobs(ds::MockDataset, idx::Int) = (randn(ds.sz...))
LearnBase.getobs(ds::MockDataset, idxs) = [getobs(ds, idx) for idx in idxs]
LearnBase.getobs!(buf, ds::MockDataset, idx::Int) = ds.inplace ? fill!(buf, 0.3) : getobs(ds, idx)
#LearnBase.getobs!(bufs, ds::MockDataset, idxs) = map(idx -> LearnBase.getobs!(buf, ds, idx), idxs)
LearnBase.nobs(ds::MockDataset) = ds.n


@testset ExtendedTestSet "collate" begin
    @test collate([([1, 2], 3), ([4, 5], 6)]) == ([1 4; 2 5], [3, 6])
    @test collate([(x = [1, 2], y = 3), (x = [4, 5], y = 6)]) == (x = [1 4; 2 5], y = [3, 6])
    @test collate([Dict("x" => [1, 2], "y" => 3), Dict("x" => [4, 5], "y" => 6)]) == Dict("x" => [1 4; 2 5], "y" => [3, 6])
    @test collate([(1, 2), (3, 4)]) == ([1, 3], [2, 4])
end


@testset ExtendedTestSet "obsslices" begin
    batch = rand(10, 10, 16)
    @test_nowarn for obs in obsslices(batch, BatchDimLast())
        @assert size(obs) == (10, 10)
    end

    @test_nowarn for obs in obsslices(batch, BatchDimFirst())
        @assert size(obs) == (10, 16)
    end

    batch2 = Dict(:x => rand(5, 16), :y => rand(1, 16))
    obs = first(obsslices(batch2))
    @test obs isa Dict
    @test size(obs[:x]) == (5,)
    @test size(obs[:y]) == (1,)
end


@testset ExtendedTestSet "BatchViewCollated" begin
    data = rand(2, 100)
    @testset ExtendedTestSet "basic" begin
        bv = DataLoaders.BatchViewCollated(data, 5)
        @test nobs(bv) == 20
        @test_nowarn for i = 1:nobs(bv)
            getobs(bv, i)
        end
    end

    @testset ExtendedTestSet "getobs!" begin
        bv = DataLoaders.BatchViewCollated(data, 5)
        buf = getobs(bv, 1)
        getobs!(buf, bv, 2)
        @test buf == getobs(bv, 2)
    end

    @testset ExtendedTestSet "droplast" begin
        bv = BatchViewCollated(data, 3, droplast = false)
        @test nobs(bv) == 34

    end

    @testset ExtendedTestSet "inplace loading for non-supporting datasets" begin
        ds = MockDataset(64, (10,), false)
        bv = BatchViewCollated(ds, 4)
        buf = getobs(bv, 1)
        x = buf[1]
        getobs!(buf, bv, 2)
        @test x != buf[1]
    end
end
#=
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

=#
