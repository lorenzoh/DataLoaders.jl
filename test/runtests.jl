using Test
using TestSetExtensions
using DataLoaders
using DataLoaders:
    BatchViewCollated, BatchDimLast, BatchDimFirst, WorkerPool, RingBuffer, collate, obsslices
using MLDataPattern
using LearnBase
using Random


struct MockDataset
    n::Int
    sz::Any
    inplace::Bool
end


LearnBase.getobs(ds::MockDataset, idx::Int) = (randn(ds.sz...))
LearnBase.getobs(ds::MockDataset, idxs) = [getobs(ds, idx) for idx in idxs]
LearnBase.getobs!(buf, ds::MockDataset, idx::Int) =
    ds.inplace ? fill!(buf, 0.3) : getobs(ds, idx)
#LearnBase.getobs!(bufs, ds::MockDataset, idxs) = map(idx -> LearnBase.getobs!(buf, ds, idx), idxs)
LearnBase.nobs(ds::MockDataset) = ds.n


@testset ExtendedTestSet "collate" begin
    @test collate([([1, 2], 3), ([4, 5], 6)]) == ([1 4; 2 5], [3, 6])
    @test collate([(x = [1, 2], y = 3), (x = [4, 5], y = 6)]) ==
          (x = [1 4; 2 5], y = [3, 6])
    @test collate([Dict("x" => [1, 2], "y" => 3), Dict("x" => [4, 5], "y" => 6)]) ==
          Dict("x" => [1 4; 2 5], "y" => [3, 6])
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

@testset ExtendedTestSet "RingBuffer" begin
    @testset ExtendedTestSet "fills correctly" begin
        x = rand(10, 10)
        ringbuffer = RingBuffer(10, x)
        @async begin
            for i ∈ 1:100
                put!(ringbuffer) do buf
                    rand!(buf)
                end
            end
        end
        @test_nowarn for _ ∈ 1:100
            take!(ringbuffer)
        end
    end

    @testset ExtendedTestSet "does mutate" begin
        x = rand(10, 10)
        ringbuffer = RingBuffer(1, copy(x))
        put!(ringbuffer) do buf
            @test x ≈ buf
            copy!(buf, rand(10, 10))
            buf
        end
        x_ = take!(ringbuffer)
        @test !(x ≈ x_)
    end
end

@testset ExtendedTestSet "WorkerPool" begin
    xs = falses(10)

    @test !any(xs)
    pool = WorkerPool(collect(zip(1:10))) do i
        xs[i] = true
    end
    DataLoaders.run(pool)
    @test all(xs)


end


@testset ExtendedTestSet "GetObsAsync" begin
    make() = DataLoaders.GetObsAsync(rand(10, 64))

    @testset ExtendedTestSet "iterate" begin
        dl = make()
        x, (ringbuffer, workerpool, idx) = iterate(dl)
        @test idx == 1
        @test x == getobs(dl.data, 1)
        @test_nowarn for obs in dl
        end
    end
end


@testset ExtendedTestSet "BufferGetObsAsync" begin
    make() = DataLoaders.BufferGetObsAsync(rand(10, 64))

    @testset ExtendedTestSet "iterate" begin
        dl = make()
        x, (ringbuffer, workerpool, idx) = iterate(dl)
        @test idx == 1
        @test x == getobs(dl.data, 1)
        @test_nowarn for obs in dl
        end
    end
end


@testset ExtendedTestSet "batchviewcollated" begin
    @testset ExtendedTestSet "basic" begin
        data = rand(10, 64)
        bv = DataLoaders.batchviewcollated(data, 4)
        @test nobs(bv) == 16
        @test_nowarn for _ in obsview(bv) end
    end

    @testset ExtendedTestSet "partial" begin
        data = rand(10, 10)
        bv = DataLoaders.batchviewcollated(data, 3)
        @test nobs(bv) == 4
        buf = getobs(bv, 4)
        buf = getobs!(buf, bv, 4)

    end

end
