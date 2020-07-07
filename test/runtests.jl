using Test
using TestSetExtensions
using DataLoaders
using DataLoaders: getbatchindices
import LearnBase: nobs, getobs

struct TestDataset
    n
    sz
end
nobs(ds::TestDataset) = ds.n
getobs(ds::TestDataset, ::Int) = (rand(Float32, ds.sz[1], ds.sz[2], 3), rand(Float32, 10))

@testset ExtendedTestSet "" begin
    ds = TestDataset(1024, (128, 128))

    @testset ExtendedTestSet "Single-Threaded" begin
        dl = DataLoader(ds, numworkers = 1)
        @test_nowarn for batch in dl end
    end

    @testset ExtendedTestSet "Multi-Threaded" begin
        if Threads.nthreads() == 1
            @warn "Can't run multi-threaded tests for `DataLoader`"
            numworkers = 1
        else
            numworkers = Threads.nthreads() - 1
        end
        dl = DataLoader(ds, numworkers = numworkers)
        @test_nowarn for batch in dl end
    end

end


@testset ExtendedTestSet "DataLoaders.jl tests" begin
    @testset ExtendedTestSet "collate" begin

        @test collate([([1, 2], 3), ([4, 5], 6)]) == ([1 4; 2 5], [3, 6])
        @test collate([(x = [1, 2], y = 3), (x = [4, 5], y = 6)]) == (x = [1 4; 2 5], y = [3, 6])
        @test collate([Dict("x" => [1, 2], "y" => 3), Dict("x" => [4, 5], "y" => 6)]) == Dict("x" => [1 4; 2 5], "y" => [3, 6])

        @test collate([(1, 2), (3, 4)]) == ([1, 3], [2, 4])
    end

    @testset ExtendedTestSet "getbatchindices" begin
        @test getbatchindices(6, 2, false) == [[1, 2], [3, 4], [5, 6]]
        @test getbatchindices(5, 2, false, false) == [[1, 2], [3, 4], [5]]
        @test getbatchindices(5, 2, false, true) == [[1, 2], [3, 4]]
        @test length(getbatchindices(100, 10, true)) == 10
    end
    @testset ExtendedTestSet "DataLoader length" begin
        dl = DataLoader(collect(1:127), batchsize = 8, collatefn = identity, droplast = false)
        @test length(collect(dl)) == length(dl) == 128 ÷ 8

        dl = DataLoader(collect(1:127), batchsize = 8, collatefn = identity, droplast = true)
        @test length(collect(dl)) == length(dl) == (128 ÷ 8) - 1
    end

    @testset ExtendedTestSet "DataLoader data" begin
        dl = DataLoader(collect(1:8), batchsize = 2, collatefn = identity, shuffle = false, numworkers = 1)
        @test collect(dl) == [[1, 2], [3, 4], [5, 6], [7, 8]]
    end
end

DataLoader(ones(100))

dl = DataLoader(collect(1:1024), batchsize = 2, collatefn = identity, shuffle = false, numworkers = 1)
@time collect(dl);

dl = DataLoader(collect(1:1024), batchsize = 2, collatefn = identity, shuffle = false, numworkers = 11)
@time collect(dl);
