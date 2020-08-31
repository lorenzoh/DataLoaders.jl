
"""
    RingBuffer(size, buf)

A `Channel`-like data structure that rotates through
`size` buffers. `put!`s work by mutating one of the buffers:

```
put!(ringbuffer) do buf
    rand!(buf)
end
```

The result can then be `take!`n:

```
res = take!(ringbuffer)
```

!!! warning "Invalidation"

    Only one result is valid at a time! On the next `take!`, the previous
    result will be reused as a buffer and be mutated by a `put!`

See also [`put!`](#)
"""
mutable struct RingBuffer{T}
    buffers::Channel{T}
    results::Channel{T}
    current::T
end

function RingBuffer(bufs::Vector{T}) where T
    size = length(bufs) - 1
    buffers = Channel{T}(size + 1)
    results = Channel{T}(size)
    foreach(bufs[2:end]) do buf
        put!(buffers, buf)
    end

    return RingBuffer{T}(buffers, results, bufs[1])
end

function RingBuffer(size, buffer::T) where T
    buffers = [buffer]
    for i ∈ 1:size
        push!(buffers, deepcopy(buffer))
    end
    return RingBuffer(buffers)
end


function Base.take!(ringbuffer::RingBuffer)
    put!(ringbuffer.buffers, ringbuffer.current)
    take!(ringbuffer.results)
end


"""
    put!(f!, ringbuffer::RingBuffer)

Apply f! to a buffer in `ringbuffer` and put into the results
channel.

```julia
x = rand(10, 10)
ringbuffer = RingBuffer(1, x)
put!(ringbuffer) do buf
    @test x == buf
    copy!(buf, rand(10, 10))
end
x_ = take!(ringbuffer)
@test !(x ≈ x_)

```
"""
function Base.put!(f!, ringbuffer::RingBuffer)
    buf = take!(ringbuffer.buffers)
    buf_ = f!(buf)
    @assert buf_ === buf
    put!(ringbuffer.results, buf_)
end


function Base.close(ringbuffer::RingBuffer)
    close(ringbuffer.results)
    close(ringbuffer.buffers)
end
