function threaded_mean(arr::Vector{T}) where T
    num_threads = Threads.nthreads()
    sums = Vector{T}(undef, num_threads)

    @threads for i in 1:num_threads
        start_index = div(length(arr)*(i - 1), num_threads) + 1
        end_index = div(length(arr)*i, num_threads)
        sums[i] = sum(arr[start_index:end_index])
    end

    # Compute the overall mean
    return sum(sums) ./ length(arr)
end

function convert_to_matrix_without_mean(m::Vector{Vector{T}}) where T <: Number
    mean_ = threaded_mean(m)

    l2 = length(m[1])
    M = Matrix{T}(undef, length(m), l2)
    
    Threads.@threads for i in 1:length(m)
        #@assert length(m[i]) == l2
        M[i, :] .= m[i] .- mean_
    end

    return M, mean_
end