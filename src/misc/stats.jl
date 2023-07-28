function wmean(arr::AbstractArray{<:Number}; weights=nothing, kwargs...)
    if weights === nothing
        arr = arr .* weights
    end
    return mean(arr; kwargs...)
end
function get_sum_size(arr::AbstractArray{<:Number}; dims=nothing, kwargs...)
    if dims === nothing
        return prod(size(arr))
    else
        if dims isa Int
            dims = [dims]
        end
        return prod(size(arr)[i] for i in dims)
    end
end

function wmean_and_var(arr::AbstractArray{<:Number}; weights=nothing, kwargs...)
    local mean_, var_
    size_ = get_sum_size(arr; kwargs...)
    if weights !== nothing
        @assert prod(size(weights)) == size_ "The size of the weights array must be the same as the dimensions of the array to be averaged."
        n = sum(weights; kwargs...)
        mean_ = sum(arr .* weights; kwargs...) ./ n
        f = size_ / (size_ - 1) ./ n
        arr_m_ = arr .- mean_
        var_ = sum(arr_m_ .* conj(arr_m_).* weights; kwargs...)  .* f
    else
        mean_ = mean(arr; kwargs...)
        f = size_ / (size_ - 1)
        arr_m_ = arr .- mean_
        var_ = mean(arr_m_ .* conj(arr_m_); kwargs...) .* f
    end
    return mean_, var_
end

wvar(arr::AbstractArray{<:Number}; kwargs...) = wmean_and_var(arr; kwargs...)[2]
wstd(arr::AbstractArray{<:Number}; kwargs...) = sqrt.(wvar(arr; kwargs...))

function threaded_mean(arr::Vector{T}; weights=nothing) where T
    @assert length(weights) == length(arr)
    num_threads = Threads.nthreads()
    sums = Vector{T}(undef, num_threads)
    if weights === nothing
        @threads for i in 1:num_threads
            start_index = div(length(arr)*(i - 1), num_threads) + 1
            end_index = div(length(arr)*i, num_threads)
            sums[i] = sum(arr[start_index:end_index])
        end
    else
        @threads for i in 1:num_threads
            start_index = div(length(arr)*(i - 1), num_threads) + 1
            end_index = div(length(arr)*i, num_threads)
            sums[i] = sum(arr[start_index:end_index].*weights[start_index:end_index])
        end
    end

    # Compute the overall mean
    return sum(sums) ./ length(arr)
end

function convert_to_matrix_without_mean(m::Vector{Vector{T}}; kwargs...) where T <: Number
    mean_ = threaded_mean(m)

    l2 = length(m[1])
    M = Matrix{T}(undef, length(m), l2)
    
    Threads.@threads for i in 1:length(m)
        #@assert length(m[i]) == l2
        M[i, :] .= m[i] .- mean_
    end

    return M, mean_
end