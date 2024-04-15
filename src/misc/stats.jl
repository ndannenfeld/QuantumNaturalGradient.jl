function wmean(arr::AbstractArray{<:Number}; weights=nothing, kwargs...)
    if weights !== nothing
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
        n = mean(weights; kwargs...)
        mean_ = mean(arr .* weights; kwargs...) ./ n
        f = size_ / (size_ - 1) ./ n
        arr_m_ = arr .- mean_
        var_ = mean(arr_m_ .* conj(arr_m_).* weights; kwargs...)  .* f
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
    num_threads = Threads.nthreads()
    sums = Vector{T}(undef, num_threads)
    local norm
    len = length(arr)
    if weights === nothing
        @threads for i in 1:num_threads
            start_index = div(len*(i - 1), num_threads) + 1
            end_index = div(len*i, num_threads)
            sums[i] = sum(arr[start_index:end_index])
        end
        norm = len
    else
        @assert length(weights) == len
        norms = Vector{T}(undef, num_threads)
        @threads for i in 1:num_threads
            start_index = div(len*(i - 1), num_threads) + 1
            end_index = div(len*i, num_threads)
            sums[i] = sum(arr[start_index:end_index].*weights[start_index:end_index])
            norms[i] = sum(weights[start_index:end_index])
        end
        norm = sum(norms)
    end

    # Compute the overall mean
    return sum(sums) ./ length(arr)
end

function convert_to_matrix_without_mean(m::Vector{Vector{T}}; mean_=nothing, kwargs...) where T <: Number
    if mean_ === nothing
        mean_ = threaded_mean(m)
    end

    l2 = length(m[1])
    M = Matrix{T}(undef, length(m), l2)
    
    Threads.@threads for i in 1:length(m)
        #@assert length(m[i]) == l2
        M[i, :] .= m[i] .- mean_
    end

    return M, mean_
end