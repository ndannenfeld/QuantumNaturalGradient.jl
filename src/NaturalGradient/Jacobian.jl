struct Jacobian{T <: Number}
    data::AbstractArray{T, 2}
    data_mean::Vector{T}
    importance_weights::Union{Vector{<:Real}, Nothing}
    function Jacobian(m::AbstractMatrix{T}; importance_weights=nothing, mean_=nothing, inplace=true) where T <: Number
        if mean_ === nothing
            data_mean = wmean(m; weights_=importance_weights, dims=1)
        else
            data_mean = reshape(mean_, 1, :)
        end
        if inplace
            #m .-= data_mean
            for i in 1:size(m, 1)
                @views m[i, :] .-= data_mean[1, :]
            end
        else
            m = m .- data_mean
        end
        return Jacobian(m, data_mean[1, :]; importance_weights, inplace=true)
    end

    function Jacobian(m::Transpose{T}; importance_weights=nothing, mean_=nothing, inplace=true) where T <: Number
        mt = transpose(m)
        if mean_ === nothing
            data_mean = wmean(mt; weights_=importance_weights, dims=2)
        else
            data_mean = reshape(mean_, :, 1)
        end
        
        if inplace
            Threads.@threads for i in 1:size(mt, 2)
                @views mt[:, i] .-= data_mean
            end
        else
            mt = mt .- data_mean
        end
        
        return Jacobian(transpose(mt), data_mean[:, 1]; importance_weights, inplace=true)
    end
    
    function Jacobian(m::AbstractMatrix{T}, data_mean::Vector{T}; importance_weights=nothing, inplace=true) where T <: Number
        if importance_weights !== nothing
            if inplace
                Threads.@threads for j in 1:size(m, 1)
                    @views m[j, :] .*= sqrt(importance_weights[j])
                end
            else
                m = m .* sqrt.(importance_weights)
            end
        end
        return new{T}(m, data_mean, importance_weights)
    end

    function Jacobian(m::Transpose{T}, data_mean::Vector{T}; importance_weights=nothing, inplace=true) where T <: Number
        if importance_weights !== nothing
            w = reshape(importance_weights, 1, :)
            mt = transpose(m)
            if inplace
                Threads.@threads for j in 1:size(mt, 2)
                    @views mt[:, j] .*= sqrt(w[j])
                end
            else
                mt = mt .* sqrt.(w)
            end
            m = transpose(mt)
        end
        return new{T}(m, data_mean, importance_weights)
    end

    function Jacobian(m::Vector{Vector{T}}; importance_weights=nothing, mean_=nothing) where T <: Number
        m, data_mean = convert_to_matrix_without_mean(m; weights_=importance_weights, mean_)
        return Jacobian(m, data_mean; importance_weights)
    end
end

Base.size(J::Jacobian) = size(J.data)
Base.size(J::Jacobian, i) = size(J.data, i)
nr_parameters(J::Jacobian) = size(J.data, 2)
nr_samples(J::Jacobian) = size(J.data, 1)
Base.length(J::Jacobian) = size(J.data, 1)

function Base.show(io::IO, J::Jacobian)
    print(io, "Jacobian(Nₚ=$(nr_parameters(J)), Nₛ=$(nr_samples(J)))")
end

function get_importance_weights(J::Jacobian)
    if J.importance_weights === nothing
        return ones(nr_samples(J))
    else
        return J.importance_weights
    end
end

centered(J::Jacobian; kwargs...) = centered(J.data, J.importance_weights; kwargs...)

function centered(data::AbstractMatrix{T}, importance_weights; mode=:importance_sqrt) where T <: Number
    if importance_weights === nothing
        return data
    end
    if mode == :importance_sqrt
        return data
    elseif mode == :importance
        return data .* sqrt.(importance_weights)
    elseif mode == :no_importance
        return data ./ sqrt.(importance_weights)
    else
        error("mode should be :importance_sqrt, :importance or :no_importance. $mode was given.")
    end
end

function centered(data::Transpose{T}, importance_weights; mode=:importance_sqrt) where T <: Number
    if importance_weights === nothing
        return data
    end
    d = centered(transpose(data),  reshape(importance_weights, 1, :); mode)
    return transpose(d)
end

    
function uncentered(J::Jacobian)
    Jd = centered(J; mode=:no_importance)
    if J.data isa Transpose
        data_mean = reshape(J.data_mean, :, 1)
        return transpose(transpose(Jd) .+ data_mean)
    end
    data_mean = reshape(J.data_mean, 1, :)
    return Jd .+ data_mean 
end

Statistics.mean(J::Jacobian) = J.data_mean

"""
    dense_T(J::Jacobian)

Compute the `T` matrix = J*Jᵀ (size N×N if J is N×D). 
Dispatch to dense_T(::AbstractMatrix) or dense_T(::Transpose) 
depending on whether `J.data` is a plain matrix or a `Transpose` wrapper.
"""
dense_T(J::Jacobian) = dense_T(centered(J))  # `centered(J)` returns either a Matrix or a Transpose, 


"""
    dense_T(M::AbstractMatrix{T})

If `M` is an N×D matrix, returns `M * Mᵀ`, an N×N matrix.
J * J' is slow, so we use BLAS.gemm! instead.
"""
function dense_T(M::AbstractMatrix{T}) where T <: Number
    N = size(M, 1)
    C = Matrix{T}(undef, N, N)
    # J * J' is slow, so we use BLAS.gemm! instead
    BLAS.gemm!('N', 'C', T(1), M, M, T(0), C)
    return C
end

"""
    dense_T(M::Transpose{T})

If `M` is logically N×D but stored as a `Transpose` (physically D×N), 
still return an N×N matrix = M * Mᵀ.
J * J' is slow, so we use BLAS.gemm! instead.
"""
function dense_T(M::Transpose{T}) where T <: Number
    N = size(M, 1)
    C = Matrix{T}(undef, N, N)
    Mt = transpose(M)
    BLAS.gemm!('C', 'N', T(1), Mt, Mt, T(0), C)
    return conj(C)
end


"""
    dense_S(J::Jacobian)

Compute the `S` matrix = (1 / nr_samples(J)) * (Jᵀ * J) (size D×D if J is N×D).
Again dispatch to specialized matrix/transposed versions.
"""
dense_S(J::Jacobian) = dense_S(centered(J)) 

"""
    dense_S(M::AbstractMatrix{T})

If M is N×D, returns (1/N)*(Mᵀ * M), a D×D matrix.
J' * J is slow, so we use BLAS.gemm! instead.
"""
function dense_S(M::AbstractMatrix{T}) where T
    N, D = size(M)
    C = Matrix{T}(undef, D, D)
    # J' * J is slow, so we use BLAS.gemm! instead
    BLAS.gemm!('C', 'N', T(1. / N), M, M, T(0), C)
    return C
end

"""
    dense_S(M::Transpose{T})

If M is logically N×D but stored as a `Transpose` (physically D×N), 
we still want a D×D result. We do M * Mᵀ and divide by N.
J' * J is slow, so we use BLAS.gemm! instead.
"""
function dense_S(M::Transpose{T}) where T <: Number
    # If M is logically N×D, size(M,1) = N, size(M,2)=D
    # But physically it’s D×N. We want a D×D result => M * Mᵀ
    N, D = size(M)
    C = Matrix{T}(undef, D, D)
    Mt = transpose(M)
    # J' * J is slow, so we use BLAS.gemm! instead
    BLAS.gemm!('N', 'C', T(1. / N), Mt, Mt, T(0), C)
    return conj(C)
end