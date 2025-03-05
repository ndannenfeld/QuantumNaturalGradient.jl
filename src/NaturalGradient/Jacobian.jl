struct Jacobian{T <: Number}
    data::AbstractArray{T, 2}
    data_mean::Vector{T}
    importance_weights::Union{Vector{<:Real}, Nothing}
    function Jacobian(m::AbstractMatrix{T}; importance_weights=nothing, mean_=nothing) where T <: Number
        if mean_ === nothing
            data_mean = wmean(m; weights=importance_weights, dims=1)
        else
            data_mean = reshape(mean_, 1, :)
        end
        
        m = m .- data_mean
        return Jacobian(m, data_mean[1, :]; importance_weights)
    end
    function Jacobian(m::AbstractMatrix{T}, data_mean::Vector{T}; importance_weights=nothing) where T <: Number
        if importance_weights !== nothing
            m = m .* sqrt.(importance_weights)
        end
        return new{T}(m, data_mean, importance_weights)
    end
    function Jacobian(m::Vector{Vector{T}}; importance_weights=nothing, mean_=nothing) where T <: Number
        m, data_mean = convert_to_matrix_without_mean(m; weights=importance_weights, mean_)
        return Jacobian(m, data_mean; importance_weights)
    end
end

Base.size(J::Jacobian) = size(J.data)
Base.size(J::Jacobian, i) = size(J.data, i)
nr_parameters(J::Jacobian) = size(J.data, 2)
nr_samples(J::Jacobian) = size(J.data, 1)

Base.length(J::Jacobian) = size(J.data, 1)
function get_importance_weights(J::Jacobian)
    if J.importance_weights === nothing
        return ones(nr_samples(J))
    else
        return J.importance_weights
    end
end

function centered(J::Jacobian; mode=:importance_sqrt)
    if J.importance_weights === nothing
        return J.data
    end
    if mode == :importance_sqrt
        return J.data
    elseif mode == :importance
        return J.data .* sqrt.(J.importance_weights)
    elseif mode == :no_importance
        return J.data ./ sqrt.(J.importance_weights)
    else
        error("mode should be :importance_sqrt, :importance or :no_importance. $mode was given.")
    end
end
    
function uncentered(J::Jacobian)
    Jd = centered(J; mode=:no_importance)
    return Jd .+ reshape(J.data_mean, 1, :)
end

Statistics.mean(J::Jacobian) = J.data_mean

function dense_T(J::Jacobian)
    J = centered(J)
    # J * J' is slow, so we use BLAS.gemm! instead
    C = Matrix{eltype(J)}(undef, size(J, 1), size(J, 1))
    return BLAS.gemm!('N', 'C', 1., J, J, 0., C)
end

function dense_S(J::Jacobian)
    J = centered(J)
    # (J' * J) ./ nr_parameters(G)
    C = Matrix{eltype(J)}(undef, size(J, 2), size(J, 2))
    return BLAS.gemm!('T', 'N', 1/nr_samples(G), J, J, 0., C)
end

function Base.show(io::IO, J::Jacobian)
    print(io, "Jacobian(Nₚ=$(nr_parameters(J)), Nₛ=$(nr_samples(J))")
end