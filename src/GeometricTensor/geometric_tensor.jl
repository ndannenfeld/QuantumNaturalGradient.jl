struct SparseGeometricTensor{T <: Number}
    data::AbstractArray{T, 2}
    data_mean::Vector{T}
    function SparseGeometricTensor(m::AbstractMatrix{T}) where T <: Number
        data_mean = mean(m, dims=1)
        m = m .- data_mean
        return new{T}(m, data_mean[1, :])
    end
    function SparseGeometricTensor(m::AbstractMatrix{T}, data_mean::Vector{T}) where T <: Number
        return new{T}(m, data_mean)
    end
    function SparseGeometricTensor(m::Vector{Vector{T}}) where T <: Number
        m, data_mean = convert_to_matrix_without_mean(m)
        return new{T}(m, data_mean)
    end
end

Base.size(GT::SparseGeometricTensor) = size(GT.data)
Base.size(GT::SparseGeometricTensor, i) = size(GT.data, i)
Base.length(GT::SparseGeometricTensor) = size(GT.data, 1)

dense_T(G::SparseGeometricTensor) = G.data * G.data'
dense_S(G::SparseGeometricTensor) = G.data' * G.data ./ size(G.data, 1)

mutable struct StochasticReconfiguration{T <: Number, T2 <: Number, Tint <: Integer}
    samples::Vector{Vector{Tint}}
    GT::SparseGeometricTensor{T}
    Es::EnergySummary
    logψσs::Vector{Complex{Float64}}
    grad::Vector{T2}
    θdot::Union{Vector{T2}, Nothing}
    tdvp_error::Union{Float64, Nothing}
end

Base.length(sr::StochasticReconfiguration) = length(sr.Es)
Base.show(io::IO, sr::StochasticReconfiguration) = print(io, "StochasticReconfiguration($(sr.Es), tdvp_error=$(sr.tdvp_error))")


function get_θdot(sr::StochasticReconfiguration; θtype=ComplexF64)
    if eltype(sr.θdot) <: Real
        return sr.θdot
    else
        if θtype <: Real
            return real.(sr.θdot)
        else
            return sr.θdot
        end
    end
end

function centered(Oks::Vector{Vector{T}}) where T <: Number
    m = mean(Oks)
    return [ok .- m for ok in Oks]
end

function StochasticReconfiguration(θ::Vector, Oks_and_Eks; sample_nr=100, kwargs...)
    Oks, Eks, logψσs, samples = Oks_and_Eks(θ, sample_nr)
    return StochasticReconfiguration(Oks, Eks, logψσs, samples; kwargs...)
end

function StochasticReconfiguration(Oks, Eks::Vector, logψσs::Vector, samples::Vector; solver=nothing, discard_outliers=0.)
    Es = EnergySummary(Eks)
    
    if discard_outliers > 0
        l = max(Int(round(length(Eks) * discard_outliers / 2)), 1)
        s = sortperm(Eks);
        remove = sort(vcat(s[1:l], s[end-l+1:end]))
        deleteat!(Eks, remove)
        deleteat!(Oks, remove)
        deleteat!(samples, remove)
    end

    Ekms = centered(Es)

    GT = SparseGeometricTensor(Oks)
    grad = 2 * GT.data' * Ekms ./ length(Es)

    sr = StochasticReconfiguration(samples, GT, Es, logψσs, grad, nothing, nothing)

    if solver !== nothing
        solver(sr)
    end

    return sr
end

function StochasticReconfiguration(θ, construct_mps, H::MPO;
                                   sample_nr::Integer=100, parallel=false,
                                   solver=nothing, discard_outliers=0.,
                                   kwargs...)
    
    if parallel
        Oks_and_Eks_ = generate_Oks_and_Eks_parallel(construct_mps, H; kwargs...)
    else
        Oks_and_Eks_ = (θ, sample_nr) -> Oks_and_Eks(θ, construct_mps, H, sample_nr; kwargs...)
    end
    return StochasticReconfiguration(θ, Oks_and_Eks_; sample_nr, solver, discard_outliers)
end


function tdvp_error(sr::StochasticReconfiguration)
    return tdvp_error(sr.GT, sr.Es, sr.grad./2, sr.θdot)
end

function tdvp_error!(sr::StochasticReconfiguration)
    sr.tdvp_error = tdvp_error(sr)
    return sr.tdvp_error
end


function tdvp_error(sr::StochasticReconfiguration, SR_control::StochasticReconfiguration)
    return tdvp_error(SR_control.GT, SR_control.Es, SR_control.grad./2, sr.θdot)
end

function tdvp_error!(sr::StochasticReconfiguration, SR_control::StochasticReconfiguration)
    sr.tdvp_error = tdvp_error(sr, SR_control)
    return sr.tdvp_error
end

function tdvp_error(GT::SparseGeometricTensor, Es::EnergySummary, grad_half::Vector, θdot::Vector)
    var_E = var(Es)

    Eks_eff = -(GT.data * θdot) 

    Eks = centered(Es)
    relative_error = std(Eks_eff .- Eks) / (std(Eks) + 1e-10)

    
    var_eff_1 = -Eks_eff' * Eks_eff / length(Es)
    # var_eff_1 = -var(Eks_eff)

    var_eff_1 = -var(Eks_eff)
    var_eff_2 = θdot' * grad_half - grad_half' * θdot
    var_eff = var_eff_1 + real(var_eff_2)

    return 1 + var_eff/var_E
end

function tdvp_relative_error(sr::StochasticReconfiguration)
    return tdvp_relative_error(sr.GT, sr.Es, sr.θdot)
end

function tdvp_relative_error(sr::StochasticReconfiguration, sr_control::StochasticReconfiguration)
    return tdvp_relative_error(sr_control.GT, sr_control.Es, sr.θdot)
end

function tdvp_relative_error(GT::SparseGeometricTensor, Es::EnergySummary, θdot::Vector)
    Eks_eff = -(GT.data * θdot)
    Eks = centered(Es)
    relative_error = std(Eks_eff .- Eks) / (std(Eks) + 1e-10)
    return relative_error
end