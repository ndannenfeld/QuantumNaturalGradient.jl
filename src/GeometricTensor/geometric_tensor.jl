struct SparseGeometricTensor{T <: Number}
    data::AbstractArray{T, 2}
    data_mean::Vector{T}
    function SparseGeometricTensor(m::AbstractMatrix{T}; importance_weights=nothing) where T <: Number
        data_mean = wmean(m; weights=importance_weights, dims=1)
        m = m .- data_mean
        return SparseGeometricTensor(m, data_mean[1, :]; importance_weights)
    end
    function SparseGeometricTensor(m::AbstractMatrix{T}, data_mean::Vector{T}; importance_weights=nothing) where T <: Number
        if importance_weights !== nothing
            m = m .* sqrt.(importance_weights)
        end
        return new{T}(m, data_mean)
    end
    function SparseGeometricTensor(m::Vector{Vector{T}}; importance_weights=nothing) where T <: Number
        m, data_mean = convert_to_matrix_without_mean(m; weights=importance_weights)
        return SparseGeometricTensor(m, data_mean; importance_weights)
    end
end

Base.size(GT::SparseGeometricTensor) = size(GT.data)
Base.size(GT::SparseGeometricTensor, i) = size(GT.data, i)
Base.length(GT::SparseGeometricTensor) = size(GT.data, 1)

dense_T(G::SparseGeometricTensor) = G.data * G.data'
dense_S(G::SparseGeometricTensor) = G.data' * G.data ./ size(G.data, 1)

mutable struct NaturalGradient{T <: Number, T2 <: Number, Tint <: Integer}
    samples::Vector{Vector{Tint}}
    GT::SparseGeometricTensor{T}
    Es::EnergySummary
    logψσs::Vector{Complex{Float64}}
    grad::Vector{T2}
    θdot::Union{Vector{T2}, Nothing}
    tdvp_error::Union{Real, Nothing}
    importance_weights::Union{Vector{<:Real}, Nothing}
    function NaturalGradient(samples::Vector{Vector{Tint}}, GT::SparseGeometricTensor{T}, Es::EnergySummary, logψσs::Vector{Complex{Float64}}, grad::Vector{T2}, θdot::Union{Vector{T2}, Nothing}=nothing, tdvp_error::Union{Float64, Nothing}=nothing; importance_weights=nothing) where {T <: Number, T2 <: Number, Tint <: Integer}
        return new{T, T2, Tint}(samples, GT, Es, logψσs, grad, θdot, tdvp_error, importance_weights)
    end
    function NaturalGradient(samples::Matrix{Tint}, GT::SparseGeometricTensor{T}, Es::EnergySummary, logψσs::Vector{Complex{Float64}}, grad::Vector{T2}, θdot::Union{Vector{T2}, Nothing}=nothing, tdvp_error::Union{Float64, Nothing}=nothing; importance_weights=nothing) where {T <: Number, T2 <: Number, Tint <: Integer}
        return new{T, T2, Tint}(convert_to_vector(samples), GT, Es, logψσs, grad, θdot, tdvp_error, importance_weights)
    end
end
function convert_to_vector(samples::Matrix{T}) where T <: Integer
    return [Vector{T}(samples[i, :]) for i in 1:size(samples, 1)]
end

Base.length(sr::NaturalGradient) = length(sr.Es)
Base.show(io::IO, sr::NaturalGradient) = print(io, "NaturalGradient($(sr.Es), tdvp_error=$(sr.tdvp_error))")


function get_θdot(sr::NaturalGradient; θtype=ComplexF64)
    if eltype(sr.θdot) <: Real
        return real(θtype).(sr.θdot)
    else
        if θtype <: Real
            return θtype.(real.(sr.θdot))
        else
            return sr.θdot
        end
    end
end

function centered(Oks::Vector{Vector{T}}) where T <: Number
    m = mean(Oks)
    return [ok .- m for ok in Oks]
end

function NaturalGradient(θ::Vector, Oks_and_Eks; sample_nr=100, timer=TimerOutput(), kwargs...)
    out = @timeit timer "Oks_and_Eks" Oks_and_Eks(θ, sample_nr)
    if length(out) == 4
        Oks, Eks, logψσs, samples = out
        return NaturalGradient(Oks, Eks, logψσs, samples; timer, kwargs...)
    elseif length(out) == 5
        Oks, Eks, logψσs, samples, importance_weights = out
        return NaturalGradient(Oks, Eks, logψσs, samples; timer, importance_weights, kwargs...)
    else 
        error("Oks_and_Eks should return 4 or 5 values")
    end
    
end

function NaturalGradient(Oks, Eks::Vector, logψσs::Vector, samples::Union{Vector, Matrix};
    importance_weights=nothing, solver=nothing, discard_outliers=0., timer=TimerOutput(), kwargs...) 
    if discard_outliers > 0
        l = max(Int(round(length(Eks) * discard_outliers / 2)), 1)
        s = sortperm(Eks);
        remove = sort(vcat(s[1:l], s[end-l+1:end]))
        deleteat!(Eks, remove)
        if importance_weights !== nothing
            deleteat!(importance_weights, remove)
        end
        deleteat!(logψσs, remove)
        deleteat!(Oks, remove)
        deleteat!(samples, remove)
    end
    if importance_weights !== nothing
        importance_weights ./= mean(importance_weights)
    end
    
    Es = EnergySummary(Eks; importance_weights)

    Ekms = centered(Es)

    GT = @timeit timer "copy Oks" SparseGeometricTensor(Oks; importance_weights)
    grad = @timeit timer "grad" 2 * GT.data' * Ekms ./ length(Es)

    sr = NaturalGradient(samples, GT, Es, logψσs, grad; importance_weights)

    if solver !== nothing
        @timeit timer "solver" solver(sr)
    end

    return sr
end

function NaturalGradient(θ, construct_mps, H::MPO;
                                   sample_nr::Integer=100, parallel=false,
                                   solver=nothing, discard_outliers=0.,
                                   kwargs...)
    
    if parallel
        Oks_and_Eks_ = generate_Oks_and_Eks_parallel(construct_mps, H; kwargs...)
    else
        Oks_and_Eks_ = (θ, sample_nr) -> Oks_and_Eks(θ, construct_mps, H, sample_nr; kwargs...)
    end
    return NaturalGradient(θ, Oks_and_Eks_; sample_nr, solver, discard_outliers)
end


function tdvp_error(sr::NaturalGradient)
    return tdvp_error(sr.GT, sr.Es, sr.grad./2, sr.θdot)
end

function tdvp_error!(sr::NaturalGradient)
    sr.tdvp_error = tdvp_error(sr)
    return sr.tdvp_error
end


function tdvp_error(sr::NaturalGradient, SR_control::NaturalGradient)
    return tdvp_error(SR_control.GT, SR_control.Es, SR_control.grad./2, sr.θdot)
end

function tdvp_error!(sr::NaturalGradient, SR_control::NaturalGradient)
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

function tdvp_relative_error(sr::NaturalGradient)
    return tdvp_relative_error(sr.GT, sr.Es, sr.θdot)
end

function tdvp_relative_error(sr::NaturalGradient, sr_control::NaturalGradient)
    return tdvp_relative_error(sr_control.GT, sr_control.Es, sr.θdot)
end

function tdvp_relative_error(GT::SparseGeometricTensor, Es::EnergySummary, θdot::Vector)
    Eks_eff = -(GT.data * θdot)
    Eks = centered(Es)
    relative_error = std(Eks_eff .- Eks) / (std(Eks) + 1e-10)
    return relative_error
end