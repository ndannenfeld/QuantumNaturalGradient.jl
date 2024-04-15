struct SparseGeometricTensor{T <: Number}
    data::AbstractArray{T, 2}
    data_mean::Vector{T}
    importance_weights::Union{Vector{<:Real}, Nothing}
    function SparseGeometricTensor(m::AbstractMatrix{T}; importance_weights=nothing, mean_=nothing) where T <: Number
        if mean_ === nothing
            data_mean = wmean(m; weights=importance_weights, dims=1)[1, :]
        else
            data_mean = mean_
        end
        
        m = m .- data_mean
        return SparseGeometricTensor(m, data_mean; importance_weights)
    end
    function SparseGeometricTensor(m::AbstractMatrix{T}, data_mean::Vector{T}; importance_weights=nothing) where T <: Number
        if importance_weights !== nothing
            m = m .* sqrt.(importance_weights)
        end
        return new{T}(m, data_mean, importance_weights)
    end
    function SparseGeometricTensor(m::Vector{Vector{T}}; importance_weights=nothing, mean_=nothing) where T <: Number
        m, data_mean = convert_to_matrix_without_mean(m; weights=importance_weights, mean_)
        return SparseGeometricTensor(m, data_mean; importance_weights)
    end
end

Base.size(GT::SparseGeometricTensor) = size(GT.data)
Base.size(GT::SparseGeometricTensor, i) = size(GT.data, i)
Base.length(GT::SparseGeometricTensor) = size(GT.data, 1)
function get_importance_weights(GT::SparseGeometricTensor)
    if GT.importance_weights === nothing
        return ones(length(GT))
    else
        return GT.importance_weights
    end
end

function centered(GT::SparseGeometricTensor; mode=:importance_sqrt)
    if GT.importance_weights === nothing
        return GT.data
    end
    if mode == :importance_sqrt
        return GT.data
    elseif mode == :importance
        return GT.data .* sqrt.(GT.importance_weights)
    elseif mode == :no_importance
        return GT.data ./ sqrt.(GT.importance_weights)
    else
        error("mode should be :importance_sqrt, :importance or :no_importance. $mode was given.")
    end
end
    
function uncentered(GT::SparseGeometricTensor)
    GTd = centered(GT; mode=:no_importance)
    return GTd .+ reshape(GT.data_mean, 1, :)
end

Statistics.mean(GT::SparseGeometricTensor) = GT.data_mean

function dense_T(G::SparseGeometricTensor)
    GT = centered(G)
    return GT * GT'
end

function dense_S(G::SparseGeometricTensor)
    GT = centered(G)
    return (GT' * GT) ./ length(G)
end

mutable struct NaturalGradient{T <: Number}
    samples
    GT::SparseGeometricTensor{T}
    Es::EnergySummary
    logψσs::Vector{Complex{Float64}}
    grad
    θdot
    tdvp_error::Union{Real, Nothing}
    importance_weights::Union{Vector{<:Real}, Nothing}
    function NaturalGradient(samples, GT::SparseGeometricTensor{T}, Es::EnergySummary,
         logψσs::Vector{Complex{Float64}}, θdot=nothing,
          tdvp_error::Union{Float64, Nothing}=nothing;
          importance_weights=nothing, grad=nothing) where {T <: Number}

        return new{T}(samples, GT, Es, logψσs, grad, θdot, tdvp_error, importance_weights)
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
    kwargs = Dict{Any, Any}(kwargs...)

    if length(out) == 4
        Oks, Eks, logψσs, samples = out
    elseif length(out) == 5
        Oks, Eks, logψσs, samples, kwargs[:importance_weights] = out
    else 
        error("Oks_and_Eks should return 4 or 5 values. If 4 are returned, importance_weights is assumed to be  equal to 1.")
    end

    if Oks isa Tuple
        @assert length(Oks) == 2 "Oks should be a Tuple with 2 elements, Oks and Oks_mean"
        Oks, kwargs[:Oks_mean] = Oks
    end

    if Eks isa Tuple
        @assert length(Eks) == 3 "Eks should be a Tuple with 3 elements, Eks, Ek_mean and Ek_var"
        Eks, kwargs[:Eks_mean], kwargs[:Eks_var] = Eks
    end

    NaturalGradient(Oks, Eks, logψσs, samples; timer, kwargs...)
end

function NaturalGradient(Oks, Eks::Vector, logψσs::Vector, samples;
    importance_weights=nothing, Eks_mean=nothing, Eks_var=nothing, Oks_mean=nothing,
    solver=nothing, discard_outliers=0., timer=TimerOutput(), verbose=true) 

    if importance_weights !== nothing
        importance_weights ./= mean(importance_weights)
    end

    if discard_outliers > 0
        Eks, Oks, logψσs, samples, importance_weights = remove_outliers!(Eks, Oks, logψσs, samples, importance_weights; importance_weights, cut=discard_outliers, verbose)
    end
    
    Es = EnergySummary(Eks; importance_weights, mean_=Eks_mean, var_=Eks_var)
    GT = @timeit timer "copy Oks" SparseGeometricTensor(Oks; importance_weights, mean_=Oks_mean)

    sr = NaturalGradient(samples, GT, Es, logψσs; importance_weights)

    if solver !== nothing
        @timeit timer "solver" solver(sr)
    end

    return sr
end

function get_gradient(sr::NaturalGradient)
    if sr.grad === nothing
        sr.grad = centered(sr.GT)' * centered(sr.Es) .* (2/ length(sr.Es))
    end
    return sr.grad
end


function tdvp_error(sr::NaturalGradient)
    return tdvp_error(sr.GT, sr.Es, get_gradient(sr)./2, sr.θdot)
end

function tdvp_error!(sr::NaturalGradient)
    sr.tdvp_error = tdvp_error(sr)
    return sr.tdvp_error
end


function tdvp_error(sr::NaturalGradient, SR_control::NaturalGradient)
    return tdvp_error(SR_control.GT, SR_control.Es, get_gradient(SR_control)./2, sr.θdot)
end

function tdvp_error!(sr::NaturalGradient, SR_control::NaturalGradient)
    sr.tdvp_error = tdvp_error(sr, SR_control)
    return sr.tdvp_error
end

function tdvp_error(GT::SparseGeometricTensor, Es::EnergySummary, grad_half::Vector, θdot::Vector)
    var_E = var(Es)

    Eks_eff = -(centered(GT) * θdot) 

    Eks = centered(Es)
    #relative_error = std(Eks_eff .- Eks) / (std(Eks) + 1e-10)

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
    Eks_eff = -(centered(GT) * θdot)
    Eks = centered(Es)
    relative_error = std(Eks_eff .- Eks) / (std(Eks) + 1e-10)
    return relative_error
end