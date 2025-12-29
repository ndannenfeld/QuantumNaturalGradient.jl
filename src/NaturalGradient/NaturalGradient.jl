include("EnergySummary.jl")
include("Jacobian.jl")
include("Oks.jl")

mutable struct NaturalGradient{T <: Number}
    samples
    J::Jacobian{T}
    Es::EnergySummary
    logψσs::Vector{Complex{Float64}}
    grad
    θdot
    tdvp_error::Union{Real, Nothing}
    importance_weights::Union{Vector{<:Real}, Nothing}
    saved_properties
    function NaturalGradient(samples, J::Jacobian{T}, Es::EnergySummary,
         logψσs::Vector{Complex{Float64}}, θdot=nothing,
          tdvp_error::Union{Float64, Nothing}=nothing;
          importance_weights=nothing, grad=nothing, saved_properties=nothing) where {T <: Number}

        return new{T}(samples, J, Es, logψσs, grad, θdot, tdvp_error, importance_weights, saved_properties)
    end
end
function convert_to_vector(samples::Matrix{T}) where T <: Integer
    return [Vector{T}(samples[i, :]) for i in 1:size(samples, 1)]
end

Base.length(ng::NaturalGradient) = length(ng.Es)
Base.show(io::IO, ng::NaturalGradient) = print(io, "NaturalGradient($(ng.Es), tdvp_error=$(ng.tdvp_error))")


function get_θdot(ng::NaturalGradient; θtype=ComplexF64)
    if eltype(ng.θdot) === θtype
        return ng.θdot
    end
    if eltype(ng.θdot) <: Real
        return real(θtype).(ng.θdot)
    else
        if θtype <: Real
            return θtype.(real.(ng.θdot))
        else
            return ng.θdot
        end
    end
end

function centered(Oks::Vector{Vector{T}}) where T <: Number
    m = mean(Oks)
    return [ok .- m for ok in Oks]
end

function NaturalGradient(θ::ParameterTypes, Oks_and_Eks; sample_nr=100, timer=TimerOutput(), kwargs_Oks_and_Eks=Dict(), kwargs...)
    out = @timeit timer "Oks_and_Eks" Oks_and_Eks(θ, sample_nr; kwargs_Oks_and_Eks...)
    kwargs = Dict{Any, Any}(kwargs...)
    saved_properties = Dict{Symbol, Any}()

    if haskey(out, :Eks)
        Eks = out[:Eks]
    else error("Oks_and_Eks should return Dict with key :Eks") end
    if haskey(out, :Oks)
        Oks = out[:Oks]
    else error("Oks_and_Eks should return Dict with key :Oks") end
    if haskey(out, :logψs)
        logψσs = out[:logψs]
    else error("Oks_and_Eks should return Dict with key :logψs") end
    if haskey(out, :samples)
        samples = out[:samples]
    else error("Oks_and_Eks should return Dict with key :samples") end
    if haskey(out, :weights)
        kwargs[:importance_weights] = out[:weights]
    end

    for key in keys(out)
        if !(key in [:Eks, :Oks, :logψs, :samples, :weights])
            saved_properties[key] = out[key]
        end
    end
    kwargs[:saved_properties] = saved_properties

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
    solver=nothing, discard_outliers=0., timer=TimerOutput(), verbose=true, saved_properties=nothing)

    if importance_weights !== nothing
        importance_weights ./= mean(importance_weights)
    end

    if discard_outliers > 0
        Eks, Oks, logψσs, samples, importance_weights = remove_outliers!(Eks, Oks, logψσs, samples, importance_weights; importance_weights, cut=discard_outliers, verbose)
    end
    
    Es = EnergySummary(Eks; importance_weights, mean_=Eks_mean, var_=Eks_var)
    J = @timeit timer "jacobi_mean" Jacobian(Oks; importance_weights, mean_=Oks_mean)

    ng = NaturalGradient(samples, J, Es, logψσs; importance_weights, saved_properties)

    # if solver !== nothing, then it's a function that has been passed to the evolve funnction earlier (cf. min. working example: `solver = QNG.EigenSolver()`)
    if solver !== nothing
        @timeit timer "solver" solver(ng; timer)
    end

    return ng
end

function get_gradient(ng::NaturalGradient)
    if ng.grad === nothing
        ng.grad = centered(ng.J)' * centered(ng.Es) .* (2/ length(ng.Es))
    end
    return ng.grad
end


function tdvp_error(ng::NaturalGradient)
    return tdvp_error(ng.J, ng.Es, get_gradient(ng)./2, ng.θdot)
end

function tdvp_error!(ng::NaturalGradient)
    ng.tdvp_error = tdvp_error(ng)
    return ng.tdvp_error
end


function tdvp_error(ng::NaturalGradient, ng_control::NaturalGradient)
    return tdvp_error(ng_control.J, ng_control.Es, get_gradient(ng_control)./2, ng.θdot)
end

function tdvp_error!(ng::NaturalGradient, ng_control::NaturalGradient)
    ng.tdvp_error = tdvp_error(ng, ng_control)
    return ng.tdvp_error
end

function tdvp_error(J::Jacobian, Es::EnergySummary, grad_half::Vector, θdot::Vector)
    var_E = var(Es)

    Eks_eff = -(centered(J) * θdot)
    Eks_eff = centered(Es) 
    var_eff_1 = -Eks_eff' * Eks_eff / (length(Es) - 1)

    f = length(Es) / (length(Es) - 1)
    var_eff_2 = θdot' * grad_half * f
    
    var_eff = var_eff_1 + real(var_eff_2)

    return 1 + var_eff/var_E/2
end

function tdvp_relative_error(ng::NaturalGradient)
    return tdvp_relative_error(ng.J, ng.Es, ng.θdot)
end

function tdvp_relative_error(ng::NaturalGradient, sr_control::NaturalGradient)
    return tdvp_relative_error(sr_control.J, sr_control.Es, ng.θdot)
end

function tdvp_relative_error(J::Jacobian, Es::EnergySummary, θdot::Vector)
    Eks_eff = -(centered(J) * θdot)
    Eks = centered(Es)
    relative_error = std(Eks_eff .- Eks) / (std(Eks) + 1e-10)
    return relative_error
end

include("outlier.jl")