
abstract type AbstractTemperatureSchedule  <: AbstractSchedule end

VecNothing = Union{Vector{<:Real}, Nothing}
mutable struct LangevinNoise <: AbstractCompositeIntegrator
    integrator::AbstractIntegrator
    T_schedule::AbstractTemperatureSchedule
    θ_std::VecNothing
    norm_preserving::Bool
    L2_lr::Union{Float64, Nothing}
    verbose::Bool
    history::VecNothing
    history_view
end

function LangevinNoise(integrator::AbstractIntegrator, T_schedule::AbstractTemperatureSchedule;
    θ_std::VecNothing=nothing, norm_preserving::Bool=false, L2_lr=nothing,
    verbose::Bool=false, save_history=false)

    history = save_history ? Float64[] : nothing
    history_view = save_history ? reshape(view(history, :), 2, 0) : nothing

    return LangevinNoise(integrator, T_schedule, θ_std, norm_preserving, L2_lr, verbose, history, history_view)
end

function LangevinNoise(integrator::AbstractIntegrator, T::Real; steps::Integer=1, final_T::Float64=0.9, kwargs...)
    T_schedule = TemperatureExpDecay(T, final_T, steps)
    return LangevinNoise(integrator, T_schedule; kwargs...)
end

function (integrator::LangevinNoise)(θ::AbstractVector, Oks_and_Eks_; kwargs...)
    T = integrator.T_schedule.T
    lr = integrator.lr

    θ_old = copy(θ)
    θ, sr = integrator.integrator(θ, Oks_and_Eks_; kwargs...)

    σ = sqrt(2 * lr * T)
    if integrator.θ_std !== nothing
        σ = σ .* integrator.θ_std
    end

    dθ = θ - θ_old

    θn = θ .+ randn(length(θ)) .* σ

    if eltype(θ) <: Complex
        θn = θn .+ randn(length(θ)) .* (σ * im)
    end

    if integrator.L2_lr !== nothing
        f = integrator.L2_lr * lr * T
        θn -= θ .* f
    end

    noise_grad_ratio = norm(dθ) ./ sqrt(length(θ))
    if integrator.L2_lr !== nothing
        f = integrator.L2_lr * lr * T
        noise_grad_ratio = norm(dθ) / norm(θ) / f
    end

    if integrator.norm_preserving
        θn = θn .* (norm(θ) / norm(θn))
    end

    
    if integrator.history !== nothing
        append!(integrator.history, noise_grad_ratio, T)
        integrator.history_view = reshape(view(integrator.history, :), 2, length(integrator.history) ÷ 2)
    end

    if integrator.verbose
        println("T = ", T, ", noise_grad_ratio = ", noise_grad_ratio)
    end
    
    integrator.T_schedule(integrator, θ, noise_grad_ratio) # update T
    return θn, sr
end



mutable struct TemperatureExpDecay <: AbstractTemperatureSchedule
    T::Real
    T_decay_factor::Real
    function TemperatureExpDecay(T::Real, T_decay_factor::Real)
        if T_decay_factor > 1
            throw(ArgumentError("T_decay_factor must be smaller than 1"))
        end
        return new(T, T_decay_factor)
    end
    function TemperatureExpDecay(T::Real, final_T::Real, steps::Integer)
        T_decay_factor = (final_T/T)^(1/steps)
        return TemperatureExpDecay(T, T_decay_factor)
    end
end

function (schedule::TemperatureExpDecay)(integrator::LangevinNoise, θ::AbstractVector, noise_grad_ratio::Real)
    schedule.T *= schedule.T_decay_factor
end

include("adaptive.jl")