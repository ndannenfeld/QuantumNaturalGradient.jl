mutable struct LangevinNoise <: AbstractIntegrator
    integrator::AbstractIntegrator
    T::Float64
    decay_factor::Float64
    verbose::Bool
    history::Union{Vector{Float64}, Nothing}
    LangevinNoise(integrator::AbstractIntegrator, T::Float64, decay_factor::Float64; verbose::Bool=false, save_history=false) = new(integrator, T, decay_factor, verbose)
    function LangevinNoise(integrator::AbstractIntegrator, T::Float64; steps::Integer=1, final_T::Float64=0.9, verbose::Bool=false, save_history=false) 
        history = save_history ? Float64[] : nothing
        return new(integrator, T, (final_T/T)^(1/steps), verbose, history)
    end
end

function (integrator::LangevinNoise)(θ::AbstractVector, Oks_and_Eks_; kwargs...)
    θ_old = copy(θ)
    θ, sr = integrator.integrator(θ, Oks_and_Eks_; kwargs...)
    σ = sqrt(2*integrator.integrator.lr * integrator.T)

    dθ = θ - θ_old
    noise_grad_ratio = norm(dθ) / σ^2 / sqrt(length(θ))
    if integrator.history !== nothing
        push!(integrator.history, noise_grad_ratio)
    end

    if integrator.verbose
        println("T = ", integrator.T, ", noise_grad_ratio = ", noise_grad_ratio)
    end

    θ += randn(length(θ)) * σ

    if eltype(θ) <: Complex
        θ = θ + randn(length(θ)) * σ * im
    end
    
    integrator.T *= integrator.decay_factor
    return θ, sr
end