mutable struct DecayIntegrator <: AbstractCompositeIntegrator
    integrator::AbstractIntegrator
    L2_lr::Float64
    decay_factor::Float64
    DecayIntegrator(integrator::AbstractIntegrator, L2_lr::Float64=0.1, decay_factor::Float64=1.) = new(integrator, L2_lr, decay_factor)
end

function (integrator::DecayIntegrator)(θ::AbstractVector, Oks_and_Eks_; kwargs...)
    θ, sr = integrator.integrator(θ, Oks_and_Eks_; kwargs...)
    f = integrator.L2_lr * integrator.lr
    @. θ -= f * θ
    integrator.L2_lr *= integrator.decay_factor
    return θ, sr
end