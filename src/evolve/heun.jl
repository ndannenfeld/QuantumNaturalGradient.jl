mutable struct Heun <: AbstractIntegrator
    lr::Float64
    ϵ::Float64
    set_seed::Bool
    adaptive::Bool
    
    Heun(lr=0.1, ϵ=1e-6; set_seed=false, adaptive=false) = new(lr, ϵ, set_seed, adaptive)
end

function (integrator::Heun)(θ::AbstractVector, Oks_and_Eks_; kwargs...)
    Random.seed!(1)
    if integrator.set_seed
        seed = rand(UInt)
    end
    
    lr = integrator.lr
    
    integrator.set_seed && Random.seed!(seed)
    sr1 = NaturalGradient(θ, Oks_and_Eks_; kwargs...)
    k1 = get_θdot(sr1; θtype=eltype(θ))
    
    integrator.set_seed && Random.seed!(seed)
    θ2 = θ .+ lr .* k1
    sr2 = NaturalGradient(θ2, Oks_and_Eks_; kwargs...)
    k2 = get_θdot(sr2; θtype=eltype(θ))
    
    θ = θ .+ lr .* (k1 .+ k2) ./ 2
    #Δθ = (k2.- k1) ./ 2 
    #Δθ2 = (k2.+ k1) ./ 2 
    @assert ! integrator.adaptive "Adaptive Heun not implemented yet"
    
    
    #Δθ_n = norm(centered(sr1.J) * Δθ) / sqrt(length(sr1))
    #tdvp_error_ = (QuantumNaturalGradient.tdvp_relative_error(sr1) + QuantumNaturalGradient.tdvp_relative_error(sr2)) ./2
    #tdvp_error = (QuantumNaturalGradient.tdvp_relative_error(sr1, sr2) + QuantumNaturalGradient.tdvp_relative_error(sr2, sr1)) ./ 2
    
    
    return θ
end