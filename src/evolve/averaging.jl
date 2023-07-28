abstract type AbstractIntegratorAveraging <: AbstractIntegrator end

mutable struct EulerAveraging <: AbstractIntegratorAveraging
    lr::Float64
    beta::Float64
    memory_size::Union{Integer, Nothing}
    Eks
    Oks
    EulerAveraging(lr=0.1, beta=0.9, memory_size=nothing) = new(lr, beta, memory_size, nothing, nothing)
end

function (integrator::EulerAveraging)(θ::AbstractVector, Oks_and_Eks_; solver=nothing, kwargs...)
    @assert solver !== nothing "solver must be specified"
    sr = NaturalGradient(θ, Oks_and_Eks_; kwargs...)

    if integrator.memory_size !== nothing
        memory_size = integrator.memory_size
    else
        memory_size = length(sr)
    end

    Eks = centered(sr.Es)
    if integrator.Oks !== nothing
        factor = norm(integrator.Eks) / norm(Eks)

        beta = integrator.beta

        Oksc = cat(factor * beta .* integrator.Oks,  (1-beta) .* sr.GT.data, dims=1)
        Eksc = vcat(factor * beta .* integrator.Eks, (1-beta) .* Eks)

        memory_size = min(memory_size, length(sr) + length(integrator.Eks))
    else
        Oksc = sr.GT.data
        Eksc = Eks

        memory_size = min(memory_size, length(sr))
    end
    
    U = RandomizedLinAlg.rrange(Oksc, memory_size)'
    integrator.Oks = U * Oksc
    integrator.Eks = U * Eksc

    GTd = integrator.Oks * integrator.Oks'
    θdot_raw = -solver(GTd, integrator.Eks)
    sr.θdot = integrator.Oks' * θdot_raw

    tdvp_error!(sr)

    θ = θ + integrator.lr .* get_θdot(sr; θtype=eltype(θ))
    return θ, sr
end

mutable struct EulerAveragingS <: AbstractIntegratorAveraging
    lr::Float64
    beta::Float64
    Fs
    GTd
    EulerAveragingS(lr=0.1, beta=0.9) = new(lr, beta, nothing, nothing)
end

function (integrator::EulerAveragingS)(θ::AbstractVector, Oks_and_Eks_; solver=nothing, kwargs...)
    @assert solver !== nothing "solver must be specified"
    sr = NaturalGradient(θ, Oks_and_Eks_; kwargs...)

    Gtd = dense_S(sr.GT)
    Fs = sr.grad ./ 2
    
    if integrator.Fs !== nothing
        integrator.Fs = Fs
        integrator.GTd = Gtd
    else
        integrator.Fs = beta .* integrator.Fs + (1-beta) .* Fs
        integrator.GTd = beta .* integrator.GTd + (1-beta) .* Gtd
    end

    sr.θdot = -solver(integrator.GTd, integrator.Fs)

    tdvp_error!(sr)

    θ = θ + integrator.lr .* get_θdot(sr; θtype=eltype(θ))
    return θ, sr
end