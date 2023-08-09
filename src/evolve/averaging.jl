abstract type AbstractIntegratorAveraging <: AbstractIntegrator end


@with_kw mutable struct EulerAveraging <: AbstractIntegratorAveraging
    lr::Float64
    beta::Float64 = 0.9
    memory_size::Union{Integer, Nothing} = nothing
    use_clipping::Bool = false
    clip_norm::Float64 = 5
    clip_val::Float64 = 1
    Eks = nothing
    Oks = nothing
    outlier_threshold::Float64 = 10
end

function (integrator::EulerAveraging)(θ::AbstractVector, Oks_and_Eks_; solver=nothing, kwargs...)
    @assert solver !== nothing "solver must be specified"
    sr = NaturalGradient(θ, Oks_and_Eks_; kwargs...)

    if integrator.memory_size !== nothing
        memory_size = integrator.memory_size
    else
        memory_size = length(sr)
    end

    Oksc = sr.GT.data ./ sqrt(length(sr))
    Eksc = centered(sr.Es) ./ sqrt(length(sr))

    if integrator.Oks !== nothing
        if integrator.outlier_threshold < norm(Eksc) / norm(integrator.Eks)
            random_number = rand(1:100000)
            name = "outlier_$random_number.jld2"
            @info "Outlier detected, resampling, saving as $name"
            save(name , "sr", sr)
            sr = NaturalGradient(θ, Oks_and_Eks_; kwargs...)
            Oksc = sr.GT.data ./ sqrt(length(sr))
            Eksc = centered(sr.Es) ./ sqrt(length(sr))
        end

        beta_sqrt = sqrt(integrator.beta)
        betam_sqrt = sqrt(1-integrator.beta)

        Oksc = cat(beta_sqrt .* integrator.Oks,  betam_sqrt .* Oksc, dims=1)
        Eksc = vcat(beta_sqrt .* integrator.Eks, betam_sqrt .* Eks)
        @show norm(Eksc), norm(Eks), norm(integrator.Eks)

        memory_size = min(memory_size, length(sr) + length(integrator.Eks))
    else
        

        memory_size = min(memory_size, length(sr))
    end
    
    U = RandomizedLinAlg.rrange(Oksc, memory_size)'
    integrator.Oks = U * Oksc
    integrator.Eks = U * Eksc

    GTd = integrator.Oks * integrator.Oks'

    local θdot_raw
    if :timer in keys(kwargs)
        θdot_raw = @timeit kwargs[:timer] "solver" -solver(GTd, integrator.Eks)
    else
        θdot_raw = -solver(GTd, integrator.Eks)
    end
    
    sr.θdot = integrator.Oks' * θdot_raw

    tdvp_error!(sr)

    θdot = get_θdot(sr; θtype=eltype(θ))
    if integrator.use_clipping
        clamp_and_norm!(θdot, integrator.clip_val, integrator.clip_norm)
    end

    θ = θ + integrator.lr .* θdot
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