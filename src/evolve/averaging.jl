abstract type AbstractIntegratorAveraging <: AbstractIntegrator end

function (integrator::AbstractIntegratorAveraging)(θ::AbstractVector, Oks_and_Eks_; solver=nothing, kwargs...)
    @assert solver !== nothing "solver must be specified"
    sr = NaturalGradient(θ, Oks_and_Eks_; kwargs...)

    if integrator.memory_size !== nothing
        memory_size = integrator.memory_size
    else
        memory_size = length(sr)
    end

    if integrator.solver_size !== nothing
        solver_size = integrator.solver_size
    else
        solver_size = memory_size
    end

    Eks_solve, Oks_solve = get_effective_Oks_and_Eks!(integrator, sr, θ, memory_size, solver_size)

    GTd = Oks_solve * Oks_solve'

    local θdot_raw
    if :timer in keys(kwargs)
        θdot_raw = @timeit kwargs[:timer] "solver" -solver(GTd, Eks_solve)
    else
        θdot_raw = -solver(GTd, Eks_solve)
    end
    
    sr.θdot = Oks_solve' * θdot_raw

    tdvp_error!(sr)

    θdot = get_θdot(sr; θtype=eltype(θ))
    if integrator.use_clipping
        clamp_and_norm!(θdot, integrator.clip_val, integrator.clip_norm)
    end

    @. θ = θ + integrator.lr * θdot
    return θ, sr
end

@with_kw mutable struct EulerAveraging <: AbstractIntegratorAveraging
    lr::Float64
    beta::Float64 = 0.9
    memory_size::Union{Integer, Nothing} = nothing
    solver_size::Union{Integer, Nothing} = nothing
    use_clipping::Bool = false
    clip_norm::Float64 = 5
    clip_val::Float64 = 1
    Eks = nothing
    Oks = nothing
    outlier_threshold::Float64 = 10
    state::IdDict{Any, Any} = IdDict()
end

function get_effective_Oks_and_Eks!(integrator::EulerAveraging, sr, θ, memory_size, solver_size)
    
    Oks = centered(sr.GT) ./ sqrt(length(sr))
    Eks = centered(sr.Es) ./ sqrt(length(sr))

    beta_sqrt = sqrt(integrator.beta)
    betam_sqrt = sqrt(1-integrator.beta)

    βp, = get!(integrator.state, θ) do
        (Float64[1.],)
    end

    βp .*= integrator.beta

    if integrator.Oks === nothing
        Eksc = Eks .* betam_sqrt
        Oksc = Oks .* betam_sqrt
        memory_size = min(memory_size, length(sr))
        solver_size = min(solver_size, memory_size)
    else
        if integrator.outlier_threshold < norm(Eks) / norm(integrator.Eks)
            random_number = rand(1:100000)
            name = "outlier_$random_number.jld2"
            @info "Outlier detected, resampling, saving as $name"
            save(name , "sr", sr)
            sr = NaturalGradient(θ, Oks_and_Eks_; kwargs...)
            Oks = centered(sr.GT) ./ sqrt(length(sr))
            Eks = centered(sr.Es) ./ sqrt(length(sr))
        end

        factor = norm(Eks) / norm(integrator.Eks)
        @show factor

        Eksc = vcat(beta_sqrt .* integrator.Eks, betam_sqrt .* Eks)
        Oksc =  cat(beta_sqrt .* integrator.Oks, betam_sqrt .* Oks, dims=1)

        memory_size = min(memory_size, length(sr) + length(integrator.Eks))
        solver_size = min(solver_size, memory_size)
    end

    Us = RandomizedLinAlg.rrange(Oksc, solver_size)'
    Eks_solve = Us * Eksc
    Oks_solve = Us * Oksc
    
    if solver_size == memory_size
        integrator.Eks = Eks_solve
        integrator.Oks = Oks_solve
    elseif solver_size < memory_size 
        U = RandomizedLinAlg.rrange(Oksc, memory_size)'
        integrator.Eks = U * Eksc
        integrator.Oks = U * Oksc
    else
        U = RandomizedLinAlg.rrange(Oks_solve, memory_size)'
        integrator.Eks = U * Eks_solve
        integrator.Oks = U * Oks_solve
    end

    @show norm(Eks), norm(Eksc)/ sqrt(1 - βp[1]), norm(integrator.Eks)/ sqrt(1 - βp[1])

    return Eks_solve, Oks_solve 
end

@with_kw mutable struct EulerWindowing <: AbstractIntegratorAveraging
    lr::Float64
    beta::Float64 = 0.9
    memory_size::Union{Integer, Nothing} = nothing
    solver_size::Union{Integer, Nothing} = nothing
    use_clipping::Bool = false
    clip_norm::Float64 = 5
    clip_val::Float64 = 1
    Eks = nothing
    Oks = nothing
    weights = nothing
    outlier_threshold::Float64 = 10
    state::IdDict{Any, Any} = IdDict()
end

function get_effective_Oks_and_Eks!(integrator::EulerWindowing, sr, θ, memory_size, solver_size)
    Oks = uncentered(sr.GT)
    Eks = uncentered(sr.Es)

    #mean_Ek = mean(sr.Es)
    #mean_Ok = mean(sr.GT)

    βp, = get!(integrator.state, θ) do
        (Float64[1.],)
    end

    #mean_Ek_h .= mean_Ek_h .* beta_sqrt .+ mean_Ek .* betam_sqrt
    #mean_Ok_h .= mean_Ok_h .* beta_sqrt .+ mean_Ok .* betam_sqrt

    βp .*= integrator.beta

    if integrator.Eks === nothing
        integrator.Eks = Eks
        integrator.Oks = Oks

        integrator.weights = get_importance_weights(sr.GT) .* (1-integrator.beta)
        memory_size = min(memory_size, length(sr))
    else
        if integrator.outlier_threshold < norm(Eks) / norm(integrator.Eks)
            random_number = rand(1:100000)
            name = "outlier_$random_number.jld2"
            @info "Outlier detected, resampling, saving as $name"
            save(name , "sr", sr)
            sr = NaturalGradient(θ, Oks_and_Eks_; kwargs...)
            Oks = uncentered(sr.GT) ./ sqrt(length(sr))
            Eks = uncentered(sr.Es) ./ sqrt(length(sr))
        end
        f = length(Eks) / length(integrator.Eks) 
        @show norm(Eks), norm(integrator.Eks) * sqrt(f)
        integrator.Eks = vcat(integrator.Eks, Eks)
        integrator.Oks =  cat(integrator.Oks, Oks, dims=1)

        weights = get_importance_weights(sr.GT)
        integrator.weights = vcat(integrator.beta .* integrator.weights, (1-integrator.beta) .* weights)

        memory_size = min(memory_size, length(sr) + length(integrator.Eks))

        if memory_size < length(integrator.Eks)
            integrator.Eks = integrator.Eks[end - memory_size + 1:end]
            integrator.Oks = integrator.Oks[end - memory_size + 1:end, :]
            integrator.weights = integrator.weights[end - memory_size + 1:end]
        end
    end

    solver_size = min(solver_size, memory_size)

    weights = integrator.weights ./ mean(integrator.weights)
    sr_solve = NaturalGradient(integrator.Oks, integrator.Eks, zeros(ComplexF64, length(integrator.Eks)), zeros(Int, length(integrator.Eks), 1); importance_weights=weights)
    
    Eks_solve = centered(sr_solve.Es)
    Oks_solve = centered(sr_solve.GT)
    if solver_size != memory_size
        Us = RandomizedLinAlg.rrange(Oks_solve, solver_size)'
        Eks_solve = Us * Eks_solve
        Oks_solve = Us * Oks_solve
    end

    #@show norm(Eks), norm(Eksc)/ sqrt(1 - βp[1]), norm(integrator.Eks)/ sqrt(1 - βp[1])

    return Eks_solve, Oks_solve 
end

@with_kw mutable struct EulerNormAveraging <: AbstractIntegratorAveraging
    lr::Float64
    beta::Float64 = 0.9
    memory_size::Union{Integer, Nothing} = nothing
    solver_size::Union{Integer, Nothing} = nothing
    use_clipping::Bool = false
    clip_norm::Float64 = 5
    clip_val::Float64 = 1
    weights = nothing
    outlier_threshold::Float64 = 10
    state::IdDict{Any, Any} = IdDict()
end

function get_effective_Oks_and_Eks!(integrator::EulerNormAveraging, sr, θ, memory_size, solver_size)
    Oks = uncentered(sr.GT)
    Eks = uncentered(sr.Es)

    mean_Ek = mean(sr.Es)
    mean_Ok = mean(sr.GT)
    
    N, mean_Ek_h, mean_Ok_h = get!(integrator.state, θ) do
        (Float64[0.], 
        zeros(eltype(mean_Ek), 1),
        zeros(eltype(mean_Ok), size(mean_Ok)),
        )
    end
    
    @. mean_Ek_h = mean_Ek_h * integrator.beta + mean_Ek * (1-integrator.beta)
    @. mean_Ok_h = mean_Ok_h * integrator.beta + mean_Ok * (1-integrator.beta)
    @. N = N * integrator.beta + (1-integrator.beta)

    Eksc = Eks .- mean_Ek_h[1] ./ N[1]
    Oksc = Oks .- reshape(mean_Ok_h, 1, :) ./ N[1]

    solver_size = min(solver_size, length(sr))

    if sr.importance_weights !== nothing
        Oksc = Oksc .* sqrt.(reshape(sr.GT.importance_weights, :, 1))
        Eksc = Eksc .* sqrt.(sr.GT.importance_weights)
    end

    if solver_size < length(sr)
        Us = RandomizedLinAlg.rrange(Oksc, solver_size)'
        Eksc = Us * Eksc
        Oksc = Us * Oksc
    end

    return Eksc, Oksc
end