abstract type AbstractIntegrator end
abstract type AbstractCompositeIntegrator <: AbstractIntegrator end
abstract type AbstractShedule end

function Base.getproperty(integrator::AbstractCompositeIntegrator, property::Symbol)
    if property === :lr
        return getproperty(integrator.integrator, property)
    else
        return getfield(integrator, property)
    end
end

struct Euler <: AbstractIntegrator
    lr::Float64
end

function (integrator::Euler)(θ::AbstractVector, Oks_and_Eks_; kwargs...)
    sr = StochasticReconfiguration(θ, Oks_and_Eks_; kwargs...)
    θ = θ + integrator.lr * get_θdot(sr; θtype=eltype(θ))
    return θ, sr
end

include("heun.jl")
include("averaging.jl")
include("noise.jl")
include("decay.jl")

function evolve(construct_mps, θ::T, H::MPO;
    integrator=Euler(0.1), lr=nothing, solver=EigenSolver(1e-6),
    maxiter=100,
    callback = (args...; kwargs...) -> nothing,
    copy=true, sample_nr=1000, parallel=false,
    verbosity=0, save_params=false, save_rng=false,
    save_sr=false,
    niter_start=1, history_old=nothing,
    discard_outliers=0.,
    transform_θ=x->x, transform_sr=(args...) -> args,
    kwargs...
    ) where {T}
    if lr !== nothing
        integrator = Euler(lr)
        @info "evolve: Warning lr is deprecated, use integrator=Euler(lr) instead"
    end

    if copy
        θ = deepcopy(θ)
    end

    if save_params
        history_params = Matrix{Float64}(undef, maxiter, length(θ))
    end

    if save_rng
        history_rng = Vector{Random.AbstractRNG}(undef, maxiter)
    end

    history = Matrix{Float64}(undef, maxiter, 7)
    if history_old !== nothing
        history[1:size(history_old, 1), :] = history_old
    end


    history_legend = Dict("energy" => 1, "var_energy" => 2, "sample_nr" => 3, "norm_grad" => 4, "norm_θ" => 5, "var_energy" => 6, "tdvp_error" => 7)
    misc = Dict()
    energy = 0.0

    if parallel
        Oks_and_Eks_ = generate_Oks_and_Eks_parallel(construct_mps, H; kwargs...)
    else
        Oks_and_Eks_ = (θ, sample_nr) -> Oks_and_Eks(θ, construct_mps, H, sample_nr; kwargs...)
    end

    for niter in niter_start:maxiter
        θ_old = θ
        θ, sr = integrator(θ, Oks_and_Eks_; sample_nr, solver, discard_outliers)

        # Transform sr
        sr, Oks_and_Eks_, solver, sample_nr = transform_sr(sr, Oks_and_Eks_, solver, sample_nr)

        # Compute energy
        energy = real(mean(sr.Es))
        var_energy = real(var(sr.Es))
        norm_grad = norm(get_θdot(sr; θtype=eltype(θ)))
        norm_θ = norm(θ_old)

        # Saving the energy and norms
        history[niter, :] .= energy, var_energy, length(sr), norm_grad, norm_θ, var(sr.Es), sr.tdvp_error
        if save_params
            history_params[niter, :] .= θ_old
        end
        if save_rng
            history_rng[niter] = copy(Random.default_rng())
        end

        # Update
        

        # Transform θ
        θ = transform_θ(θ)

        # Callback
        misc = Dict("energy" => energy, "niter" => niter, "history" => history[1:niter, :],
                    "history_legend" => history_legend)
        if save_params
            misc["history_params"] = history_params[1:niter, :]
        end
        if save_sr
            misc["sr"] = sr
        end
        
        stop = callback(; energy_value=energy, model=θ, misc=misc, niter=niter)
        
        if verbosity >= 2
            @info "iter $niter: $(sr.Es), ‖∇f‖ = $(norm_grad), ‖θ‖ = $(norm_θ), tdvp_error = $(sr.tdvp_error)"
            flush(stdout)
            flush(stderr)
        end

        if stop === :stop
            break
        end
    end

    return energy, θ, construct_mps(θ), misc
end

optimize = evolve