# Abstract types and common utilities
abstract type AbstractIntegrator end
abstract type AbstractCompositeIntegrator <: AbstractIntegrator end
abstract type AbstractSchedule end

function Base.getproperty(integrator::AbstractCompositeIntegrator, property::Symbol)
    if property === :lr || property === :step
        return getproperty(integrator.integrator, property)
    else
        return getfield(integrator, property)
    end
end

function clamp_and_norm!(gradients, clip_val, clip_norm)
    clamp!(gradients, -clip_val, clip_val)
    norm(gradients) > clip_norm ? gradients .= (gradients ./ norm(gradients)) .* clip_norm : nothing
    return gradients
end

# Euler integrator structure
mutable struct Euler <: AbstractIntegrator
    lr::Float64
    step::Integer
    use_clipping::Bool
    clip_norm::Float64
    clip_val::Float64
    Euler(;lr=0.05, step=0, use_clipping=false, clip_norm=5.0, clip_val=1.0) = new(lr, step, use_clipping, clip_norm, clip_val)
end

# Euler integrator step function
function (integrator::Euler)(θ::AbstractVector, Oks_and_Eks_; kwargs...)
    if kwargs[:timer] !== nothing 
        ng = @timeit kwargs[:timer] "NaturalGradient" NaturalGradient(θ, Oks_and_Eks_; kwargs...) 
    else
        ng = NaturalGradient(θ, Oks_and_Eks_; kwargs...)
    end
    g = get_θdot(ng; θtype=eltype(θ))
    if integrator.use_clipping
        clamp_and_norm!(g, integrator.clip_val, integrator.clip_norm)
    end
    θ = θ .+ integrator.lr .* g
    integrator.step += 1
    return θ, ng
end

# Initialize and manage the optimization state
mutable struct OptimizationState
    Oks_and_Eks::Function
    callback::Function
    θ
    integrator::AbstractIntegrator
    solver
    energy::Real
    gradient
    niter::Int
    history::DataFrame
    timer::TimerOutput
    discard_outliers
    transform
    sample_nr
    maxiter::Int
    gradtol::Real
    verbosity::Int
end

get_misc(state::OptimizationState) = Dict("energy" => state.energy, "niter" => state.niter, "history" => state.history, "rng" => get_rng())

function get_rng()
    t = current_task()
    return [t.rngState0, t.rngState1, t.rngState2, t.rngState3]
end

function OptimizationState(Oks_and_Eks, θ::T, integrator; 
    sample_nr=1000, 
    solver=EigenSolver(1e-6),
    callback=(args...; kwargs...) -> nothing, 
    logger_funcs=[], 
    misc_restart=nothing, 
    timer=TimerOutput(), discard_outliers=0, transform=(args...) -> args, maxiter=50, gradtol=1e-10, verbosity=0) where {T}
    
    energy(; energy) = energy
    norm_grad(; norm_grad) = norm_grad
    norm_θ(; norm_θ) = norm_θ
    niter(; niter) = niter

    history = Observer(niter, energy, norm_grad, norm_θ, logger_funcs...)
    state = OptimizationState(
        Oks_and_Eks, callback, θ, integrator, solver,
        1e10, nothing, 1, history, timer, discard_outliers, transform, sample_nr, maxiter, gradtol, verbosity
    )
    
    if misc_restart !== nothing
        set_misc(state, misc_restart)
        #state.niter += 1
    end

    return state
end

function set_misc(state::OptimizationState, misc)
    if haskey(misc, "history")
        @assert state.history isa DataFrame "The history is not a DataFrame."
        state.history = misc["history"]
    else
        error("The misc dictionary does not have a history key")
    end
    
    if haskey(misc, "niter")
        state.niter = misc["niter"]
    else
        error("The misc dictionary does not have a niter key")
    end
    
    if haskey(misc, "energy")
        state.energy = misc["energy"]
    end

    if haskey(misc, "rng")
        set_rng(misc["rng"])
    end
end

function set_rng(r)
    t = current_task()
    t.rngState0 = r[1]
    t.rngState1 = r[2]
    t.rngState2 = r[3]
    t.rngState3 = r[4]
end

function evolve(construct_mps, θ::T, H; integrator=Euler(0.1), maxiter=10, callback=(args...; kwargs...) -> nothing, 
    logger_funcs=[], copy=true, misc_restart=nothing, timer=TimerOutput(), kwargs...) where {T}
    
    Oks_and_Eks_ = (θ, sample_nr) -> Oks_and_Eks(θ, construct_mps, H, sample_nr; kwargs...)
    energy, θ, misc = evolve(Oks_and_Eks_, θ; integrator, maxiter, callback, logger_funcs, copy, misc_restart, timer)
    return energy, θ, construct_mps(θ), misc
end

function evolve(Oks_and_Eks_, θ::T; 
    integrator=Euler(0.1),
    solver=EigenSolver(1e-6),
    maxiter=10, 
    sample_nr=1000,
    callback=(args...; kwargs...) -> nothing, 
    verbosity=0,
    logger_funcs=[], 
    copy=true, 
    misc_restart=nothing, 
    discard_outliers=0.,
    transform=(args...) -> args,
    timer=TimerOutput(), gradtol=1e-10) where {T}

    if copy
        θ = deepcopy(θ)
    end

    # Prepare the optimization state    
    state = OptimizationState(
        Oks_and_Eks_, 
        θ, 
        integrator; 
        sample_nr,
        solver,
        callback, 
        logger_funcs, 
        misc_restart,
        timer=timer, discard_outliers,
        transform, maxiter, gradtol, verbosity
    )

    return evolve!(state)
end

function evolve!(state::OptimizationState)
    # Main optimization loop
    dynamic_kwargs = Dict()
    while step!(state, dynamic_kwargs)
    end

    if state.verbosity >= 2
        @info "evolve: Done"
        show(state.timer)
    end

    # Collect the results
    return state.energy, state.θ, get_misc(state)
end

function step!(o::OptimizationState, dynamic_kwargs)

    if o.niter-1 >= o.maxiter
        if o.verbosity >= 1
            @info "$(typeof(o.integrator)): Maximum number of iterations reached ($(o.niter))"
        end
        return false
    end

    θ, o.gradient = @timeit o.timer "integrator" o.integrator(o.θ, o.Oks_and_Eks; sample_nr=o.sample_nr, solver=o.solver, discard_outliers=o.discard_outliers, timer=o.timer, dynamic_kwargs...)
    
    # Transform θ and ng
    θ, o.gradient, o.Oks_and_Eks, o.solver, o.sample_nr = o.transform(θ, o.gradient, o.Oks_and_Eks, o.solver, o.sample_nr)

    dict_save = Dict()
    
    o.energy = real(mean(o.gradient.Es))
    
    norm_grad = norm(get_θdot(o.gradient; θtype=eltype(θ)))

    norm_θ = norm(θ)

    # Saving the energy and other variables
    dict_save[:θ_old] = o.θ

    Observers.update!(o.history; gradient=o.gradient, θ=θ, niter=o.niter, energy=o.energy, norm_grad, norm_θ, dict_save..., o.gradient.saved_properties...)


    o.θ = θ

    stop = o.callback(; energy_value=o.energy, model=o.θ, misc=o.history, niter=o.niter)

    if o.verbosity >= 2
        @info "iter $(o.niter): $(o.gradient.Es), ‖∇f‖ = $(norm_grad), ‖θ‖ = $(norm_θ), tdvp_error = $(o.gradient.tdvp_error)"
        flush(stdout)
        flush(stderr)
    end
    
    if stop === false
        return false
    end

    if norm_grad < o.gradtol
        if o.verbosity >= 1
            @info "$(typeof(o.integrator)): Gradient tolerance reached"
        end
        return false
    end

    o.niter += 1
    return true
end