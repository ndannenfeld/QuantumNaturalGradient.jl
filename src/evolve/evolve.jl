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
function (integrator::Euler)(θ::ParameterTypes, Oks_and_Eks_, mode::String="IMAG"; kwargs...)
    if kwargs[:timer] !== nothing 
        natural_gradient = @timeit kwargs[:timer] "NaturalGradient" NaturalGradient(θ, Oks_and_Eks_; kwargs...) 
    else
        natural_gradient = NaturalGradient(θ, Oks_and_Eks_; kwargs...)
    end
    g = get_θdot(natural_gradient; θtype=eltype(θ))
    if integrator.use_clipping
        clamp_and_norm!(g, integrator.clip_val, integrator.clip_norm)
    end
    if mode == "REAL"
        g .*= im # now, it's real-time evolution instead of imag-time evolution
    end
    θ .+= integrator.lr .* g
    integrator.step += 1
    return θ, natural_gradient
end

# Initialize and manage the optimization state
mutable struct OptimizationState
    Oks_and_Eks::Function
    callback::Function
    transform!::Function
    θ
    integrator::AbstractIntegrator
    solver::AbstractSolver
    energy::Real
    niter::Int
    history::DataFrame
    timer::TimerOutput
    discard_outliers
    sample_nr::Int
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
    transform! = (o, x) -> x,
    logger_funcs=[], 
    misc_restart=nothing, 
    timer=TimerOutput(), discard_outliers=0, maxiter=50, gradtol=1e-10, verbosity=0) where {T}
    
    energy(; energy) = energy
    var_energy(; var_energy) = var_energy
    norm_natgrad(; norm_natgrad) = norm_natgrad
    norm_θ(; norm_θ) = norm_θ
    niter(; niter) = niter
    logger_funcs = [niter, energy, var_energy, norm_natgrad, norm_θ, logger_funcs...]
    history = Observer(logger_funcs...)
    state = OptimizationState(
        Oks_and_Eks, callback, transform!, θ, integrator, solver,
        1e10, 1, history, timer, discard_outliers, sample_nr, maxiter, gradtol, verbosity
    )
    
    if misc_restart !== nothing
        set_misc(state, misc_restart)
        fixbrokenObserver!(state.history, logger_funcs)
        state.niter += 1
    end

    return state
end

function fixbrokenObserver!(history, logger_funcs)
    for f in logger_funcs
        if !(Observers.get_function(history, string(f)) isa Function)
            Observers.set_function!(history, f)
        end
    end
    return history
end

function set_misc(state::OptimizationState, misc)

    if misc isa DataFrame # This is for backward compatibility
        @warn "The misc dictionary is a DataFrame. This is deprecated. Please use a dictionary instead."
        state.history = misc
        enes = state.history[!, "energy"]
        state.niter = length(enes)
        state.energy = enes[end]
        println("Restarting from niter = $(state.niter)")
        return nothing
    end

    if haskey(misc, "history")
        @assert state.history isa DataFrame "The history is not a DataFrame."
        state.history = misc["history"]
    else
        error("The misc dictionary does not have a history key")
    end
    
    if haskey(misc, "niter")
        state.niter = misc["niter"]
        println("Restarting from niter = $(state.niter)")
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

# as of now, this wrapper has not been adapted to call evolve(Oks_and_Eks_, theta) with Real Time Evolution, i.e. the call will default to imaginary time evolution and there is no argument to choose the mode.
function evolve(construct_mps, θ::T, H; integrator=Euler(0.1), maxiter=10, callback=(args...; kwargs...) -> nothing, 
    logger_funcs=[], copy=true, misc_restart=nothing, timer=TimerOutput(), kwargs...) where {T}
    
    Oks_and_Eks_ = (θ, sample_nr) -> Oks_and_Eks(θ, construct_mps, H, sample_nr; kwargs...)
    energy, θ, misc = evolve(Oks_and_Eks_, θ; integrator, maxiter, callback, logger_funcs, copy, misc_restart, timer)
    return energy, θ, construct_mps(θ), misc
end

function evolve(Oks_and_Eks_, θ::T, mode::String="IMAG"; 
    integrator=Euler(0.1),
    solver=EigenSolver(1e-6),
    maxiter=10, 
    sample_nr=1000,
    callback=(args...; kwargs...) -> nothing, 
    transform! = (o, x) -> x,
    verbosity=0,
    logger_funcs=[], 
    copy=false, 
    misc_restart=nothing, 
    discard_outliers=0.,
    timer=TimerOutput(), gradtol=1e-10) where {T}

    if mode != "IMAG" && mode != "REAL"
        @warn "QuantumNaturalGradient.evolve: Unknown value for mode argument, defaulting to \"IMAG\", i.e. imaginary-time evolution. For real-time evolution choose \"REAL\"."
        flush(stdout); flush(stderr)
        mode = "IMAG"
    end

    if verbosity >= 1
        @info "Evolving in $(mode == "REAL" ? "real" : "imaginary") time"
        flush(stdout); flush(stderr)
    end

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
        transform!,
        logger_funcs, 
        misc_restart,
        timer=timer, discard_outliers,
        maxiter, gradtol, verbosity
    )

    return evolve!(state, mode)
end

function evolve!(state::OptimizationState, mode::String="IMAG")
    # Main optimization loop
    dynamic_kwargs = Dict()
    while step!(state, dynamic_kwargs, mode)
    end

    if state.verbosity >= 2
        @info "QuantumNaturalGradient.evolve: Done"
        show(state.timer)
        flush(stdout); flush(stderr)
    end

    # Collect the results
    return state.energy, state.θ, get_misc(state)
end

function step!(o::OptimizationState, dynamic_kwargs, mode::String="IMAG")

    if o.niter > o.maxiter
        if o.verbosity >= 1
            @info "$(typeof(o.integrator)): Maximum number of iterations completed ($(o.niter-1))"
            flush(stdout); flush(stderr)
        end
        return false
    end

    o.θ, natural_gradient = @timeit o.timer "integrator" o.integrator(o.θ, o.Oks_and_Eks, mode; sample_nr=o.sample_nr, solver=o.solver, discard_outliers=o.discard_outliers, timer=o.timer, dynamic_kwargs...)
    
    # Transform the optimization state and natural_gradient
    natural_gradient = o.transform!(o, natural_gradient)
    
    o.energy = real(mean(natural_gradient.Es))
    var_energy = real(var(natural_gradient.Es))
    norm_natgrad = norm(get_θdot(natural_gradient; θtype=eltype(o.θ)))
    norm_θ = norm(o.θ)

    # Saving the energy and other variables
    Observers.update!(o.history; natural_gradient, θ=o.θ, niter=o.niter, energy=o.energy, var_energy, norm_natgrad, norm_θ, natural_gradient.saved_properties...)

    stop = o.callback(; energy_value=o.energy, model=o.θ, misc=get_misc(o), niter=o.niter)

    if o.verbosity >= 2
        @info "iter $(o.niter): $(natural_gradient.Es), ‖θdot‖ = $(norm_natgrad), ‖θ‖ = $(norm_θ), tdvp_error = $(natural_gradient.tdvp_error)"
        flush(stdout); flush(stderr)
    end
    
    if stop === false
        @info "QuantumNaturalGradient.evovle: Ending evolution because callback function has returned false."
        flush(stdout); flush(stderr)
        return false
    end

    if norm_natgrad < o.gradtol
        if o.verbosity >= 1
            @info "$(typeof(o.integrator)): Gradient tolerance reached"
            flush(stdout); flush(stderr)
        end
        return false
    end

    o.niter += 1
    return true
end

include("heun.jl")
include("decay.jl")