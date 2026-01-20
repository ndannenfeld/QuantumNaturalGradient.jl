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
    
    θtype = eltype(θ)
    h = integrator.lr
    if mode=="REAL" h *= im end

    if kwargs[:timer] !== nothing 
        natural_gradient = @timeit kwargs[:timer] "NaturalGradient" NaturalGradient(θ, Oks_and_Eks_; kwargs...) 
    else
        natural_gradient = NaturalGradient(θ, Oks_and_Eks_; kwargs...)
    end
    g = get_θdot(natural_gradient; θtype)
    if integrator.use_clipping
        clamp_and_norm!(g, integrator.clip_val, integrator.clip_norm)
    end
    θ .+= h .* g
    integrator.step += 1
    return θ, natural_gradient
end

mutable struct RK4 <: AbstractIntegrator
    lr::Float64
    step::Integer
    use_clipping::Bool
    clip_norm::Float64
    clip_val::Float64

    RK4(;lr=0.05, step=0, use_clipping=false, clip_norm=5.0, clip_val=1.0) = new(lr, step, use_clipping, clip_norm, clip_val)
end

function (integrator::RK4)(θ::ParameterTypes, Oks_and_Eks_::Function, mode::String="IMAG"; kwargs...)

    θtype = eltype(θ)
    h = integrator.lr
    if mode=="REAL" h *= im end

    if kwargs[:timer] !== nothing 
        ng1 = @timeit kwargs[:timer] "NaturalGradient" NaturalGradient(θ, Oks_and_Eks_; kwargs...) 
    else
        ng1 = NaturalGradient(θ, Oks_and_Eks_; kwargs...)
    end
    k1 = get_θdot(ng1; θtype)
    if integrator.use_clipping
        clamp_and_norm!(k1, integrator.clip_val, integrator.clip_norm)
    end

    # θ .+= h/6 * k1

    θ2 = deepcopy(θ)  
    θ2 .+= (h/2) .* k1
    if kwargs[:timer] !== nothing 
        ng2 = @timeit kwargs[:timer] "NaturalGradient" NaturalGradient(θ2, Oks_and_Eks_; kwargs...) 
    else
        ng2 = NaturalGradient(θ2, Oks_and_Eks_; kwargs...)
    end
    k2 = get_θdot(ng2; θtype)
    if integrator.use_clipping
        clamp_and_norm!(k2, integrator.clip_val, integrator.clip_norm)
    end

    # θ .+= h/6 * 2 * k2
    
    θ3 = deepcopy(θ)  
    θ3 .+= (h/2) .* k2
    if kwargs[:timer] !== nothing 
        ng3 = @timeit kwargs[:timer] "NaturalGradient" NaturalGradient(θ3, Oks_and_Eks_; kwargs...) 
    else
        ng3 = NaturalGradient(θ3, Oks_and_Eks_; kwargs...)
    end
    k3 = get_θdot(ng3; θtype)
    if integrator.use_clipping
        clamp_and_norm!(k3, integrator.clip_val, integrator.clip_norm)
    end
    
    # θ .+= h/6 * 2 * k3

    θ4 = deepcopy(θ)  
    θ4 .+= h .* k3
    if kwargs[:timer] !== nothing 
        ng4 = @timeit kwargs[:timer] "NaturalGradient" NaturalGradient(θ4, Oks_and_Eks_; kwargs...) 
    else
        ng4 = NaturalGradient(θ4, Oks_and_Eks_; kwargs...)
    end
    k4 = get_θdot(ng4; θtype)
    if integrator.use_clipping 
        clamp_and_norm!(k4, integrator.clip_val, integrator.clip_norm)
    end

    # θ .+= h/6 * k4

    # GC.gc()
    @. θ += (h/6) * (k1 + 2*k2 + 2*k3 + k4)

    integrator.step += 1

    return θ, ng1
end