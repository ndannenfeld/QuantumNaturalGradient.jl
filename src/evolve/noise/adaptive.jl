abstract type AbstractCompositeTemperatureSchedule  <: AbstractTemperatureSchedule end

function Base.getproperty(schedule::AbstractCompositeTemperatureSchedule, property::Symbol)
    if property === :T
        return getproperty(schedule.schedule, property)
    else
        return getfield(schedule, property)
    end
end

function Base.setproperty!(schedule::AbstractCompositeTemperatureSchedule, property::Symbol, value)
    if property === :T
        return setproperty!(schedule.schedule, property, value)
    else
        return setfield!(schedule, property, value)
    end
end

mutable struct AdaptiveFindInitalTemperature <: AbstractCompositeTemperatureSchedule
    schedule::AbstractTemperatureSchedule
    number_of_steps::Integer
    factor::Real
    estimate_list::Vector{Float64}
    verbose::Bool
    function AdaptiveFindInitalTemperature(schedule::AbstractTemperatureSchedule, number_of_steps::Integer; factor::Real=1.1, verbose::Bool=false)
        return new(schedule, number_of_steps, factor, [], verbose)
    end
end

function (schedule::AdaptiveFindInitalTemperature)(integrator::LangevinNoise, θ::AbstractVector, noise_grad_ratio::Real)
    
    if schedule.number_of_steps <= length(schedule.estimate_list)
        schedule.schedule(integrator, θ, noise_grad_ratio)
    else
        push!(schedule.estimate_list, noise_grad_ratio * schedule.T / schedule.factor)
        schedule.T = mean(schedule.estimate_list)
        if schedule.number_of_steps == length(schedule.estimate_list) && schedule.verbose
            @info "Estimated initial temperature: $(schedule.T)"
        end
    end
end

mutable struct ZeroTemperatureSchedule <: AbstractCompositeTemperatureSchedule
    schedule::AbstractTemperatureSchedule
    number_of_steps::Integer
    function ZeroTemperatureSchedule(schedule::AbstractTemperatureSchedule, number_of_steps::Integer)
        return new(schedule, number_of_steps)
    end
end

function (schedule::ZeroTemperatureSchedule)(integrator::LangevinNoise, θ::AbstractVector, noise_grad_ratio::Real)
    if schedule.number_of_steps >= integrator.step
        schedule.schedule(integrator, θ, noise_grad_ratio)
    else
        schedule.T = 0
    end
end