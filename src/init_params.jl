function search_init_params(construct_mps, θ, H; gains= 10 .^ collect(range(-0.3, 1, 10)),
                                                 solver=EigenSolver(1e-4; save_info=true),
                                                 sample_nr=100, verbose=false, get_gain=false, kwargs...)
    gains = collect(gains)
    null_space_sizes = []

    for gain in gains
        sr = StochasticReconfiguration(θ .* gain, construct_mps, H; solver, sample_nr=sample_nr, kwargs...)
        push!(null_space_sizes, solver.info[:Nz])
        if verbose
            @info "Null space size: $(solver.info[:Nz]) - gain: $gain"
        end
    end
    arg = argmin(null_space_sizes)
    null_space_size = null_space_sizes[arg]
    gain = gains[arg]
    if verbose
        @info "Best Null space size: $null_space_size - $(round(null_space_size/length(θ)*100))%  - gain: $gain - worse null space size: $(maximum(null_space_sizes))"
    end
    
    if get_gain
        return θ .* gain, gain
    end
    
    return θ .* gain
end