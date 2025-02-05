mutable struct NoiseSolver <: AbstractCompositeSolver
    solver::AbstractSolver
    σ::Real
end

function (solver::NoiseSolver)(sr::NaturalGradient; method=:auto, kwargs...)
    error("Not implemented properly.") # The samples should be drawn completely randomly for this to work. Also normalization from the Es and Ok should be removed.
    Es_noisy = sr.Es.data + randn(length(sr.Es)) * solver.σ
    Es_noisy = EnergySummary(Es_noisy)

    if method === :T || (method === :auto && nr_samples(J) < nr_parameters(J))
        sr.θdot = solve_T(solver, sr.J, Es_noisy; kwargs...)
    else
        Ekms = centered(Es_noisy)
        grad_half = centered(J)' * Ekms ./ length(Es)
        sr.θdot = solve_S(solver, sr.J, grad_half; kwargs...)
    end
    
    tdvp_error!(sr)
    return sr
end