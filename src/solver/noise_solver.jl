mutable struct NoiseSolver <: AbstractCompositeSolver
    solver::AbstractSolver
    σ::Real
end

function (solver::NoiseSolver)(sr::NaturalGradient; method=:auto, kwargs...)
    error("Not implemented properly.") # The samples should be drawn completely randomly for this to work. Also normalization from the Es and Ok should be removed.
    Es_noisy = sr.Es.data + randn(length(sr.Es)) * solver.σ
    Es_noisy = EnergySummary(Es_noisy)

    if method === :T || (method === :auto && size(sr.GT, 1) < size(sr.GT, 2))
        sr.θdot = solve_T(solver, sr.GT, Es_noisy; kwargs...)
    else
        Ekms = centered(Es_noisy)
        grad_half = GT.data' * Ekms ./ length(Es)
        sr.θdot = solve_S(solver, sr.GT, grad_half; kwargs...)
    end
    
    tdvp_error!(sr)
    return sr
end