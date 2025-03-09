abstract type AbstractSolver end

function solve_S(solver::AbstractSolver, J::Jacobian, grad_half::Vector; timer=TimerOutput(), kwargs...)
    @timeit "dense_S" Jd = dense_S(J)
    @timeit "solve" θdot = -solver(Jd, grad_half; kwargs...)
    
    return θdot
end

function solve_T(solver::AbstractSolver, J::Jacobian, Es::EnergySummary; timer=TimerOutput(), kwargs...)
    @timeit "dense_T" Jd = dense_T(J)
    Ekms = centered(Es)

    @timeit "solve" θdot_raw = -solver(Jd, Ekms; kwargs...)
    θdot = centered(J)' * θdot_raw

    return θdot
end

function (solver::AbstractSolver)(ng::NaturalGradient; method=:auto, compute_error=true, kwargs...)
    if method === :T || (method === :auto && nr_samples(ng.J) < nr_parameters(ng.J))
        ng.θdot = solve_T(solver, ng.J, ng.Es; kwargs...)
    else
        ng.θdot = solve_S(solver, ng.J, get_gradient(ng) ./ 2; kwargs...)
    end
    if compute_error
        tdvp_error!(ng)
    end
    return ng
end


function (solver::AbstractSolver)(M::AbstractMatrix, v::AbstractArray, double::Bool; method=:auto, kwargs...)
    if double
        return solver(M, v; kwargs...)
    end
    
    if method === :T || (method === :auto && size(M, 1) < size(M, 2))
        return M' * solver(M * M', v; kwargs...)
    else
        return solver(M' * M, M' * v; kwargs...)
    end
end

abstract type AbstractCompositeSolver <: AbstractSolver end

function (solver::AbstractCompositeSolver)(M::Matrix, v::Vector)
    return solver.solver(M, v)
end


include("eigen_solver.jl")
include("reduce_solver.jl")
include("eigen_solver_autocut.jl")
include("LinearSolveWrapper.jl")
include("SlowKrylovSolver.jl")