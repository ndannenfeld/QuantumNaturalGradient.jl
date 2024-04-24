abstract type AbstractSolver end

function solve_S(solver::AbstractSolver, GT::SparseGeometricTensor, grad_half::Vector; kwargs...)
    GTd = dense_S(GT)
    θdot = -solver(GTd, grad_half; kwargs...)
    
    return θdot
end

function solve_T(solver::AbstractSolver, GT::SparseGeometricTensor, Es::EnergySummary; kwargs...)
    GTd = dense_T(GT)
    Ekms = centered(Es)

    θdot_raw = -solver(GTd, Ekms; kwargs...)
    θdot = centered(GT)' * θdot_raw

    return θdot
end

function (solver::AbstractSolver)(sr::NaturalGradient; method=:auto, kwargs...)
    if method === :T || (method === :auto && size(sr.GT, 1) < size(sr.GT, 2))
        sr.θdot = solve_T(solver, sr.GT, sr.Es; kwargs...)
    else
        sr.θdot = solve_S(solver, sr.GT, get_gradient(sr) ./ 2; kwargs...)
    end
    
    tdvp_error!(sr)
    return sr
end


function (solver::AbstractSolver)(M::AbstractMatrix, v::AbstractArray, double::Bool; method=:auto, kwargs...)
    if double
        return solver(M, v)
    end
    
    if method === :T || (method === :auto && size(M, 1) < size(M, 2))
        return M' * solver(M * M', v)
    else
        return solver(M' * M, M' * v)
    end
end

abstract type AbstractCompositeSolver <: AbstractSolver end

function (solver::AbstractCompositeSolver)(M::Matrix, v::Vector)
    return solver.solver(M, v)
end


include("eigen_solver.jl")
include("reduce_solver.jl")
include("eigen_solver_autocut.jl")