abstract type AbstractSolver end

function solve_S(solver::AbstractSolver, GT::SparseGeometricTensor, Es::EnergySummary, grad_half::Vector; kwargs...)
    GTd = dense_S(GT)
    θdot = -solver(GTd, grad_half; kwargs...)
    
    return θdot
end

function solve_T(solver::AbstractSolver, GT::SparseGeometricTensor, Es::EnergySummary; kwargs...)
    GTd = dense_T(GT)

    Ekms = centered(Es)
    #println("Ekms = ", Ekms)
    θdot_raw = -solver(GTd, Ekms; kwargs...)
    θdot = GT.data' * θdot_raw

    return θdot
end

function (solver::AbstractSolver)(sr::StochasticReconfiguration; method=:auto, kwargs...)
    #sample_nr = 
    if method === :T || (method === :auto && size(sr.GT, 1) < size(sr.GT, 2))
        sr.θdot = solve_T(solver, sr.GT, sr.Es; kwargs...)
    else
        sr.θdot = solve_S(solver, sr.GT, sr.Es, sr.grad ./ 2; kwargs...)
    end
    
    tdvp_error!(sr)
    return sr
end

include("eigen_solver.jl")
include("reduce_solver.jl")