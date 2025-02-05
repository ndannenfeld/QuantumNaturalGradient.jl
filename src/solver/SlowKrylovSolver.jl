using KrylovKit
mutable struct SlowKrylovSolver <: AbstractSolver
    diagshift::Float64
    tol::Float64
    krylovdim::Int64
    verbose::Bool
    save_info::Bool
    info
    SlowKrylovSolver(;diagshift::Float64=1e-5, tol::Float64=1e-5, krylovdim::Int64=200, verbose=false, save_info=false) = new(diagshift, tol, krylovdim, verbose, save_info, nothing)
end

function (solver::SlowKrylovSolver)(ng::NaturalGradient; method=:auto, kwargs...)
    GT = centered(ng.GT)
    np = nr_parameters(ng.GT)
    function S_times_v(v)
        v1 = GT * v
        v = zeros(eltype(v), length(v))
        BLAS.gemv!('C', 1.0/ns, GT, v1, 0.0, v)
        # Instead of GT' * (GT * v) is much faster
        return v
    end
    grad_half = get_gradient(ng) ./ 2
    
    if solver.tol > 0
        θdot, info = linsolve(S_times_v, grad_half, solver.diagshift; tol=solver.tol, isposdef=true, krylovdim=solver.krylovdim)
    else
        θdot, info = linsolve(S_times_v, grad_half, solver.diagshift; isposdef=true, krylovdim=solver.krylovdim)
    end
    
    if solver.save_info
        solver.info = info
    end
    
    ng.θdot = -θdot
    tdvp_error!(ng)
    
    return ng
end