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
    J = centered(ng.J)
    
    local Jt, Jtc
    if J isa Transpose
        Jt = transpose(J)
        Jtc = conj(Jt) # Does nothing if real
    end

    ns = nr_samples(ng.J)


    function S_times_v_(v)
        v1 = J * v
        v = zeros(eltype(v), length(v))
        BLAS.gemv!('C', 1.0/ns, J, v1, 0., v)
        # Instead of J' * (J * v) is much faster
        return v
    end

    function S_times_v_t(v)
        v1 = zeros(eltype(v), ns)
        v1 = BLAS.gemv!('T', 1.0, Jt, v, 0., v1)
        v = zeros(eltype(v), length(v))
        BLAS.gemv!('N', 1.0/ns, Jtc, v1, 0., v)
        return v
    end

    local S_times_v
    if J isa Transpose
        S_times_v = S_times_v_t
    else
        S_times_v = S_times_v_
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