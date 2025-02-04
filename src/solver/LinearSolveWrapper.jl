using LinearSolve

mutable struct LinearSolveWrapper <: QuantumNaturalGradient.AbstractSolver
    diagshift::Float64
    alg
    verbose::Bool
    save_info::Bool
    info
    LinearSolveWrapper(diagshift::Float64=1e-5, alg=KrylovJL_CG(); verbose=false, save_info=false) = new(diagshift, alg, verbose, save_info, nothing)
end


function (solver::LinearSolveWrapper)(M::AbstractMatrix, v::AbstractArray)
    #@assert ishermitian(M) "LinearSolveWrapper: M is not Hermitian"
    if solver.diagshift != 0
        M = Hermitian(M + I(size(M, 1)) * solver.diagshift)
    else
        M = Hermitian(M)
    end
    
    prob = LinearProblem(M, v)
    sol = solve(prob, solver.alg)
    if solver.save_info
        solver.info = sol
    end
    return sol.u
end