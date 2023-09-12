mutable struct ReduceSolver <: AbstractCompositeSolver
    solver::AbstractSolver
    reduction_factor::Number
    reduction_method::Symbol
    ReduceSolver(solver::AbstractSolver, reduction_factor::Number=2, reduction_method::Symbol=:unitary_rrange) = new(solver, reduction_factor, reduction_method)
    #ReduceSolver(solver::AbstractSolver, reduced_size::Number=2, reduction_method::Symbol=:unitary_rrange) = new(solver, reduction_factor, reduction_method)
end

function (solver::ReduceSolver)(sr::NaturalGradient; kwargs...)
    Ekms = centered(sr.Es)
    
    sample_nr = length(Ekms)
    new_sample_nr = Integer(ceil(sample_nr / solver.reduction_factor))

    if solver.reduction_method === :unitary || solver.reduction_method === :unitary_rrange

        if solver.reduction_method === :unitary
            U = NDTensors.random_unitary(Float64, new_sample_nr, sample_nr)
        else
            U = RandomizedLinAlg.rrange(centered(sr.GT), new_sample_nr)'
        end
        
        GT = U * centered(sr.GT)
        GTd = GT * GT'
        
        θdot_raw = -solver(GTd, U * Ekms; kwargs...)
        sr.θdot = GT' * θdot_raw

    elseif solver.reduction_method === :sum

        reduction_factor = Integer(round(solver.reduction_factor))
        
        # Get a sample_nr_eff that is a multiple of reduction_factor
        sample_nr_eff = sample_nr ÷ reduction_factor * reduction_factor
        new_sample_nr = sample_nr_eff ÷ reduction_factor
        
        GT = centered(sr.GT)[1:sample_nr_eff, :]

        GT = reshape(GT[1:sample_nr_eff, :], new_sample_nr, reduction_factor, size(GT, 2))
        GT = mean(GT, dims=2)[:, 1, :]
        GTd = GT * GT'

        Ekms = reshape(Ekms[1:sample_nr_eff], new_sample_nr, reduction_factor)
        Ekms = mean(Ekms, dims=2)[:, 1]
        
        θdot_raw = -solver(GTd, Ekms; kwargs...)
        sr.θdot = GT' * θdot_raw
    else
        error("Unknown reduction method: $(solver.reduction_method) (should be :unitary_rrange, :unitary or :sum)")
    end

    tdvp_error!(sr)
    return sr.θdot
end