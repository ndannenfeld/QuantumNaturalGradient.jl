mutable struct EigenSolverAutocut <: AbstractSolver
    validation_ratio::Float64
    revcut::Float64
    abscut::Float64
    verbose::Bool
    save_info::Bool
    info
    EigenSolverAutocut(validation_ratio::Float64=0.5, revcut::Float64=1e-12, abscut::Float64=0.; verbose=false, save_info=false) = new(validation_ratio, revcut, abscut, verbose, save_info, nothing)
end

function (solver::EigenSolverAutocut)(sr::NaturalGradient; method=:auto, kwargs...)
    if method === :T || (method === :auto && size(sr.GT, 1) < size(sr.GT, 2))
        sr.θdot = solve_T(solver, sr.GT, sr.Es; kwargs...)
    else
        sr.θdot = solve_S(solver, sr.GT, get_gradient(sr) ./ 2; kwargs...)
    end
    
    tdvp_error!(sr)
    return sr
end

function solve_S(solver::EigenSolverAutocut, GT::SparseGeometricTensor, grad_half::Vector; kwargs...)
    error("Not implemented")
end

function solve_T(solver::EigenSolverAutocut, GT::SparseGeometricTensor, Es::EnergySummary; kwargs...)
    Ekms = centered(Es)

    θdot_raw = -solver(centered(GT), Ekms; kwargs...)
    θdot = centered(GT)' * θdot_raw

    return θdot
end

function (solver::EigenSolverAutocut)(O::AbstractMatrix, v::AbstractArray)
    #@assert ishermitian(M) "EigenSolver: M is not Hermitian"
    
    validation_size = round(Int, size(O, 1) * solver.validation_ratio)
    train_size = size(O, 1) - validation_size

    O_t = O[1:train_size, :]
    O_val = O[train_size+1:end, :]

    v_t = v[1:train_size]
    v_val = v[train_size+1:end]

    M_t = O_t * O_t'
    eig = eigen(Hermitian(M_t))
    max_val = eig.values[end]
    cond_t = eig.values ./ max_val
    vt = eig.vectors' * v_t

    function error_func(revcut)
        soft_cond = @. ifelse(cond_t < 1e-13, 0, (max_val*revcut + solver.abscut) / abs(eig.values)) ^ 6
        inv_eigen = @. ifelse(cond_t < 1e-13, 0, 1 / (eig.values * (1 + soft_cond)))

        gt = vt .* inv_eigen
        o = eig.vectors * gt
        #@show size(o)
        θdot = O_t' * o

        error = norm(O_val * θdot - v_val)
        return error
    end
    cond_t = abs.(cond_t)
    min_cut  = round(Int, minimum(log10.(cond_t)))
    r = 10. .^ (-1. * collect(1:min_cut))
    cuts = vcat(cond_t, r)
    cuts = sort(cuts)
    # get 15 point in cond_t distributed evenly
    cuts = cuts[1:length(cuts)÷15:end]
    errors = [error_func(revcut) for revcut in cuts]

    perfect_revcut = cuts[argmin(errors)]
    M = O * O'
    eig = eigen(Hermitian(M))
    max_val = eig.values[end]
    cond = eig.values ./ max_val
    soft_cond = @. ifelse(cond < 1e-13, 0, (max_val*perfect_revcut + solver.abscut) / abs(eig.values)) ^ 6
    inv_eigen = @. ifelse(cond < 1e-13, 0, 1 / (eig.values * (1 + soft_cond)))
    gt = eig.vectors' * v .* inv_eigen
    o = eig.vectors * gt

    # Logging
    Nz = sum(cond .< perfect_revcut)
    condition_number = std(log.(abs.(cond[Nz+1:end])))
    if solver.verbose
        
        p = Nz / length(cond) * 100
        if Nz > 0
            @info "EigenSolver: Null space size: $Nz - $(round(p, digits=1))%  - cn: $(round_auto(condition_number)) - max_val: $(round_auto(max_val)) - cut: $(round_auto(perfect_revcut))"
        else
            @info "EigenSolver: No null space - cn: $(round_auto(condition_number)) - max_val: $(round_auto(max_val)) - cut: $(round_auto(perfect_revcut))"
        end
        flush(stdout)
        flush(stderr)
    end

    if solver.save_info
        solver.info = Dict(:eig => eig, :Nz => Nz, :condition_number => condition_number, :errors => errors, :cuts=>cuts)
    end

    return o
end

