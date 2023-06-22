mutable struct EigenSolver <: AbstractSolver
    revcut::Float64
    abscut::Float64
    verbose::Bool
    save_info::Bool
    info
    EigenSolver(revcut::Float64=1e-5, abscut::Float64=0.; verbose=false, save_info=false) = new(revcut, abscut, verbose, save_info, nothing)
end


function (solver::EigenSolver)(M::Matrix, v::Vector)
    eig = eigen(Hermitian(M))
    max_val = eig.values[end]
    cond = eig.values ./ max_val
    condition_number = std(log.(cond))

    Nz = sum(cond .< solver.revcut)

    if solver.save_info
        solver.info = Dict(:eig => eig, :Nz => Nz, :condition_number => condition_number)
    end

    if solver.verbose
        p = Nz / length(cond) * 100
        #println(cond)
        if Nz > 0
            @info "EigenSolver: Zero eigenvalues: $Nz - $(round(p, digits=1))%  - cn: $condition_number"
        else
            @info "EigenSolver: No zero eigenvalues - cn: $condition_number"
        end
        flush(stdout)
        flush(stderr)
    end
    
    soft_cond = @. ifelse(cond < 1e-13, 0, (max_val*solver.revcut + solver.abscut) / abs(eig.values)) ^ 6
    inv_eigen = @. ifelse(cond < 1e-13, 0, 1 / (eig.values * (1 + soft_cond)))

    #inv_eigen = ifelse.(cond .< solver.revcut, 0, 1 ./ eig.values)

    gt = eig.vectors' * v .* inv_eigen
    
    return eig.vectors * gt
end