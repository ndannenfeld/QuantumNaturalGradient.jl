function find_outliers(Eks; cut=1.0)
    order = sortperm(Eks)
    s_Eks = Eks[order]
    #magnitude = mean(abs.(s_Eks[2:end-1]))
    magnitude = mean(abs.(s_Eks))

    cutter = diff(Eks[order]) ./ magnitude
    cutter = cutter .> cut
    cut_small = argmin(cutter) - 1
    cut_big = length(cutter) - argmin(cutter[end:-1:1]) + 3
    return sort(vcat(order[1:cut_small], order[cut_big:end]))
end

function bootrap_energy(Eks; importance_weights=ones(length(Eks)))
    Eks2 = Eks .^ 2
    wm = sum(importance_weights)
    Em = sum(Eks .* importance_weights)
    Ev = sum(Eks2 .* importance_weights)
    
    E_boot = (Em .- Eks .* importance_weights) ./ (wm .- importance_weights)
    Ev_boot = (Ev .- Eks2 .* importance_weights) ./ (wm .- importance_weights)
    return E_boot, Ev_boot .- E_boot.^2
end


function find_outliers_bootstrap(Eks; cut=0.1, importance_weights=ones(length(Eks)))
    E_boot, Ev_boot = bootrap_energy(Eks; importance_weights)
    error1 = abs.(Ev_boot .- mean(Ev_boot)) ./ abs.(Ev_boot)
    error2 = abs.(E_boot .- mean(E_boot)) ./ abs.(E_boot)
    error = max.(error1, error2)
    return findall(error .> cut)
end

"""
    remove_outliers!(Eks, args...; importance_weights=nothing, cut=1.0, verbose=false)
    find outliers in Eks and remove them from Eks and all other args.
"""
function remove_outliers!(Eks::AbstractVector, args...; importance_weights=ones(length(Eks)), cut=0.005, verbose=false, find_outliers_=find_outliers_bootstrap)
    remove = find_outliers_(Eks; cut, importance_weights)
    
    local args_new
    if length(remove) > 0
        if verbose
            println("Removing ", length(remove), " outliers")
        end
        Eks = deleteat_(Eks, remove)
        args_new = Any[Eks]
        for arg in args
            arg = deleteat_(arg, remove)
            push!(args_new, arg)
        end
    else
        args_new = Any[Eks, args...]
    end
    return args_new
end

remove_outliers(Eks::AbstractVector, args...; kwargs...) = remove_outliers!(copy(Eks), deepcopy(args)...; kwargs...)

function deleteat_(m::AbstractMatrix, v::Vector{<:Integer})
    new_o = setdiff(1:size(m, 1), v)
    return m[new_o, :]
end

function deleteat_(m::AbstractVector, v::Vector{<:Integer})
    m = copy(m)
    deleteat!(m, v)
    return m
end