function find_outliers(Eks; cut=1.0)
    order = sortperm(Eks)
    s_Eks = Eks[order]
    magnitude = mean(abs.(s_Eks[2:end-1]))
    magnitude = mean(abs.(s_Eks))

    cutter = diff(Eks[order]) ./ magnitude
    cutter = cutter .> cut
    cut_small = argmin(cutter) - 1
    cut_big = length(cutter) - argmin(cutter[end:-1:1]) + 3
    return sort(vcat(order[1:cut_small], order[cut_big:end]))
end

function remove_outliers!(Eks::AbstractVector, args...; importance_weights=nothing, cut=1.0, verbose=false)
    local remove
    if importance_weights !== nothing
        remove = find_outliers(Eks .* importance_weights; cut)
    else
        remove = find_outliers(Eks; cut)
    end

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

function deleteat_(m::AbstractMatrix, v::Vector{<:Integer})
    new_o = setdiff(1:size(m, 1), v)
    return m[new_o, :]
end

function deleteat_(m::AbstractVector, v::Vector{<:Integer})
    m = copy(m)
    deleteat!(m, v)
    return m
end