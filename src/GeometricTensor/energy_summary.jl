struct EnergySummary{T <: Number}
    data::Vector{T}
    mean::T
    var::Float64
    std_of_var::Float64
end

EnergySummary(ψ::MPS, H::MPO; sample_nr=1000) = EnergySummary([Ek(ψ, H) for _ in 1:sample_nr])

function EnergySummary(Eks::Vector{Complex{Float64}}; importance_weights=nothing)
    if any(imag.(Eks) .> 1e-10)
        mean_ = wmean(Eks; weights=importance_weights)
        Eks_c = Eks .- mean_
        var_ = wvar(Eks_c; weights=importance_weights)
        std_of_var = wstd(Eks_c * conj(Eks_c); weights=importance_weights)
        return EnergySummary(Eks_c .* sqrt.(importance_weights), mean_, var_, std_of_var)
    end
    return EnergySummary(real.(Eks), importance_weights)
end

function EnergySummary(Eks::Vector{Float64}; importance_weights=nothing)
    local mean_, std_of_var
    mean_, var_ = wmean_and_var(Eks; weights=importance_weights)
    Eks_c = real.(Eks .- mean_)
    std_of_var = wstd(Eks_c .^ 2; weights=importance_weights)

    return EnergySummary(Eks_c .* sqrt.(importance_weights), mean_, var_, std_of_var)
end

Statistics.mean(Es::EnergySummary) = Es.mean
Statistics.var(Es::EnergySummary) = Es.var
Statistics.std(Es::EnergySummary) = sqrt(Es.var)
Base.length(Es::EnergySummary) = length(Es.data)

energy_error(Es::EnergySummary) = std(Es) / sqrt(length(Es))
energy_var_error(Es::EnergySummary) = Es.std_of_var / sqrt(length(Es))

centered(Es::EnergySummary) = Es.data

function Base.show(io::IO, Es::EnergySummary)
    error = energy_error(Es)
    digits = Int(min(ceil(-log10(error)), 10)) + 1
    E_str = "E = $(round(real(Es.mean), digits=digits)) ± $(round(error, digits=digits))"

    error2 = energy_var_error(Es)
    digits = Int(min(ceil(-log10(error2)), 10)) + 1
    Evar_str = "var(E) = $(round(Es.var, digits=digits)) ± $(round(error2, digits=digits))"

    print(io, "EnergySummary($E_str, $Evar_str, Nₛ=$(length(Es)))")
end