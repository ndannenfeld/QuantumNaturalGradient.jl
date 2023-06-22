struct EnergySummary
    data::Vector{Float64}
    mean::Float64
    var::Float64
    std_of_var::Float64
end

EnergySummary(ψ::MPS, H::MPO; sample_nr=1000) = EnergySummary([Ek(ψ, H) for _ in 1:sample_nr])

function EnergySummary(Eks::Vector{Float64})
    mean_ = mean(Eks)
    Eks_c = Eks .- mean_
    std_of_var = std(Eks_c .^ 2)
    return EnergySummary(Eks, mean_, var(Eks), std_of_var)
end

Statistics.mean(Es::EnergySummary) = Es.mean
Statistics.var(Es::EnergySummary) = Es.var
Statistics.std(Es::EnergySummary) = sqrt(Es.var)
Base.length(Es::EnergySummary) = length(Es.data)

energy_error(Es::EnergySummary) = std(Es) / sqrt(length(Es))
energy_var_error(Es::EnergySummary) = Es.std_of_var / sqrt(length(Es))

centered(Es::EnergySummary) = Es.data .- mean(Es)

function Base.show(io::IO, Es::EnergySummary)
    error = energy_error(Es)
    digits = Int(min(ceil(-log10(error)), 10)) + 1
    E_str = "E = $(round(Es.mean, digits=digits)) ± $(round(error, digits=digits))"

    error2 = energy_var_error(Es)
    digits = Int(min(ceil(-log10(error2)), 10)) + 1
    Evar_str = "var(E) = $(round(Es.var, digits=digits)) ± $(round(error2, digits=digits))"

    print(io, "EnergySummary($E_str, $Evar_str, Nₛ=$(length(Es)))")
end