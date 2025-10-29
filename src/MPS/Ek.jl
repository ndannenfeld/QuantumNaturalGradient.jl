function Ek(ψ::MPS, H::MPO; kwargs...)
    sample_ = sample(ψ, 1)[1]
    return Ek(ψ, H, sample_; kwargs...)
end

#=function Ek(ψ::MPS, H::MPO, sample_; get_amplitude=false, kwargs...)
    hilbert = siteinds(ψ)
    
    mps_sample = productstate(hilbert, sample_ .- 1)
    Ek = inner(mps_sample', H, ψ; kwargs...)
    ψσ = inner(mps_sample, ψ; kwargs...)
    if get_amplitude
        return Ek / ψσ, ψσ
    end
    
    return Ek / ψσ
end=#
