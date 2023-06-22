function ITensors.sample(ψ::MPS, sample_nr::Integer)
    ψo = orthogonalize(ψ, 1)
    ψo[1] ./= norm(ψo[1])
    return [sample(ψo) for _ in 1:sample_nr::Integer]
end

function get_enviroments(ψ1, ψ2, right::Bool=false)
    @assert length(ψ1) == length(ψ2)
    #TODO: prime linkdim of psi2
    if right
        ψ1 = ψ1[end:-1:1]
        ψ2 = ψ2[end:-1:1]
    end
    
    envs = Vector{ITensor}(undef, length(ψ1))
    envs[1] = ψ1[1]*ψ2[1]
    
    for (i, (ψ1i, ψ2i)) in enumerate(zip(ψ1[2:end], ψ2[2:end]))
        envs[i + 1] = envs[i] * ψ1i * ψ2i
    end
    
    if right
        envs = envs[end:-1:1]
    end
    
    return envs
end
    

"""
Compute ψ respect to the parameters of the MPS.
"""
function Ok(ψ::MPS, dψ::MPS)
    @assert length(ψ) == length(dψ)
    
    N = length(ψ)
    
    env_L = get_enviroments(ψ, dψ, false)
    env_R = get_enviroments(ψ, dψ, true)
    grads = Vector{ITensor}(undef, length(ψ))
    for (i, dψi) in enumerate(dψ)
        if i == 1
            grads[i] = dψi * env_R[2]
        elseif i == N
            grads[i] = dψi * env_L[end-1]
        else
            grads[i] = env_L[i-1] * dψi * env_R[i + 1]
        end
    end
    return grads
end

function Ok_and_Ek(θ, construct_mps, H::MPO; kwargs...)
    ψ, pull = pullback(construct_mps, θ)
    sample_ = sample(ψ, 1)[1]
    return Ok_and_Ek(ψ, H, sample_, pull; kwargs...)
end

function Ok_and_Ek(ψ::MPS, H::MPO, sample_, pull; kwargs...)
    hilbert = siteinds(ψ)
    
    mps_sample = productstate(hilbert, sample_ .- 1)
    Ek = inner(mps_sample', H, ψ; kwargs...)

    f = ψ -> inner(mps_sample, ψ; kwargs...)
    ψσ, g = withgradient(f, ψ)
    
    return pull(g...)[1] ./ ψσ, Ek / ψσ
end

function Ek(ψ::MPS, H::MPO; kwargs...)
    sample_ = sample(ψ, 1)[1]
    return Ek(ψ, H, sample_; kwargs...)
end

function Ek(ψ::MPS, H::MPO, sample_; get_amplitude=false, kwargs...)
    hilbert = siteinds(ψ)
    
    mps_sample = productstate(hilbert, sample_ .- 1)
    Ek = inner(mps_sample', H, ψ; kwargs...)
    ψσ = inner(mps_sample, ψ; kwargs...)
    if get_amplitude
        return Ek / ψσ, ψσ
    end
    
    return Ek / ψσ
end

function Oks_and_Eks(ψ::MPS, H::MPO, samples::Vector{Vector{T}}, pull; kwargs...) where T <: Number

    Oks_Eks = map(sample_ -> Ok_and_Ek(ψ, H, sample_, pull, kwargs...), samples)

    Oks = [Oks_Eks[i][1] for i in 1:length(Oks_Eks)]
    Eks = [Oks_Eks[i][2] for i in 1:length(Oks_Eks)]
    
    return Oks, Eks, samples
end

function Oks_and_Eks(θ, construct_mps, H::MPO, sample_nr::Integer; kwargs...)
    ψ, pull = pullback(construct_mps, θ)
    samples = sample(ψ, sample_nr)
    Oks, Eks = Oks_and_Eks(ψ, H, samples, pull; kwargs...)
    return Oks, Eks, samples
end

function generate_Oks_and_Eks_parallel(construct_mps, H::MPO; kwargs...)
    @everywhere @eval Main begin
        using Random
        using Zygote
        using ITensors
        using SRMPS
        using SRMPS: Ok_and_Ek
    end

    sendto(workers(), construct_mps=construct_mps, H=H, kwargs=kwargs)

    function Oks_and_Eks_parallel(θ, sample_nr::Integer)
        seed = rand(UInt)

        sendto(workers(), θ=θ)
        @everywhere workers() @eval Main begin
            ψ, pull = pullback(construct_mps, θ)
            # Lazy sampling, should compute the enviroments here
            ψo = orthogonalize(ψ, 1)
            ψo[1] ./= norm(ψo[1])
        end

        Oks_Eks = @distributed (vcat) for i = 1:sample_nr
            Random.seed!(seed + i)
            sample_ = sample(Main.ψo)
            Ok_and_Ek(Main.ψ, Main.H, sample_, Main.pull; Main.kwargs...), sample_
        end
        #warning("Oks_and_Eks_parallel: toverify")
        Oks = [Oks_Eks[i][1][1] for i in 1:length(Oks_Eks)]
        Eks = [Oks_Eks[i][1][2] for i in 1:length(Oks_Eks)]
        samples = [Oks_Eks[i][2] for i in 1:length(Oks_Eks)]

        return Oks, Eks, samples
    end

    return Oks_and_Eks_parallel
end

function generate_Oks_and_Eks_ansatz_parallel(construct_mps, H::MPO, sample_nr::Integer; kwargs...)
    
    @everywhere @eval using SRMPS
    sendto(workers(), construct_mps=construct_mps, H=H, kwargs=kwargs)

    @everywhere @eval Main begin
        using Random
        using Zygote
        using SRMPS: Ok_and_Ek, sample
    end
    
    function Oks_and_Eks_parallel(θ)
        seed = rand(UInt)
        sendto(workers(), θ=θ)

        Oks_Eks = @distributed (vcat) for i = 1:sample_nr
            Random.seed!(seed + i)
            θ1 = 2pi .* randn(size(Main.θ))
            ψ, pull = pullback(construct_mps, θ1)
            sample_ = sample(ψ, 1)[1]
            Ok_and_Ek(ψ, Main.H, sample_, pull; Main.kwargs...)
        end
    
        Oks = [Oks_Eks[i][1] for i in 1:length(Oks_Eks)]
        Eks = [Oks_Eks[i][2] for i in 1:length(Oks_Eks)]

        return Oks, Eks
    end

    return Oks_and_Eks_parallel
end
