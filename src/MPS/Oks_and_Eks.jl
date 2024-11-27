function Oks_and_Eks(ψ::MPS, H::MPO, samples::Vector{Vector{T}}, pull; θ_complex=false, kwargs...) where T <: Number

    Oks_Eks = map(sample_ -> Ok_and_Ek(ψ, H, sample_, pull; θ_complex, kwargs...), samples)

    Oks = [Oks_Eks[i][1] for i in 1:length(Oks_Eks)]
    Eks = [Oks_Eks[i][2] for i in 1:length(Oks_Eks)]
    logψσs = [Oks_Eks[i][3] for i in 1:length(Oks_Eks)]
    
    return Oks, Eks, logψσs, samples
end

function Oks_and_Eks(θ, construct_mps, H::MPO, sample_nr::Integer; kwargs...)
    ψ, pull = pullback(construct_mps, θ)
    θ_complex = eltype(θ) <: Complex
    samples = sample(ψ, sample_nr)
    Oks, Eks = Oks_and_Eks(ψ, H, samples, pull; θ_complex, kwargs...)
    return Oks, Eks, samples
end

function generate_Oks_and_Eks_parallel(construct_mps, H::MPO;
                                       pull_in_advance=false, force_holomorphic=false, kwargs...)
    @everywhere @eval Main begin
        using Random
        using Zygote
        using ITensors
        using QuantumNaturalGradient
    end
    sendto(workers(); construct_mps, H, kwargs, pull_in_advance)

    function Oks_and_Eks_parallel(θ, sample_nr::Integer; samples=nothing)
        if samples !== nothing
            sample_nr = length(samples)
        end
        seed = rand(UInt)

        if force_holomorphic
            θ = Complex.(θ)
        end

        sendto(workers(); θ, seed, samples)
        
        @everywhere workers() @eval Main begin
            function get_the_pull()
                ψ, pull = pullback(construct_mps, θ)
                if ψ isa Tuple
                    ψ, loglike = ψ
                else
                    loglike = nothing
                end
                # Lazy sampling, should compute orthogonalization in only once
                ψo = orthogonalize(ψ, 1)
                ψo[1] ./= norm(ψo[1])
                return ψ, ψo, pull, loglike
            end
            if pull_in_advance
                global ψ, ψo, pull, loglike
                ψ, ψo, pull, loglike = get_the_pull()
            end
            θ_complex  = eltype(θ) <: Complex
        end
        
        Oks_Eks = @distributed (vcat) for i in 1:sample_nr
            @eval Main begin
            j = $i
            Random.seed!(seed + j)
            if !pull_in_advance
                ψ, ψo, pull, loglike = get_the_pull()
            end
            if samples === nothing
                sample_ = sample(ψo)
            else
                sample_ = samples[j]
            end
            
            if loglike === nothing
                return QuantumNaturalGradient.Ok_and_Ek(ψ, H, sample_, pull; θ_complex, kwargs...), sample_
            else
                return QuantumNaturalGradient.Ok_and_Ek(ψ, loglike, H, sample_, pull; θ_complex, kwargs...), sample_
            end
            end
        end

        @everywhere_async workers() Base.GC.gc()
        
        Oks = [Oks_Eks[i][1][1] for i in 1:length(Oks_Eks)]
        Eks = [Oks_Eks[i][1][2] for i in 1:length(Oks_Eks)]
        logψs = [Oks_Eks[i][1][3] for i in 1:length(Oks_Eks)]
        samples = [Oks_Eks[i][2] for i in 1:length(Oks_Eks)]
        
        return Dict(:Oks => Oks, :Eks => Eks, :logψs => logψs, :samples => samples)
    end

    return Oks_and_Eks_parallel
end

function Ok_and_Ek(θ, construct_mps, H::MPO; force_holomorphic=false, kwargs...)
    if force_holomorphic
        θ = Complex.(θ)
    end

    θ_complex  = eltype(θ) <: Complex

    ψ, pull = pullback(construct_mps, θ)

    if ψ isa Tuple
        ψ, loglike = ψ
        sample_ = sample(ψ, 1)[1]
        return Ok_and_Ek(ψ, loglike, H, sample_, pull; θ_complex, kwargs...)
    else
        sample_ = sample(ψ, 1)[1]
        return Ok_and_Ek(ψ, H, sample_, pull; θ_complex, kwargs...)
    end
end


function Ok_and_Ek(ψ::MPS, H::MPO, sample_, pull; θ_complex=false, check_holomorpic=false, kwargs...)
    hilbert = siteinds(ψ)
    
    mps_sample = productstate(hilbert, sample_ .- 1)
    Ek = inner(mps_sample', H, ψ; kwargs...)

    function f(ψ)
        ψσ = inner(mps_sample, ψ; kwargs...)
        return ψσ, log(Complex(ψσ))
    end

    (ψσ, logψσ), pull_logψσ = pullback(f, ψ)

    full_pull = x -> pull(pull_logψσ((nothing, x))...)

    g = complex_gradient(full_pull; complex_input=θ_complex, complex_output=ψσ isa Complex, check_holomorpic)

    return g, Ek / ψσ, logψσ
end

function Ok_and_Ek(ψ::MPS, loglike::Number, H::MPO, sample_, pull; θ_complex=false, check_holomorpic=false, kwargs...)
    """
    Compute the gradient and energy of the wave function ψ.
    """
    hilbert = siteinds(ψ)
    
    mps_sample = productstate(hilbert, sample_ .- 1)
    Ek = real(inner(mps_sample', H, ψ; kwargs...))

    function f(ψ)
        ψσ = inner(mps_sample, ψ; kwargs...)
        return ψσ, log(Complex(ψσ))
    end
    
    (ψσ, logψσ), pull_logψσ = pullback(f, ψ)
    # sum of the log(psi) + loglike/2
    
    pull_ = x -> pull((pull_logψσ((nothing, x))..., x./2))
    Ok = complex_gradient(pull_; complex_input=θ_complex, complex_output=ψσ isa Complex, check_holomorpic)
    return Ok, Ek / ψσ, logψσ + loglike/2
end