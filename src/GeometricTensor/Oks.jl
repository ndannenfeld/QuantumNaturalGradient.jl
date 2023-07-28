function ITensors.sample(ψ::MPS, sample_nr::Integer)
    ψo = orthogonalize(ψ, 1)
    ψo[1] ./= norm(ψo[1])
    return [sample(ψo) for _ in 1:sample_nr::Integer]
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

function complex_gradient(pull; complex_input=true, complex_output=true, check_holomorpic=false)
    if !complex_output
        g = pull(1)[1]

    elseif !complex_input
        @warn "Your wave function construction is ψ(θ): R->C. To compute gradients two pullbacks are needed. This is not efficient. Try to make your wave function construction holomorphic ψ(θ): C->C. If your wave function construction is holomorpic make sure to make your parameters complex or use force_holomorphic=true flag." maxlog=1
        g1 = pull(1)[1] # d(real(psi))/dx
        g2 = pull(1im)[1] # d(imag(psi))/dx
        g = g1 + 1im*g2

    else
        # holomorphic
        g1 = pull(1)[1] # d(real(psi))/dx + i * d(real(psi))/dy
        if check_holomorpic
            g2 = pull(1im)[1]# d(imag(psi))/dx + i * d(imag(psi))/dy
            @assert g1 ≈ -1im*g2 "Function is not holomorphic."
        end

        # dpsi/z = (dpsi/dx - i*dpsi/dy) / 2 with psi = real(psi) + i*imag(psi)
        # Cauchy–Riemann equations
        # d real(psi)/dx = d imag(psi)/dy
        # d real(psi)/dy = -d imag(psi)/dx
        # dpsi/dz = conj(g1)

        g = conj(g1)
        #g = (conj(g1) + 1im*conj(g2)) ./ 2
    end
    return g
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

function generate_Oks_and_Eks_parallel(construct_mps, H::MPO; pull_in_advance=false, force_holomorphic=false, kwargs...)
    @everywhere @eval Main begin
        using Random
        using Zygote
        using ITensors
        using QuantumNaturalGradient
    end

    sendto(workers(), construct_mps=construct_mps, H=H, kwargs=kwargs)

    function Oks_and_Eks_parallel(θ, sample_nr::Integer)
        seed = rand(UInt)

        if force_holomorphic
            θ = Complex.(θ)
        end

        sendto(workers(), θ=θ, seed=seed, pull_in_advance=pull_in_advance)
        @everywhere workers() @eval Main begin
            function get_the_pull()
                ψ, pull = pullback(construct_mps, θ)
                if ψ isa Tuple
                    ψ, loglike = ψ
                else
                    loglike = nothing
                end
                # Lazy sampling, should compute the enviroments here
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
            Random.seed!(seed + i)
            if Main.pull_in_advance
                ψ, ψo, pull, loglike = Main.ψ, Main.ψo, Main.pull, Main.loglike
            else
                ψ, ψo, pull, loglike = Main.get_the_pull()
            end
            
            sample_ = sample(ψo)
            
            if loglike === nothing
                QuantumNaturalGradient.Ok_and_Ek(ψ, Main.H, sample_, pull; θ_complex=Main.θ_complex, Main.kwargs...), sample_
            else
                QuantumNaturalGradient.Ok_and_Ek(ψ, loglike, Main.H, sample_, pull; θ_complex=Main.θ_complex, Main.kwargs...), sample_
            end
        end

        @everywhere_async workers() Base.GC.gc()
        
        Oks = [Oks_Eks[i][1][1] for i in 1:length(Oks_Eks)]
        Eks = [Oks_Eks[i][1][2] for i in 1:length(Oks_Eks)]
        logψs = [Oks_Eks[i][1][3] for i in 1:length(Oks_Eks)]
        samples = [Oks_Eks[i][2] for i in 1:length(Oks_Eks)]

        return Oks, Eks, logψs, samples
    end

    return Oks_and_Eks_parallel
end