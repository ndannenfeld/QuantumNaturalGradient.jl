include("destructure.jl")
include("Ok.jl")
include("LinkExtensions.jl")
include("Ek.jl")
include("Oks_and_Eks.jl")


function NaturalGradient(θ, construct_mps, H::MPO;
    sample_nr::Integer=100, parallel=false,
    solver=nothing, discard_outliers=0.,
    kwargs...)

    if parallel
        Oks_and_Eks_ = generate_Oks_and_Eks_parallel(construct_mps, H; kwargs...)
    else
        Oks_and_Eks_ = (θ, sample_nr) -> Oks_and_Eks(θ, construct_mps, H, sample_nr; kwargs...)
    end
    return NaturalGradient(θ, Oks_and_Eks_; sample_nr, solver, discard_outliers)
end