module QuantumNaturalGradient

using Base.Threads
using Distributed
using ParallelDataTransfer

using DataStructures
using Statistics
using StatsBase
using LinearAlgebra
using RandomizedLinAlg
using Random
using TimerOutputs
using Parameters
using JLD2
using LogExpFunctions
using Observers
using DataFrames

using Zygote

using ITensors
using ITensorMPS
#using PastaQ: productstate

include("parameters.jl")
include("distributed_extension.jl")
include("NaturalGradient/NaturalGradient.jl")
include("MPS/MPS.jl")
include("GenericEksAndOks/GenericEksAndOks.jl")
include("misc/misc.jl")

include("solver/solver.jl")

include("evolve/evolve.jl")
include("evolve/evolve_old.jl")
include("remove_params.jl")
include("init_params.jl")

include("rte_development/rte_development.jl")

# warns when using or importing the package from .julia/dev
function __init__()
    if occursin(".julia/dev/", pathof(QuantumNaturalGradient))
        @warn "You are currently on the .julia/dev/ version of QuantumNaturalGradient."
        flush(stderr)
    end
end

end # module QuantumNaturalGradient
