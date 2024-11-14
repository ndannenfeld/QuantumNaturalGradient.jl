module QuantumNaturalGradient

using Base.Threads
using Distributed
using ParallelDataTransfer

using DataStructures
using Statistics
using LinearAlgebra
using RandomizedLinAlg
using Random
using TimerOutputs
using Parameters
using JLD2
using LogExpFunctions

using Zygote

using ITensors
using PastaQ: productstate


include("distributed_extension.jl")
include("GeometricTensor/GeometricTensor.jl")
include("MPS/MPS.jl")
include("GenericEksAndOks/GenericEksAndOks.jl")
include("misc/misc.jl")


include("solver/solver.jl")

include("evolve/evolve.jl")
include("remove_params.jl")
include("init_params.jl")


end # module QuantumNaturalGradient
