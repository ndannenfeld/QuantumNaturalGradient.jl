import Base.+
abstract type AbstractParameters end

ParameterTypes = Union{AbstractVector{<:Number}, AbstractParameters}

struct Parameters{T} <: AbstractParameters
    obj::T
end

Base.BroadcastStyle(::Type{<:Parameters{T}}) where T = Broadcast.Style{Parameters{T}}()
Base.BroadcastStyle(t::Broadcast.Style{Parameters{T}}, ::Broadcast.BroadcastStyle) where T = t
Base.broadcasted(f::F, p::Parameters, bc::Base.Broadcast.Broadcasted) where F = Base.Broadcast.Broadcasted((f), (p, bc))
Base.eltype(p::Parameters{T}) where T = eltype(p.obj)

function LinearAlgebra.norm(p::Parameters{T}) where T
    error("norm not implemented for Parameters{$T}.")
end

bc_type = Base.Broadcast.Broadcasted{Broadcast.Style{Parameters{T}}, Nothing, F, Tuple{Parameters{T}, O}} where T where O where F
function Base.materialize!(dest::Parameters{T}, bc::bc_type) where T
    error("""Materialization of broadcasted Parameters{$T} is not implemented yet. Overwrite the function to implement it
bc_type = Base.Broadcast.Broadcasted{Broadcast.Style{Parameters{T}}, Nothing, F, Tuple{Parameters{T}, O}} where T <:$T where O where F
function Base.materialize!(dest::Parameters{$T}, bc::bc_type)
    θ = convert(Vector, bc.args[1].obj)
    bc = Base.Broadcast.Broadcasted((bc.f), (θ, bc.args[2]))
    θdot = Base.materialize(bc)
    write!(dest.obj, θdot)
    dest
end""")
end
# Custom parameter types can be defined by the user