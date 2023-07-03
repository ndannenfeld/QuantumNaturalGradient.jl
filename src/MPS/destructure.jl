function make_tensor(storage, inds)
    storage = NDTensors.Dense(storage)
    return ITensor(ITensors.Tensor(ITensors.AllowAlias(), storage, inds))
end


function destructure(ψ::MPS; get_destructor=false)
    
    xs = [ψi.tensor.storage for ψi in ψ]
    if get_destructor
        return vcat(vec.(xs)...), p -> _restructure(ψ, p), p -> destructure(p, ψ)
    end
    
    return vcat(vec.(xs)...), p -> _restructure(ψ, p)
end

function destructure(ψ::Union{MPS, Vector{ITensor}}, ψ_struct::MPS)
    ψ = fix_indices!(ψ, ψ_struct)

    xs = [ψi.tensor.storage for ψi in ψ]
    xs = vcat(vec.(xs)...)

    return xs, p -> _restructure(ψ, p)
end
    
MPS_tuple = NamedTuple{(:data, :llim, :rlim), Tuple{Vector{ITensor}, Nothing, Nothing}}
    
function fix_indices(t1::ITensor, t2::ITensor)
    perm = Tuple(ITensors.getperm(inds(t1), inds(t2)))
    return ITensor(ITensors.permutedims(t1.tensor, perm))
end

function fix_indices!(ψ1::Union{MPS, Vector{ITensor}}, ψ2::MPS)
    for i in 1:length(ψ1)
        ψ1[i] = fix_indices(ψ1[i], ψ2[i])
    end

    return ψ1
end

function destructure_grad(ψ::MPS, dψ::MPS)
    grads = Ok(ψ, dψ)
    return destructure(grads, ψ)
end

function _restructure(ψ::MPS, xs::Vector{T}) where T <: Number
    ψo = Vector{ITensor}(undef, length(ψ))
    l = 0
    for (i, ψi) in enumerate(ψ)
        x = ψi.tensor.storage
        ψoi_storage = xs[l.+(1:length(x))]
        ψoi = make_tensor(ψoi_storage, ψi.tensor.inds)
        l += length(x)
        ψo[i] = ψoi
    end
    return MPS(ψo)
end

function _restructure(ψ1::MPS, xs::MPS)
    error("Known bug in Zygote:, you need to use pullback instead of gradient.")
end

Zygote.@adjoint function _restructure(ψ::MPS, xs::Vector{T})  where T <: Number
    ψr = _restructure(ψ, xs)
    ψr, dm -> (nothing, destructure_grad(ψr, dm)[1])
end