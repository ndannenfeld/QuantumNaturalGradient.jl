abstract type AbstractTensorOperatorSum end

struct TensorOperatorSum <: AbstractTensorOperatorSum
    tensors::Vector{ITensor}
    hilbert::Array{<:Index}
    sites::Vector{Vector}
end
Base.size(t::TensorOperatorSum, args...) = size(t.hilbert, args...)
Base.ndims(t::TensorOperatorSum) = ndims(t.hilbert)

"""
    TensorOperatorSum(tensors, hilbert, sites)
    Generates a TensorOperatorSum from a hamiltonian and a hilbert space. It precomputes the sites where the operator acts on.
"""
function TensorOperatorSum(ham::OpSum, hilbert::Array)
    
    tensors = Vector{ITensor}(undef, length(ham))
    sites = Vector{Vector{Int}}(undef, length(ham))
    hilbert_flat = hilbert
    if ndims(hilbert) > 1
        hilbert_flat = hilbert[:]
        ## Reduce dimensionality of hamiltonian to 1D for compatibility pourposes
        ham = reduce_dim(ham, size(hilbert))
    end
    
    for (i, o) in enumerate(ham)
        tensors[i] = ITensor(o, hilbert[:])
        sites[i] = get_active_sites(o)
    end
    TensorOperatorSum(tensors, hilbert, sites)
end

"""
Generates a Dictionary with the entries being (sample_config=> energy_weight) so that Ek=sum energy_weight * exp(logψ(sample_config) - logψ(sample))
get_precomp_sOψ_elems(
    tensor::ITensor, 
    sites::Vector, 
    sample_, 
    hilbert; 
    sum_precompute = DefaultOrderedDict(()->0.), 
    offset=1, 
    get_flip_sites=false # If true, it will give a Dictionary Dict(sites_flipped=>energy_weight) instead of Dict(s'=>energy_weight)
    sites_fliiped: First index is the site, second the new s'_i.
)
"""
function get_precomp_sOψ_elems!(tensor::ITensor, sites::Vector, sample_, hilbert; sum_precompute = DefaultOrderedDict(()->0.), offset=1, get_flip_sites=false)
    sample_r = sample_[sites]
    hilbert_r = hilbert[sites]
    
    indices_sample = collect(hi' => s for (hi, s) in zip(hilbert_r, sample_r)) # Selects the indices that act on the tensor from the left O|s>
    
    for sample_r2 in bitstring_permutations(length(sites))
        sample_r2 .+= offset
        indices = vcat(indices_sample, collect(hi => s for (hi, s) in zip(hilbert_r, sample_r2))) # <s'|
        vi = tensor[indices...] # <s'|O|s>
        if vi != 0.
            if get_flip_sites
                key = get_diff(sample_r, sample_r2, sites)
            else
                sample_M = deepcopy(sample_)
                sample_M[sites] .= sample_r2
                key = sample_M
            end

            sum_precompute[key] += vi
            
        end
    end

    # Make real if the imaginary part is too small
    for (key, value) in sum_precompute
        if value isa Complex && abs(imag(value))/(abs(real(value))+1e-10) < 1e-14
            sum_precompute[key] = real(value)
        end

        if sum_precompute[key] == 0.
            delete!(sum_precompute, key)
        end
    end 

    return sum_precompute
end

"""
get_diff([1, 2], [2,2], [4, 5])

1-element Vector{Any}:
 (4, 2)
"""
function get_diff(sample_orig, sample_shifted, sites)
    diffs = []
    for (i, s_orig, s_shift) in zip(sites, sample_orig, sample_shifted)
        if s_orig != s_shift
            push!(diffs, (i, s_shift))
        end
    end
    return diffs
end

function get_precomp_sOψ_elems(tso::TensorOperatorSum, sample_::Array; sum_precompute = DefaultOrderedDict(()->0.), offset=1, get_flip_sites=false)
    @assert size(sample_) == size(tso)
    sample_ = sample_[:] # Flatten the sample
    for (tensor, sites) in zip(tso.tensors, tso.sites)
        get_precomp_sOψ_elems!(tensor, sites, sample_, tso.hilbert[:]; sum_precompute, offset, get_flip_sites)
    end

    # If not 1D, reshape the samples
    if ndims(tso) > 1
        sum_precompute = increase_dim(sum_precompute, size(tso); get_flip_sites)
    end
    return sum_precompute
end
function inner_div_ψ(sample_, tso::TensorOperatorSum, logψ_func; logψ_sample=nothing)
    sum_precompute = get_precomp_sOψ_elems(tso, sample_)

    if logψ_sample === nothing
        logψ_sample = logψ_func(sample_)
    end
    
    O0 = 0
    if sample_ in keys(sum_precompute)
        O0 = sum_precompute[sample_]
        delete!(sum_precompute, sample_)
    end
    
    O = sum(weight * exp(logψ_func(sample_r) - logψ_sample) for (sample_r, weight) in sum_precompute; init=O0)
    if O isa Complex && abs(imag(O))/(abs(real(O))+1e-10) < 1e-14
        O = real(O)
    end
    return O
end

get_Ek(sample_, tso::TensorOperatorSum, logψ_func; logψ_sample=nothing) = inner_div_ψ(sample_, tso, logψ_func; logψ_sample)

inner_div_ψ(sample_, opsum::OpSum, logψ_func, hilbert; logψ_sample=nothing) = inner_div_ψ(sample_, TensorOperatorSum(opsum, hilbert), logψ_func; logψ_sample)
get_Ek(sample_, opsum::OpSum, logψ_func, hilbert; logψ_sample=nothing) = inner_div_ψ(sample_, opsum, logψ_func, hilbert; logψ_sample)

# TODO write a more general version for higher physical dimensions
function bitstring_permutations(n)
    ss = []
    for i = 0:2^n-1
       s = bitstring(i)
       s = s[end-n+1:end]
       s = [parse(Int, si) for si in s]
       push!(ss, s)
    end
    return ss
end

function get_active_sites(o::Scaled)
    return [a.sites[1] for a in ITensors.argument(o).args[1]]
end

## Reduce dimensionality of hamiltonian to 1D for compatibility pourposes
reduce_dim(O::Op, size::Tuple) = ITensors.Op(O.which_op, reduce_dim(O.sites[1], size))
reduce_dim(v::Vector, size::Tuple) = [reduce_dim(vi, size) for vi in v]
function reduce_dim(t::Tuple, size::Tuple)
    n = 0
    mul = 1
    for (ti, si) in zip(t, size)
        n += mul * (ti-1)
        mul *= si
    end
    return n + 1
end
function increase_dim(t::Integer, size::Tuple)
    n = t - 1
    t = []
    for si in size
        push!(t, n % si + 1)
        n ÷= si
    end
    return tuple(t...)
end

function increase_dim(sum_precompute, size_; get_flip_sites=false)
    sum_precompute2 = DefaultOrderedDict(()->0.)
    if !get_flip_sites
        for (sample__, v) in sum_precompute
            sum_precompute2[reshape(sample__, size_)] = v
        end
    elseif get_flip_sites
        for (diff, v) in sum_precompute
            diff_res = []
            for (i, s) in diff
                push!(diff_res, (increase_dim(i, size_), s))
            end
            
            sum_precompute2[diff_res] = v
        end
    end
    return sum_precompute2
end

function reduce_dim!(O::ITensors.Scaled, size::Tuple)
    a = ITensors.argument(O)
    a.args[1] .= reduce_dim(a.args[1], size)
end

function reduce_dim(O::ITensors.OpSum, size::Tuple)
    O_c = deepcopy(O)
    for oi in O_c
        reduce_dim!(oi, size)
    end
    return O_c
end