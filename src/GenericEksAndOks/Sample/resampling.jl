function get_resample_probs(Ek_terms, sample; resample_energy=0, normalize=true)
    sample_flipped = Vector{Any}(undef, length(Ek_terms))
    probs = Vector{Float64}(undef, length(Ek_terms))
    for (i, (state, val)) in enumerate(Ek_terms)
        sample_flipped[i] = state
        if state isa Tuple
            sample = () # If state is a Tuple then it tells you which sites to flip. If it is empty, then it means no sites should be fliped, wich is equivalent to the untouched sample.
        end
        if state == sample
            probs[i] = abs2(val - resample_energy)
        else
            probs[i] = abs2(val)
        end
    end
    Z = sum(probs)
    if normalize  
        probs = probs ./ Z
    end
    return probs, sample_flipped, Z
end

function resample_with_H(sample, ham_op; resample_energy=0, offset=1)
    Ek_terms = QuantumNaturalGradient.get_precomp_sOψ_elems(ham_op, sample; offset)
    probs, sample_flipped = get_resample_probs(Ek_terms, sample; resample_energy)
    return sample_flipped[sample_p(probs)]
end

"""
get_logprob_resample(
    sample, 
    Ek_terms, 
    logψ_flipped, 
    ham_op; 
    offset=1, 
    resample_energy=0, 
    get_flip_sites=false
)
    sample: The current sample
    Ek_terms: The precomputed terms for the energy
    logψ_flipped: The log of the wavefunction for the flipped samples
    ham_op: The Hamiltonian operator
    resample_energy: The energy offset <H> for the resampled states
"""
function get_logprob_resample(sample, Ek_terms, logψ_flipped, ham_op; resample_energy=0)
    probs, sample_flipped, _ = get_resample_probs(Ek_terms, sample; resample_energy, normalize=false)
    
    for (i, state_patch) in enumerate(sample_flipped)
        sample_c = state_patch
        if state_patch isa Tuple
            sample_c = apply_flip_site(sample, state_patch)
        end
        Ek_terms_ = QuantumNaturalGradient.get_precomp_sOψ_elems(ham_op, sample_c)
        _, _, Z = get_resample_probs(Ek_terms_, sample_c; resample_energy, normalize=false)
        probs[i] /= Z
    end
    logψs = [logψ_flipped[s] for s in sample_flipped]
    
    # \sum_s | ⟨s′| H |s⟩ ⟨s|ψ⟩ |^2 / Z(s)
    # log(sum(probs .* exp.(2 .* real(logψs))))
    logprob = logsumexp(2 .* real(logψs) .+ log.(probs))
    return logprob
end
