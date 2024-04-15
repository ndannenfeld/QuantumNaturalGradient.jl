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

function ITensors.sample(ψ::MPS, sample_nr::Integer)
    ψo = orthogonalize(ψ, 1)
    ψo[1] ./= norm(ψo[1])
    return [sample(ψo) for _ in 1:sample_nr::Integer]
end