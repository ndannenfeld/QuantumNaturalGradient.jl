function find_removable_param(S::Matrix{Float64}; cut=1e-5)
    eig = eigen(S)
    nE = sum(eig.values ./ eig.values[end] .< cut)
    space = eig.vectors[:, 1:nE]
    space_ = space * space'
    return argmax([norm(space_[:, i]) for i in 1:size(space_, 2)]), nE
end

function find_removable_params(S::Matrix{Float64}; cut=1e-5, verbose=false)
    # https://arxiv.org/pdf/2212.00421.pdf
    ks = []
    col = collect(1:size(S, 1))
    ki, N = find_removable_param(S; cut=cut)

    while N != 0 && length(col) > 3
        push!(ks, col[ki])
        deleteat!(col, ki)
        Sr = S[col, col]
        ki, N = find_removable_param(Sr; cut=cut)
        if verbose
            @show N
        end
    end
    sort!(ks) 
    return ks
end

function number_of_eigen(S::Matrix{Float64}; cut=1e-5)
    eig = eigen(S)
    return sum(eig.values./ eig.values[end] .< cut)
end

function test_removable_param(S, k; cut=1e-5)
    col = collect(1:size(S, 1))
    deleteat!(col, k)
    nE = number_of_eigen(S[col, col]; cut=cut)
    return nE
end



function find_removable_params2(S::Matrix{Float64}; cut=1e-5)
    # https://arxiv.org/pdf/2212.00421.pdf
    # https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.040309
    ks = []
    col = collect(1:size(S, 1))
    N = number_of_eigen(S; cut=cut)
    break_ = false
    while N != 0 && length(col) > 3
        for (ki, k) in enumerate(col)
            N2 = test_removable_param(S[col, col], ki; cut=cut)
            if N2 < N
                N = N2
                push!(ks, k)
                break_ = true
                break
            end
            break_ = false
            
        end
        col = collect(1:size(S, 1))
        sort!(ks)
        deleteat!(col, ks)
        
        if !break_
            break
        end

    end
    
    
    return ks
end

function removeable_params(construct_mps, H, θ, samples::Integer)
    Oks, Eks = QuantumNaturalGradient.generate_Oks_and_Eks_ansatz_parallel(construct_mps, H, samples)(θ)
    S = cov(Oks)
    ks = get_removable_param(S)
    return ks
end