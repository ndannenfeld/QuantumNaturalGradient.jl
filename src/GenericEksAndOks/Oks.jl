# TODO: try it out and write a version for a flux model
function Ok(logψ::Function, sample_, θ; θ_complex=false, check_holomorpic=false, kwargs...)
    logψσ, pull_logψσ = pullback(f, θ->logψ(sample_, θ))
    ψσ = exp(logψσ)
    if ψσ isa Complex && abs(imag(ψσ))/(abs(real(ψσ))+1e-10) < 1e-14
        complex_output = false
    else
        complex_output = true
    end
    g = complex_gradient(pull_logψσ; complex_input=θ_complex,
                                     complex_output,
                                     check_holomorpic)

    return g, logψσ
end