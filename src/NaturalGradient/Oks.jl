function complex_gradient(pull; complex_input=true, complex_output=true, check_holomorpic=false)
    if !complex_output
        g = pull(1)[1]

    elseif !complex_input
        @warn "Your wave function construction is ψ(θ): R->C. To compute gradients two pullbacks are needed. This is not efficient. Try to make your wave function construction holomorphic ψ(θ): C->C. If your wave function construction is holomorpic make sure to make your parameters complex or use force_holomorphic=true flag." maxlog=1
        g1 = pull(1)[1] # d(real(psi))/dx
        g2 = pull(1im)[1] # d(imag(psi))/dx
        g = g1 + 1im*g2

    else
        # holomorphic
        g1 = pull(1)[1] # d(real(psi))/dx + i * d(real(psi))/dy
        if check_holomorpic
            g2 = pull(1im)[1]# d(imag(psi))/dx + i * d(imag(psi))/dy
            @assert g1 ≈ -1im*g2 "Function is not holomorphic."
        end

        # dpsi/z = (dpsi/dx - i*dpsi/dy) / 2 with psi = real(psi) + i*imag(psi)
        # Cauchy–Riemann equations
        # d real(psi)/dx = d imag(psi)/dy
        # d real(psi)/dy = -d imag(psi)/dx
        # dpsi/dz = conj(g1)

        g = conj(g1)
        #g = (conj(g1) + 1im*conj(g2)) ./ 2
    end
    return g
end