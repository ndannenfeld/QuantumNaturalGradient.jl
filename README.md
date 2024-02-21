# QuantumNaturalGradient.jl

This package facilitates the computation of the Quantum Natural Gradient, a critical component in the optimization of quantum variational algorithms, especially during the time evolution process. Below, we detail how to integrate and use this package effectively.

## Getting Started

### Prerequisites

Before you begin, ensure that you have a working implementation of the function `Oks_and_Eks(θ::Vector, sample_nr::Integer)`. This function is pivotal for the operation of the Quantum Natural Gradient algorithm, as it computes several key quantities based on a given parameter vector `θ` and a specified number of samples `sample_nr`.

### Function Output Format

Your implementation of `Oks_and_Eks` should return a tuple with the following elements:

- `::Matrix` - The gradient of `<s|psi>` with respect to `θ`, normalized by `<s|psi>`.
- `::Vector` - The expectation value of the Hamiltonian, `<s|H|psi>/<s|psi>`.
- `logψσs` - The logarithm of `<s|psi>`, useful for debugging. If not needed, return `zeros(sample_nr)`.
- `samples::Vector{Int}` - The sampled samples, useful for debugging. If not needed, return `zeros(Int, sample_nr)`.
- (Optional) `::Vector{Float}` - If you're employing importance sampling $p(s)\neq ||\psi(s)||^2)$, provide the squared absolute value of `||<s|psi>||^2/p(s)`. Otherwise, this can be omitted.

### Imaginary Time Evolution

To perform imaginary time evolution, use the following setup:

```julia
dt = 0.1  # Time step
eigen_cut = 0.1  # Eigenvalue cutoff for solver
integrator = QNG.Euler(lr=dt)  # Define the integrator with learning rate
solver = QNG.EigenSolver(eigen_cut, verbose=true)  # Eigenvalue solver with verbosity
θ = init()  # Initialize your parameter vector

# Evolve the system
@time loss_value, trained_θ, misc = QNG.evolve(Oks_and_Eks, θ; 
                                                integrator, 
                                                verbosity=2,
                                                solver,
                                                sample_nr,  # Number of samples
                                                maxiter,  # Maximum iterations
                                                callback,  # Callback function for each iteration
                                                )
```

### Obtaining the Natural Gradient Object

For direct access to the Natural Gradient object without running the full evolution, you can initialize it as follows:

```julia
ng = NaturalGradient(θ, Oks_and_Eks; sample_nr=100)
```

This allows for more granular control and inspection of the gradient for advanced use cases.
