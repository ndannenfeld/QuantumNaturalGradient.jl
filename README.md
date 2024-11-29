# QuantumNaturalGradient.jl

This package facilitates the computation of the Quantum Natural Gradient, a critical component in the optimization of quantum variational algorithms, especially during the time evolution process. Below, we detail how to integrate and use this package effectively.

## Getting Started

### Prerequisites

Before you begin, ensure that you have a working implementation of the function `Oks_and_Eks(θ::Vector, sample_nr::Integer)`. This function is pivotal for the operation of the Quantum Natural Gradient algorithm, as it computes several key quantities based on a given parameter vector `θ` and a specified number of samples `sample_nr`.

### Function Output Format

The `Oks_and_Eks` function should now return a dictionary. Required fields:

- **`:Oks`** (`Matrix`): Gradient of `<s|ψ>` w.r.t. `θ`, normalized by `<s|ψ>`. Dimensions: `(sample_nr, length(θ))`.
- **`:Eks`** (`Vector`): Expectation value of `<s|H|ψ>/<s|ψ>`.  
- **`:logψs`** (`Vector`): Logarithm of `<s|ψ>` (or `zeros(sample_nr)` if unused).  
- **`:samples`** (`Vector{Int}`): Sampled configurations (or `zeros(Int, sample_nr)` if unused).  
- **`:weights`** (`Vector{Float}`): Importance sampling weights, i.e., `||<s|ψ>||^2 / p(s)` (optional).

#### Example Output
```julia
Dict(
    :Oks => Oks,
    :Eks => Eks,
    :logψs => logψs,
    :samples => samples
)
```

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

Here’s an improved version of the text with better formatting and readability:

---

### Logger Functions

The `evolve` function accepts a keyword argument `logger_funcs`, which is a list of functions whose outputs are saved after every iteration. For example:

```julia
logger_funcs = []
history_params(; θ) = θ
push!(logger_funcs, history_params)
```

In this example, the `history_params` function saves the parameters `θ` after each optimization step.

Currently supported variables include:
- `natural_gradient` (The natural gradient object)
- `θ` (parameters)
- `niter` (number of current iteration)
- `energy` (current energy)
- `norm_natgrad` (norm of the natural gradient)
- `norm_θ` (norm of the parameters)

Additionally, you can log any parameters output by the `Oks_and_Eks` function.