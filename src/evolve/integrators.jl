# Abstract types and common utilities
abstract type AbstractIntegrator end
abstract type AbstractCompositeIntegrator <: AbstractIntegrator end
abstract type AbstractSchedule end

function Base.getproperty(integrator::AbstractCompositeIntegrator, property::Symbol)
    if property === :lr || property === :step
        return getproperty(integrator.integrator, property)
    else
        return getfield(integrator, property)
    end
end

function clamp_and_norm!(gradients, clip_val, clip_norm)
    clamp!(gradients, -clip_val, clip_val)
    norm(gradients) > clip_norm ? gradients .= (gradients ./ norm(gradients)) .* clip_norm : nothing
    return gradients
end

# Euler integrator structure
mutable struct Euler <: AbstractIntegrator
    lr::Float64
    step::Integer
    use_clipping::Bool
    clip_norm::Float64
    clip_val::Float64
    Euler(;lr=0.05, step=0, use_clipping=false, clip_norm=5.0, clip_val=1.0) = new(lr, step, use_clipping, clip_norm, clip_val)
end

# Euler integrator step function
function (integrator::Euler)(θ::ParameterTypes, Oks_and_Eks_, mode::String="IMAG"; kwargs...)
    
    θtype = eltype(θ)
    h = integrator.lr
    if mode=="REAL" h *= im end

    ng = NaturalGradient_timeit_wrapper(θ, Oks_and_Eks_; kwargs...)
    g = get_θdot(ng; θtype)
    if integrator.use_clipping clamp_and_norm!(g, integrator.clip_val, integrator.clip_norm) end

    θ .+= h .* g
    integrator.step += 1

    return θ, ng
end

# (classic) RK4 integrator structure
mutable struct RK4 <: AbstractIntegrator
    lr::Float64
    step::Integer
    use_clipping::Bool
    clip_norm::Float64
    clip_val::Float64

    RK4(;lr=0.05, step=0, use_clipping=false, clip_norm=5.0, clip_val=1.0) = new(lr, step, use_clipping, clip_norm, clip_val)
end

# (classic) RK4 integrator step function
function (integrator::RK4)(θ::ParameterTypes, Oks_and_Eks_::Function, mode::String="IMAG"; kwargs...)

    θtype = eltype(θ)
    h = integrator.lr
    if mode=="REAL" h *= im end

    ng1 = NaturalGradient_timeit_wrapper(θ, Oks_and_Eks_; kwargs...)
    k1 = get_θdot(ng1; θtype)
    if integrator.use_clipping clamp_and_norm!(k1, integrator.clip_val, integrator.clip_norm) end

    θ2 = deepcopy(θ)  
    θ2 .+= (h/2) .* k1
    ng2 = NaturalGradient_timeit_wrapper(θ2, Oks_and_Eks_; kwargs...)
    k2 = get_θdot(ng2; θtype)
    if integrator.use_clipping clamp_and_norm!(k2, integrator.clip_val, integrator.clip_norm) end
    
    θ3 = deepcopy(θ)  
    θ3 .+= (h/2) .* k2
    ng3 = NaturalGradient_timeit_wrapper(θ3, Oks_and_Eks_; kwargs...)
    k3 = get_θdot(ng3; θtype)
    if integrator.use_clipping clamp_and_norm!(k3, integrator.clip_val, integrator.clip_norm) end

    θ4 = deepcopy(θ)  
    θ4 .+= h .* k3
    ng4 = NaturalGradient_timeit_wrapper(θ4, Oks_and_Eks_; kwargs...)
    k4 = get_θdot(ng4; θtype)
    if integrator.use_clipping clamp_and_norm!(k4, integrator.clip_val, integrator.clip_norm) end

    # GC.gc()
    @. θ += (h/6) * (k1 + 2*k2 + 2*k3 + k4)

    integrator.step += 1

    return θ, ng1
end

#= RK4_custom in its current state should be functionally identical to RK4, but can be generalized easily (to pass custom parameters etc.). However, it was observed that computations are not identital to RK4 (maybe the compiler does not optimize away all multiplications with 0?). =#

# # (classic) RK4 integrator structure
# mutable struct RK4_custom <: AbstractIntegrator
#     lr::Float64
#     step::Integer
#     use_clipping::Bool
#     clip_norm::Float64
#     clip_val::Float64
#     # values from Butchers tableau:
#     A::Matrix{Float64}
#     B::Vector{Float64}
#     C::Vector{Float64}

#     RK4_custom(;lr=0.05, step=0, use_clipping=false, clip_norm=5.0, clip_val=1.0) = new(
#         lr, step, use_clipping, clip_norm, clip_val,
#         [ # A coefficients of Butchers tableau
#               0    0  0  0;
#             1/2    0  0  0;
#               0  1/2  0  0;
#               0    0  1  0
#         ],
#         [ # b coefficients of Butchers tableau
#             1/6,
#             1/3,
#             1/3,
#             1/6
#         ],
#         [ # c coefficients of Butchers tableau
#               0,
#             1/2,
#             1/2,
#               1
#         ]   
#     )
# end

# # (classic) RK4 integrator step function
# function (integrator::RK4_custom)(θ::ParameterTypes, Oks_and_Eks_::Function, mode::String="IMAG"; kwargs...)

#     θtype = eltype(θ)
#     h = integrator.lr
#     if mode=="REAL" h *= im end

#     # load parameters corresponding to Butchers tableau. The cs are not actually used here, as dθ/dt is not explicitly time dependent.
#     A, bs, cs = integrator.A, integrator.B, integrator.C

#     # needed both for intermediary evaluations and final update
#     ks = Vector{Vector{θtype}}(undef, length(bs)) # note that although typeof(θ) <: QuantumNaturalGradient.Parameters,  get_θdot returns a Vector{eltype(θ)}, and that is what will be stored in ks.
#     # first natural gradient (i.e. of the current state before updating) will be saved and later returned alongside the updated θ 
#     ng1 = Vector{NaturalGradient{θtype}}(undef, 1)

#     for (i,b) in enumerate(bs)
#         θ_ = deepcopy(θ)
#         for j in 1:i-1
#             @. θ_ += h * A[i,j] * ks[j]
#         end
#         ng = NaturalGradient_timeit_wrapper(θ_, Oks_and_Eks_; kwargs...)
#         ks[i] = get_θdot(ng; θtype)
#         if integrator.use_clipping clamp_and_norm!(ks[i], integrator.clip_val, integrator.clip_norm) end
#         @. θ += h * b * ks[i]
#         # keep 1st natural gradient to return later
#         if i == 1
#             ng1[] = ng
#         end
#     end

#     integrator.step += 1

#     return θ, ng1[]
# end

# Ralstons RK4 integrator structure
# (Ralstons RK4 scheme minimizes the local truncation error for a certain type of problems)
mutable struct RK4_Ralston <: AbstractIntegrator
    lr::Float64
    step::Integer
    use_clipping::Bool
    clip_norm::Float64
    clip_val::Float64
    # values from Butchers tableau:
    A::Matrix{Float64}
    B::Vector{Float64}
    C::Vector{Float64}

    RK4_Ralston(;lr=0.05, step=0, use_clipping=false, clip_norm=5.0, clip_val=1.0) = new(
        lr, step, use_clipping, clip_norm, clip_val,
        [ # A coefficients of Butchers tableau
                                       0                            0                                  0  0;
                                     2/5                            0                                  0  0;
            (-2_889+1_428*sqrt(5))/1_024  (3_785-1_620*sqrt(5))/1_024                                  0  0;
            (-3_365+2_094*sqrt(5))/6_040   (-975-3_046*sqrt(5))/2_552  (467_040+203_968*sqrt(5))/240_845  0
        ],
        [ # b coefficients of Butchers tableau
                             (263 + 24*sqrt(5)) / 1_812,
                          (125 - 1_000*sqrt(5)) / 3_828,
            (3_426_304 + 1_661_952*sqrt(5)) / 5_924_787,
                                 (30 - 4*sqrt(5)) / 123
        ],
        [ # c coefficients of Butchers tableau
                        0,
                      2/5,
            -3*sqrt(5)/16,
                        1
        ]
    )
end

# Ralstons RK4 integrator step function
function (integrator::RK4_Ralston)(θ::ParameterTypes, Oks_and_Eks_::Function, mode::String="IMAG"; kwargs...)

    θtype = eltype(θ)
    h = integrator.lr
    if mode=="REAL" h *= im end

    # load parameters corresponding to Butchers tableau. The cs are not actually used here, as dθ/dt is not explicitly time dependent.
    A, bs, cs = integrator.A, integrator.B, integrator.C

    # needed both for intermediary evaluations and final update
    ks = Vector{Vector{θtype}}(undef, length(bs)) # note that although typeof(θ) <: QuantumNaturalGradient.Parameters,  get_θdot returns a Vector{eltype(θ)}, and that is what will be stored in ks.
    # first natural gradient (i.e. of the current state before updating) will be saved and later returned alongside the updated θ 
    ng1 = Vector{NaturalGradient{θtype}}(undef, 1)

    for (i,b) in enumerate(bs)
        θ_ = deepcopy(θ)
        for j in 1:i-1
            @. θ_ += h * A[i,j] * ks[j]
        end
        ng = NaturalGradient_timeit_wrapper(θ_, Oks_and_Eks_; kwargs...)
        ks[i] = get_θdot(ng; θtype)
        if integrator.use_clipping clamp_and_norm!(ks[i], integrator.clip_val, integrator.clip_norm) end
        @. θ += h * b * ks[i]
        # keep 1st natural gradient to return later
        if i == 1
            ng1[] = ng
        end
    end

    integrator.step += 1

    return θ, ng1[]
end

#=  RK45 (DOPRI5) integrator structure
 This scheme uses 7 evaluations per step (practically 6, because of FSAL) to get a 5th-order global error.
 Because the 4th order accurate solution can be computed with no additional evaluations, it would also be possible to calculate the error of the 4th order solution in every step and implement an adaptive stepsize based on that (give RK45 struct additional fields B_4th_order, current_error and error_tol, then in every loop calculate and update current_error and "if current_error > error_tol : decrease integrator.step (and optionally repeat the current step) : augment integrator.step (or don't change anything)").
=#
 mutable struct RK45 <: AbstractIntegrator
    lr::Float64
    step::Integer
    use_clipping::Bool
    clip_norm::Float64
    clip_val::Float64
    # values from Butchers tableau:
    A::Matrix{Float64}
    B::Vector{Float64}
    C::Vector{Float64}

    RK45(;lr=0.05, step=0, use_clipping=false, clip_norm=5.0, clip_val=1.0) = new(
        lr, step, use_clipping, clip_norm, clip_val,
        [ # A coefficients of Butchers tableau
                       0              0             0         0              0      0  0;
                     1/5              0             0         0              0      0  0;
                    3/40           9/40             0         0              0      0  0;
                   44/45         -56/15          32/9         0              0      0  0;
            19_372/6_561  −25_360/2_187  64_448/6_561  −212/729              0      0  0;
             9_017/3_168        -355/33  46_732/5_247    49/176  -5_103/18_656      0  0;
                  35/384              0     500/1_113   125/192   -2_187/6_784  11/84  0;
        ],
        [ # b coefficients of Butchers tableau
                  35/384,
                       0,
               500/1_113,
                 125/192,
            -2_187/6_784,
                   11/84,
                       0
        ],
        [ # c coefficients of Butchers tableau
               0,
             1/5,
            3/10,
             4/5,
             8/9,
               1,
               1
        ]
    )
end

# RK45 (DOPRI5) integrator step function
function (integrator::RK45)(θ::ParameterTypes, Oks_and_Eks_::Function, mode::String="IMAG"; kwargs...)

    θtype = eltype(θ)
    h = integrator.lr
    if mode=="REAL" h *= im end

    # load parameters corresponding to Butchers tableau. The cs are not actually used here, as dθ/dt is not explicitly time dependent.
    A, bs, cs = integrator.A, integrator.B, integrator.C

    # needed both for intermediary evaluations and final update
    ks = Vector{Vector{θtype}}(undef, length(bs)) # note that although typeof(θ) <: QuantumNaturalGradient.Parameters,  get_θdot returns a Vector{eltype(θ)}, and that is what will be stored in ks.
    # first natural gradient (i.e. of the current state before updating) will be saved and later returned alongside the updated θ 
    ng1 = Vector{NaturalGradient{θtype}}(undef, 1)

    for (i,b) in enumerate(bs)
        θ_ = deepcopy(θ)
        for j in 1:i-1
            @. θ_ += h * A[i,j] * ks[j]
        end
        # first evaluation from this step is identical to to the last evaluation from the previous step (unless there is no previous step).
        if i == 1 && integrator.step > 0
            ks[1] = RK4_FSAL_k
            ng1[] = RK4_FSAL_ng
        else
            ng = NaturalGradient_timeit_wrapper(θ_, Oks_and_Eks_; kwargs...)
            # keep 1st natural gradient to return later
            if i == 1
                ng1[] = ng
            end
            ks[i] = get_θdot(ng; θtype)
            if integrator.use_clipping clamp_and_norm!(ks[i], integrator.clip_val, integrator.clip_norm) end
        end
        @. θ += h * b * ks[i]
        # save last evaluation because it is identical to the first evaluation from the next step.
        if i == length(bs)
            global RK4_FSAL_k = ks[i]
            global RK4_FSAL_ng = ng
        end
    end

    integrator.step += 1

    return θ, ng1[]
end