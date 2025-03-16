abstract type AbstractParameter end

ParameterTypes = Union{AbstractVector{<:Number}, AbstractParameter}

# Custom parameter types can be defined by the user in order to avoid copying model parameters