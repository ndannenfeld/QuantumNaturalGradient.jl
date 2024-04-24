function round_auto(x, digits=3)
    d = round(Int, log10(abs(x)))
    if d > digits-1
        return round(Int, x)
    end
    x = round(x; digits=-d+digits)
    return x
end

include("stats.jl")