function urand(output, input)
    dim = input
    max_value = sqrt(3/dim)
    min_value = -max_value
    w = min_value .+ rand(output, input) .* (max_value-min_value)
end


function initwb(input_dim::Int, output_dim::Int, atype=_atype, init=xavier, bias=true)
    w = param(output_dim, input_dim; init=init, atype=_atype)
    b = bias ? param(output_dim, 1; atype=_atype) : eltype(w)(0)
    return (w, b)
end
