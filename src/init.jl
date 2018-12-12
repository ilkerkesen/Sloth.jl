function urand(output, input)
    dim = input
    max_value = sqrt(3/dim)
    min_value = -max_value
    w = min_value .+ rand(output, input) .* (max_value-min_value)
end


function initwb(input_dim::Int, output_dim::Int, atype=_atype, init=xavier)
    w = param(output_dim, input_dim; init=init, atype=_atype)
    b = param(output_dim, 1; atype=_atype)
    return (w, b)
end
