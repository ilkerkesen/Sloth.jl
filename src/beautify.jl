import Base: show


function show(io::IO, layer::T) where T <: SlothLayer
    head = get_layer_type(layer)
    args = get_layer_args(layer)
    body = join(["$k=$v" for (k,v) in args], ", ")
    str = "$head($body)"
    print(str)
end


function show(io::IO, chain::Chain; level=1, num_spaces=2)
    println("Chain(")
    num_layers = length(chain.layers)
    for (i,l) in enumerate(chain.layers)
        print(repeat(" ", level*num_spaces))
        if isa(l, Chain)
            show(io, l; level=level+1, num_spaces=num_spaces)
        else
            show(io, l)
        end
        i == num_layers && continue
        println(",")
    end
    print(")")
end


function pretty(io::IO, chain::Chain, level=1)

end


get_layer_type(l::SlothLayer) = titlecase(split(string(typeof(l)), ".")[end])
get_layer_type(l::Dense) = l.f == identity ? "Linear" : "Dense"
get_layer_type(l::Activation) = titlecase(split(string(l.f), ".")[end])
get_layer_type(l::Pool) = ("MaxPool", "PaddedMeanPool", "MeanPool")[l.mode+1]


function get_layer_args(l::Dense)
    args = Any[("input", l.inputsize), ("output", l.outputsize)]
    l.f != identity && push!(args, ("f", split(string(l.f), ".")[end]))
    l.b == 0.0 && push!(args, ("bias", false))
    return args
end


function get_layer_args(l::Union{Conv,Deconv})
    args = Any[("input", l.inputsize), ("output", l.outputsize),
               ("kernel",l.kernelsize)]
    l.mode != 0 && push!(args, ("mode", l.mode))
    l.padding != 0 && push!(args, ("padding", l.padding))
    l.stride != 1 && push!(args, ("stride", l.stride))
    l.upscale != 1 && push!(args, ("upscale", l.upscale))
    l.b == 0.0 && push!(args, ("bias", false))
    return args
end


function get_layer_args(l::Pool)
    args = Any[("window", l.window)]
    l.padding != 0 && push!(args, ("padding", l.padding))
    l.stride != l.window && push!(args, ("stride", l.stride))
    l.alpha != 1.0 && push!(args, ("scale", l.alpha))
    return args
end


get_layer_args(l::BatchNorm) = Any[("input", l.inputsize)]


get_layer_args(l::Embedding) = Any[("vocabsize", l.vocabsize),
                                   ("embedsize", l.embedsize)]


get_layer_args(l::Dropout) = Any[("probability", l.p)]


get_layer_args(l::Activation) = Any[]
