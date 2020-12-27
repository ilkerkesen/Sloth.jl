import Base: show


function show(io::IO, layer::T) where T <: SlothLayer
    head = get_layer_type(layer)
    args = get_layer_args(layer)
    body = join(["$k=$v" for (k,v) in args], ", ")
    str = "$head($body)"
    print(str)
end


function show(io::IO, sequential::Sequential; level=1, num_spaces=2)
    println("Sequential(")
    num_layers = length(sequential.layers)
    for (i,l) in enumerate(sequential.layers)
        print(repeat(" ", level*num_spaces))
        if isa(l, Sequential)
            show(io, l; level=level+1, num_spaces=num_spaces)
        else
            show(io, l)
        end
        i == num_layers && continue
        println(",")
    end
    print(")")
end


get_layer_type(l::SlothLayer) = titlecase(split(string(typeof(l)), ".")[end])
get_layer_type(l::Pool) = ("MaxPool", "PaddedAvgPool", "AvgPool")[Int(l.mode)+1]
get_layer_type(l::Activation) = titlecase(split(string(l.f), ".")[end])


function get_layer_args(layer::Linear)
    args = Any[
        ("in_features", layer.in_features),
        ("out_features", layer.out_features)]
    isa(layer.bias, AbstractFloat) && push!(args, ("bias", false))
    return args
end


function get_layer_args(layer::Union{Conv, ConvTranspose})
    args = Any[
        ("in_channels", layer.in_channels),
        ("output", layer.outputsize),
        ("kernel",layer.kernelsize)]
    layer.mode != 0 && push!(args, ("mode", layer.mode))
    layer.padding != 0 && push!(args, ("padding", layer.padding))
    layer.stride != 1 && push!(args, ("stride", layer.stride))
    layer.bias == F(0.0) && push!(args, ("bias", false))
    return args
end


function get_layer_args(l::Pool)
    args = Any[]
    Int(layer.mode) != 0 && push!("mode", layer.mode)
    push!(args, ("window", layer.window))
    layer.padding != 0 && push!(args, ("padding", layer.padding))
    layer.stride != layer.window && push!(args, ("stride", layer.stride))
    layer.alpha != 1.0 && push!(args, ("scale", layer.alpha))
    return args
end


get_layer_args(layer::BatchNorm) = Any[("input", layer.num_features)]


get_layer_args(layer::Embedding) = Any[
    ("num_embeddings", layer.num_embeddings),
    ("embedding_dim", layer.embedding_dim)]


get_layer_args(layer::Dropout) = Any[("p", layer.p)]


get_layer_args(l::Activation) = Any[]
