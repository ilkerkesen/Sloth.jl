VGG_CONFIGS = Dict(
    "D" => [(3,64,2), (64,128,2), (128,256,3), (256,512,3), (512,512,3)],
    "E" => [(3,64,2), (64,128,2), (128,256,4), (256,512,4), (512,512,4)],
)


function VGGBody(config="D"; atype=_atype)
    layers = [VGGLayer(x...; atype=atype) for x in VGG_CONFIGS[config]]
    vcat(layers...)
end


function VGGLayer(input, output, nlayers; atype=_atype)
    layer(x) = Conv(x, output, 3; atype=atype, mode=1, padding=1)
    layers = Any[layer(input), Relu()]
    for i = 2:nlayers; push!(layers, layer(output), Relu()); end
    push!(layers, Pool())
    return layers
end


VGGTail(num_classes=1000; atype=_atype) =  [
    Dense(7*7*512, 4096; atype=atype), Dropout(),
    Dense(4096,    4096; atype=atype), Dropout(),
    Linear(4096, num_classes; atype=atype)]


function VGG(config="D", num_classes=1000; atype=_atype)
    body = VGGBody(config; atype=atype)
    tail = VGGTail(num_classes; atype=atype)
    return Chain(vcat(body, tail))
end


# function load_weights!(vgg::VGG, filepath)
#     for par in params(vgg)
#         par.
#     end
# end
