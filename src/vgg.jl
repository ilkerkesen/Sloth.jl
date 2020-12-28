VGG_CONFIGS = Dict(
    "A" => [
        64, "M",
        128, "M",
        256, 256, "M",
        512, 512, "M",
        512, 512, "M"
    ],
    "B" => [
        64, 64, "M",
        128, 128, "M",
        256, 256, "M",
        512, 512, "M",
        512, 512, "M"
    ],
    "D" => [
        64, 64, "M",
        128, 128, "M",
        256, 256, 256, "M",
        512, 512, 512, "M",
        512, 512, 512, "M"
    ],
    "E" =>  [
        64, 64, "M",
        128, 128, "M",
        256, 256, 256, 256, "M",
        512, 512, 512, 512, "M",
        512, 512, 512, 512, "M"
    ],
)

VGG_URLS = Dict(
    "A" => "https =>//download.pytorch.org/models/vgg11-bbd30ac9.pth",
    "B" => "https =>//download.pytorch.org/models/vgg13-c768596a.pth",
    # "D" => "https =>//download.pytorch.org/models/vgg16-6487cb99.mat",
    "D" => "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat",
    "E" => "https =>//download.pytorch.org/models/vgg19-dcbb9e9d.pth",
)


struct VGG
    features::Sequential
    classifier::Sequential
end


function VGG(; features::Sequential, num_classes::Int=1000)
    classifier = Sequential([
        Linear(in_features=512*7*7, out_features=4096),
        Relu(),
        Linear(in_features=4096, out_features=4096),
        Relu(),
        Linear(in_features=4096, out_features=num_classes),
    ])

    return VGG(features, classifier)
end


function (vgg::VGG)(x)
    x = vgg.features(x)
    x = mat(x)
    x = vgg.classifier(x)
end


function topk(scores, k) 
    scores = vec(Array(scores))
    ranked = sortperm(scores, rev=true)[1:k]
end


function make_vgg_layers(config)
    layers = []
    in_channels = 3
    for out_channels in config
        if out_channels == "M"
            push!(layers, MaxPool())
        else
            conv = Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                mode=1)
            append!(layers, (conv, Relu()))
            in_channels = out_channels
        end
    end
    return Sequential(layers)
end


function _vgg(config; pretrained::Bool=false)
    vgg = VGG(features=make_vgg_layers(VGG_CONFIGS[config])) 
    if pretrained
        url = VGG_URLS[config]
        filepath = dir("..", "data", split(url, "/")[end])
        !isfile(filepath) && download(url, filepath)
        _load_weights!(vgg, matread(filepath))
    end
    return vgg
end


function _load_weights!(vgg::VGG, matdata)
    _layers = matdata["layers"]
    features, classifier = _layers[1:end-6], _layers[end-5:end-1] 
    for (layer, pretrained) in zip(vgg.features.layers, features)
        length(pretrained["weights"]) != 2 && continue
        weight, bias = pretrained["weights"]
        layer.weight.value .= Knet.atype(weight)
        layer.bias.value .= reshape(Knet.atype(bias), size(layer.bias))
    end
    for (layer, pretrained) in zip(vgg.classifier.layers, classifier)
        length(pretrained["weights"]) != 2 && continue
        weight, bias = pretrained["weights"]
        layer.weight.value .= Knet.atype(transpose(mat(weight)))
        layer.bias.value .= reshape(Knet.atype(bias), size(layer.bias))
    end
end


function _get_imagenet_metadata()
    url = VGG_URLS["D"]
    filepath = dir("..", "data", split(url, "/")[end])
    !isfile(filepath) && download(url, filepath)
    meta = matread(filepath)["meta"]
    description = meta["classes"]["description"]
    average_image = meta["normalization"]["averageImage"]
    return average_image, description
end


vgg11(; pretrained::Bool=false) = _vgg("A"; pretrained=pretrained)
vgg13(; pretrained::Bool=false) = _vgg("B"; pretrained=pretrained)
vgg16(; pretrained::Bool=false) = _vgg("D"; pretrained=pretrained)
vgg19(; pretrained::Bool=false) = _vgg("E"; pretrained=pretrained)