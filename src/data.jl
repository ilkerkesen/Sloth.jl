function matconvnet(filepath)
    filepath = abspath(filepath)
    matfile = splitdir(filepath)[2]
    @info("Loading $matfile...")
    !isfile(filepath) && error("no file in the path")
    return matread(filepath)
end


function imgdata(img, averageImage=0.0f0)
    a0 = load(img)
    new_size = ntuple(i->div(size(a0,i)*224,minimum(size(a0))),2)
    a1 = Images.imresize(a0, new_size)
    i1 = div(size(a1,1)-224,2)
    j1 = div(size(a1,2)-224,2)
    b1 = a1[i1+1:i1+224,j1+1:j1+224]
    # ad-hoc solution for Mac-OS image
    macfix = convert(Array{FixedPointNumbers.Normed{UInt8,8},3}, channelview(b1))
    c1 = permutedims(macfix, (3,2,1))
    d1 = convert(Array{Float32}, c1)
    e1 = reshape(d1[:,:,1:3], (224,224,3,1))
    f1 = (255 * e1 .- averageImage)
    g1 = permutedims(f1, [2,1,3,4])
end


function get_vgg_params(matfile)
    layers = matfile["layers"]
    weights, has_weights = [], []
    layer_types = ["conv", "relu", "pool", "fc", "prob"]

    for l in layers
        get_layer_type(x) = startswith(l["name"], x)
        operation = filter(x -> get_layer_type(x), layer_types)[1]
        push!(has_weights, haskey(l, "weights") && length(l["weights"]) != 0)

        if has_weights[end]
            w = copy(l["weights"])
            if operation == "conv"
                w[2] = reshape(w[2], (1,1,length(w[2]),1))
            elseif operation == "fc"
                w[1] = transpose(mat(w[1]))
            end
            push!(weights, w)
        end
    end

    return hcat(weights...), has_weights
end
