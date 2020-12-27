import AutoGrad: recording

abstract type SlothLayer; end


struct Sequential
    layers::Array{T} where T <: Union{Sequential, SlothLayer}
end


Sequential(layers...) = Sequential([layers...])


function (model::Sequential)(x)
    for layer in model.layers
        x = layer(x)
    end
    return x
end


struct Linear <: SlothLayer
    in_features::Int
    out_features::Int
    weight::SlothWeight
    bias::SlothBias
end


function (layer::Linear)(x)
    return layer.weight * x .+ layer.bias
end


function Linear(; in_features::Int, out_features::Int, init=xavier, bias=true)
    w = param(out_features, in_features; init=init)
    b = ifelse(bias, param0(out_features, 1), F(0.0))
    return Linear(in_features, out_features, w, b)
end


struct Conv <: SlothLayer
    in_channels::Int
    out_channels::Int
    kernel_size::IntHyperparam
    stride::IntHyperparam
    padding::IntHyperparam
    dilation::IntHyperparam
    mode::Int
    weight::SlothWeight
    bias::SlothBias
end


function (layer::Conv)(x)
    y = conv4(layer.weight, x;
        padding=layer.padding,
        stride=layer.stride,
        dilation=layer.dilation,
        mode=layer.mode)
    return y .+ layer.bias
end


function Conv(;
    in_channels::Int,
    out_channels::Int,
    kernel_size::IntHyperparam,
    stride::IntHyperparam=1,
    padding::IntHyperparam=0,
    dilation::IntHyperparam=1,
    mode::Int=0,
    bias::Bool=true,
    init=xavier)

    k = kernel_size
    k = ifelse(typeof(k) <: Int, (k, k), k)

    w = param(k..., in_channels, out_channels; init=init)
    b = ifelse(bias, param0(1, 1, out_channels, 1), F(0.0))

    return Conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        mode,
        w,
        b)
end


struct ConvTranspose <: SlothLayer
    in_channels::Int
    out_channels::Int
    kernel_size::IntHyperparam
    stride::IntHyperparam
    padding::IntHyperparam
    dilation::IntHyperparam
    mode::Int
    weight::SlothWeight
    bias::SlothBias
end


function (layer::ConvTranspose)(x)
    y = deconv4(layer.weight, x;
        padding=layer.padding,
        stride=layer.stride,
        dilation=layer.dilation,
        mode=layer.mode)
    y = y .+ layer.bias
end


function ConvTranspose(;
    in_channels::Int,
    out_channels::Int,
    kernel_size::IntHyperparam,
    stride::IntHyperparam=1,
    padding::IntHyperparam=0,
    dilation::IntHyperparam=1,
    mode::Int=0,
    bias::Bool=true,
    init=xavier)

    k = kernel_size
    k = ifelse(typeof(k) <: Int, (k, k), k)

    w = param(k..., out_channels, in_channels; init=init)
    b = ifelse(bias, param0(1, 1, out_channels, 1), F(0.0))

    return ConvTranspose(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        mode,
        w,
        b)
end


@enum PoolMode MAX_POOL=0 PADDED_AVG_POOL=1 AVG_POOL=2


struct Pool <: SlothLayer
    window::IntHyperparam
    stride::IntHyperparam
    padding::IntHyperparam
    dilation::IntHyperparam
    mode::PoolMode
end


function Pool(;
    window::IntHyperparam=2,
    stride::Union{Nothing, IntHyperparam}=nothing,
    padding::IntHyperparam=0,
    dilation::IntHyperparam=1,
    mode::PoolMode
)

    dilation != 1 && @warn "Knet does not support dilated pooling."
    stride = ifelse(isa(stride, Nothing), window, stride)
    return Pool(window, stride, padding, dilation, mode)
end


function (layer::Pool)(x)
    pool(x;
        window=layer.window,
        stride=layer.stride,
        padding=layer.padding,
        mode=Int(layer.mode))
end


MaxPool(; kwargs...) = Pool(; mode=MAX_POOL, kwargs...)
PaddedAvgPool(; kwargs...) = Pool(; mode=PADDED_AVG_POOL, kwargs...)
AvgPool(; kwargs...) = Pool(; mode=AVG_POOL, kwargs...)


struct BatchNorm{T<:AbstractFloat} <: SlothLayer
    num_features::Int
    momentum::T
    eps::T
    weight::SlothWeight
    moments::BNMoments
end


function (layer::BatchNorm)(x)
    batchnorm(x, layer.moments, layer.weight)
end


function BatchNorm(;
    num_features::Int,
    momentum::T=F(0.1),
    eps::T=F(1.0f-5)
) where T <: AbstractFloat
    weight, moments = Param(Knet.atype((bnparams(num_features)))), bnmoments()
    return BatchNorm(num_features, momentum, eps, weight, moments)
end


struct Embedding <: SlothLayer
    num_embeddings::Int
    embedding_dim::Int
    weight::SlothWeight
end


function (layer::Embedding)(x)
    return layer.weight[:, x]
end


function Embedding(; num_embeddings::Int, embedding_dim::Int, init=xavier)
    weight = param(embedding_dim, num_embeddings)
    return Embedding(num_embeddings, embedding_dim, weight)
end


struct Dropout <: SlothLayer
    p::AbstractFloat
    Dropout(; p::AbstractFloat=F(0.5)) = new(p)
end


function (layer::Dropout)(x; drop=recording())
    return dropout(x, layer.p; drop=drop)
end


struct Activation <: SlothLayer
    f
end


(l::Activation)(x) = l.f.(x)


Relu() = Activation(relu)
Tanh() = Activation(tanh)
Sigm() = Activation(sigm)


struct LeakyRelu
    α::AbstractFloat
    LeakyRelu(α::AbstractFloat=F(0.02)) = new(α)
end


function (layer::LeakyRelu)(x)
    pos = relu.(x)
    neg = min.(F(0.0), x)
    return pos + layer.α .* neg
end


struct Flatten <: SlothLayer; end


(l::Flatten)(x) = reshape(x, :,  size(x)[end])