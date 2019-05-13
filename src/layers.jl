mutable struct Chain
    layers
end


function (c::Chain)(x; nlayers=length(c.layers))
    for (i,l) in enumerate(c.layers)
        x = l(x)
        i == nlayers && break
    end
    return x
end


(c::Chain)(x,y) = nll(c(x),y)


mutable struct FullyConnected
    w
    b
    f
end


(l::FullyConnected)(x) = y = l.f.(l.w * mat(x) .+ l.b)


function FullyConnected(input_dim::Int, output_dim::Int, f=relu;
                        atype=_atype, init=xavier, bias=true)
    w, b = initwb(input_dim, output_dim, atype, init, bias)
    return FullyConnected(w, b, f)
end


Dense = FullyConnected


Linear(input_dim::Int, output_dim::Int; o...) = FullyConnected(
    input_dim, output_dim, identity; o...)


mutable struct Conv
    w
    b
    padding
    stride
    upscale
    mode
end


function (l::Conv)(x; padding=l.padding, stride=l.stride,
                   mode=l.mode, upscale=l.upscale)
    conv4(l.w, x;
          padding=padding, stride=stride, mode=mode, upscale=upscale) .+ l.b
end


function Conv(ci::Int, co::Int, k::Int; padding=0, stride=1, upscale=1, mode=0,
              atype=_atype, initw=xavier, initb=zeros, bias=true)
    w = param(k, k, ci, co; atype=atype, init=initw)
    b = bias ? param(1, 1, co, 1; atype=atype, init=initb) : eltype(w)(0)
    return Conv(w, b, padding, stride, upscale, mode)
end


mutable struct Deconv
    w
    b
    padding
    stride
    upscale
    mode
end


function (l::Deconv)(x; padding=l.padding, stride=l.stride,
                     mode=l.mode, upscale=l.upscale)
    deconv4(l.w, x;
            padding=padding, stride=stride, mode=mode, upscale=upscale) .+ b
end


function Deconv(ci::Int, co::Int, k::Int;
                padding=0, stride=1, upscale=1, mode=0,
                atype=_atype, initw=xavier, initb=zeros, bias=false)
    w = param(k, k, co, ci; atype=atype, init=initw)
    b = bias ? param(1, 1, co, 1; atype=atype, init=initb) : eltype(w)(0)
    return Deconv(w, b, padding, stride)
end


mutable struct Pool
    window
    padding
    stride
    mode
    maxpoolingNanOpt
    Pool(; window=2, padding=0, stride=window, mode=0, maxpoolingNanOpt=0) =
        new(window, padding, stride, mode, maxpoolingNanOpt)
end


function (l::Pool)(x; window=l.window, padding=l.padding, stride=l.stride,
                   mode=l.mode, maxpoolingNanOpt=l.maxpoolingNanOpt)
    pool(x; window=window, padding=padding, stride=stride, mode=mode,
         maxpoolingNanOpt=maxpoolingNanOpt)
end


mutable struct BatchNorm
    w
    m
end


(l::BatchNorm)(x; o...) = batchnorm(x, l.m, l.w; o...)


function BatchNorm(dim::Int; atype=_atype)
    w = Param(atype(bnparams(dim)))
    m = bnmoments()
    return BatchNorm(w, m)
end


mutable struct Embedding
    w
end


(l::Embedding)(x) = l.w[:, x]


function Embedding(vocabsize::Int, embedsize::Int; atype=_atype, init=xavier)
    w = param(embedsize, vocabsize; init=init, atype=atype)
    return Embedding(w)
end


mutable struct Dropout
    p
end


(l::Dropout)(x; kwargs...) = dropout(x, l.p; kwargs...)


Dropout() = Dropout(0.5)


mutable struct Activation
    f
end


(l::Activation)(x) = l.f.(x)


Activation() = Activation(relu)
Relu() = Activation()
Tanh() = Activation(tanh)
Sigm() = Activation(sigm)
