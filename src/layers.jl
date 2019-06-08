abstract type SlothLayer; end

mutable struct Chain
    layers
end


function (c::Chain)(x)
    for l in c.layers; x = l(x); end
    return x
end


(c::Chain)(x,y) = nll(c(x),y)


mutable struct Dense <: SlothLayer
    inputsize
    outputsize
    f
    w
    b
end


(l::Dense)(x) = y = l.f.(l.w * mat(x) .+ l.b)


function Dense(; input::Int, output::Int, f=relu,
               atype=_atype, init=xavier, bias=true)
    w, b = initwb(input, output, atype, init, bias)
    return Dense(input, output, f, w, b)
end


function Dense(input::Int, output::Int, f=relu; kwargs...)
    Dense(input=input, output=output, f=f, kwargs...)
end


FullyConnected = Dense


function Linear(input::Int, output::Int; kwargs...)
    Dense(input=input, output=output, f=identity; kwargs...)
end


function Linear(; input::Int, output::Int, kwargs...)
    Dense(; input=input, output=output, f=identity, kwargs...)
end


mutable struct Conv <: SlothLayer
    inputsize
    outputsize
    kernelsize
    padding
    stride
    upscale
    mode
    w
    b
end


function (l::Conv)(x; padding=l.padding, stride=l.stride,
                   mode=l.mode, upscale=l.upscale)
    conv4(l.w, x;
          padding=padding, stride=stride, mode=mode, upscale=upscale) .+ l.b
end


function Conv(; input::Int, output::Int, kernel::Int,
              padding=0, stride=1, upscale=1, mode=0,
              atype=_atype, initw=xavier, initb=zeros, bias=true)
    w = param(kernel, kernel, input, output; atype=atype, init=initw)
    b = bias ? param(1, 1, output, 1; atype=atype, init=initb) : eltype(w)(0)
    return Conv(input, output, kernel, padding, stride, upscale, mode, w, b)
end


Conv(input::Int, output::Int, kernel::Int; kwargs...) = Conv(
    input=input, output=output, kernel=kernel; kwargs...)


mutable struct Deconv <: SlothLayer
    inputsize
    outputsize
    kernelsize
    padding
    stride
    upscale
    mode
    w
    b
end


function (l::Deconv)(x; padding=l.padding, stride=l.stride,
                     mode=l.mode, upscale=l.upscale)
    deconv4(l.w, x;
            padding=padding, stride=stride, mode=mode, upscale=upscale) .+ l.b
end


function Deconv(; input::Int, output::Int, kernel::Int,
                padding=0, stride=1, upscale=1, mode=0,
                atype=_atype, initw=xavier, initb=zeros, bias=false)
    w = param(kernel, kernel, output, input; atype=atype, init=initw)
    b = bias ? param(1, 1, output, 1; atype=atype, init=initb) : eltype(w)(0)
    return Deconv(input, output, kernel, padding, stride, upscale, mode, w, b)
end


Deconv(input::Int, output::Int, kernel::Int; kwargs...) = Deconv(
    input=input, output=output, kernel=kernel; kwargs...)


ConvTransposed = Deconv



mutable struct Pool <: SlothLayer
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


mutable struct BatchNorm <: SlothLayer
    inputsize
    w
    m
end


(l::BatchNorm)(x; o...) = batchnorm(x, l.m, l.w; o...)


function BatchNorm(; input::Int, atype=_atype)
    w = Param(atype(bnparams(input)))
    m = bnmoments()
    return BatchNorm(input, w, m)
end


BatchNorm(input; kwargs...) = BatchNorm(input=input; kwargs...)


mutable struct Embedding <: SlothLayer
    vocabsize
    embedsize
    w
end


(l::Embedding)(x) = l.w[:, x]


function Embedding(; vocabsize::Int, embedsize::Int, atype=_atype, init=xavier)
    w = param(embedsize, vocabsize; init=init, atype=atype)
    return Embedding(vocabsize, embedsize, w)
end


Embedding(vocabsize::Int, embedsize::Int; kwargs...) = Embedding(
    vocabsize=vocabsize, embedsize=embedsize; kwargs...)


mutable struct Dropout <: SlothLayer
    p
end


(l::Dropout)(x; p=l.p, kwargs...) = dropout(x, p; kwargs...)


Dropout() = Dropout(0.5)


mutable struct Activation <: SlothLayer
    f
end


(l::Activation)(x) = l.f.(x)


Activation() = Activation(relu)
Relu() = Activation()
ReLU() = Relu()
Tanh() = Activation(tanh)
Sigm() = Activation(sigm)
