mutable struct Linear
    w
    b
end


function (l::Linear)(x)
    y = l.w * mat(x)
    y = l.b == nothing ? y : y .+ l.b
end


function Linear(input_dim::Int, output_dim::Int;
                atype=_atype, init=xavier, bias=true)
    w, b = initwb(input_dim, output_dim, atype, init)
    b = bias ? b : nothing
    return Linear(w, b)
end


mutable struct FullyConnected
    w
    b
    activate
end


function (l::FullyConnected)(x)
    y = l.w * mat(x)
    y = l.b == nothing ? y : y .+ l.b
    l.activate.(y)
end


function FullyConnected(input_dim::Int, output_dim::Int, activate=relu;
                        atype=_atype, init=xavier, bias=true)
    w, b = initwb(input_dim, output_dim, atype, init)
    b = bias ? b : nothing
    return FullyConnected(w, b, activate)
end


mutable struct Conv
    w
    b
end


function (l::Conv)(x; o...)
    y = conv4(l.w, x; o...)
    y = l.b == nothing ? y : y .+ l.b
end


function Conv(ci::Int, co::Int, k::Int;
              atype=_atype, initw=xavier, initb=zeros, bias=true)
    w = param(k, k, ci, co; atype=atype, init=initw)
    b = bias ? param(1, 1, co, 1; atype=atype, init=initb) : nothing
    return Conv(w, b)
end


mutable struct Deconv
    w
    b
end


function (l::Deconv)(y; o...)
    x = deconv4(l.w, y; o...)
    x = l.b == nothing ? x : x .+ l.b
end


function Deconv(ci::Int, co::Int, k::Int;
                atype=_atype, initw=xavier, initb=zeros, bias=false)
    w = param(k, k, co, ci; atype=atype, init=initw)
    b = bias ? param(1, 1, co, 1; atype=atype, init=initb) : nothing
    return Deconv(w, b)
end


mutable struct BatchNorm
    w
    m
end


function (l::BatchNorm)(x; training=true)
    batchnorm(x, l.m, l.w; training=training)
end


function BatchNorm(dim::Int; atype=_atype)
    w = Param(atype(bnparams(dim)))
    m = bnmoments()
    return BatchNorm(w, m)
end


mutable struct Embedding
    w
end


function (l::Embedding)(x)
    l.w[:, x]
end


function Embedding(vocabsize::Int, embedsize::Int; atype=_atype, init=xavier)
    w = param(embedsize, vocabsize; init=init, atype=atype)
    return Embedding(w)
end
