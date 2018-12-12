mutable struct Projection
    w
end


(l::Projection)(x) = l.w * mat(x)


function Projection(input_dim::Int, output_dim::Int; atype=_atype, init=xavier)
    w = param(output_dim, input_dim; atype=atype, init=init)
    return Projection(w)
end


mutable struct Linear
    w
    b
end


(l::Linear)(x) = l.w * mat(x) .+ l.b


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


(l::FullyConnected)(x) = activate.(l.w * mat(x) .+ l.b)


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
    conv4(l.w, x; o...) .+ l.b
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
    deconv4(l.w, y; o...)
end


function Deconv(ci::Int, co::Int, k::Int;
                atype=_atype, initw=xavier, initb=zeros, bias=false)
    w = param(k, k, co, ci; atype=atype, init=initw)
    b = bias ? param(1, 1, co, 1; atype=atype, init=initb) : nothing
    return Deconv(w, b)
end


mutable struct Embedding
    w
end


(l::Embedding)(x) = l.w[:, x]


function Embedding(vocabsize::Int, embedsize::Int; atype=_atype, init=xavier)
    w = param(embedsize, vocabsize; init=init, atype=atype)
    return Embedding(w)
end
