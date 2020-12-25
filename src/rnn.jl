struct FlattenRNNHidden <: SlothLayer; end
(l::FlattenRNNHidden)(x) = ndims(x) == 1 ? x : reshape(x, size(x,1), :)


function reset!(r::RNN; h=0.0f0, c=0.f0)
    r.h, r.c = h, c
    nothing
end


function RNN(; input_size::Int, hidden_size::Int, kwargs...)
    RNN(input_size, hidden_size; kwargs...)
end


ReluRNN(; kwargs...) = RNN(; rnnType=:relu, kwargs...)
TanhRNN(; kwargs...) = RNN(; rnnType=:tanh, kwargs...)
SigmRNN(; kwargs...) = RNN(; rnnType=:sigm, kwargs...)
LSTM(; kwargs...) = RNN(; kwargs...)
BiLSTM(; kwargs...) = RNN(; bidirectional=true, kwargs...)