function init!(r::RNN; h0=0, c0=0)
    r.h, r.c = h0, c0
    nothing
end


ReluRNN(; input::Int, output::Int, kwargs...) = RNN(
    input, output; rnnType=:relu, kwargs...)


TanhRNN(; input::Int, output::Int, kwargs...) = RNN(
    input, output; rnnType=:tanh, kwargs...)


SigmRNN(; input::Int, output::Int, kwargs...) = RNN(
    input, output; rnnType=:sigm, kwargs...)


LSTM(; input::Int, output::Int, kwargs...) = RNN(
    input, output; rnnType=:lstm, kwargs...)


BiLSTM(; input::Int, output::Int, kwargs...) = LSTM(;
    input=input, output=output, bidirectional=true, kwargs...)
