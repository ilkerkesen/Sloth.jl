module Sloth

using Knet
using MAT
using Images, FileIO


const Float = Union{Float32, Float64}
const SlothArray = Union{Array{T}, KnetArray{T}} where T <: Float
const SlothParam = Param{T} where T <: SlothArray
const IntHyperparam = Union{Int, Tuple{Vararg{Int}}}


include("layers.jl")
export Sequential
export Linear
export Conv
export ConvTranspose
export Pool, MaxPool, AvgPool
export BatchNorm
export Dropout
export Embedding
export Activation, Relu, Tanh, Sigm, LeakyRelu
export Flatten, FlattenRNNHidden

include("rnn.jl"); export reset!, ReluRNN, TanhRNN, SigmRNN, LSTM, BiLSTM
# include("beautify.jl"); export show

end # module
