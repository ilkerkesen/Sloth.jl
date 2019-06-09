module Sloth

using Knet
import Knet: relu, sigm; export relu, sigm
using MAT
using Images, FileIO

_etype = gpu() >= 0 ? Float32 : Float64
_atype = gpu() >= 0 ? KnetArray{_etype} : Array{_etype}

include("init.jl")
include("data.jl")

include("layers.jl")
export Chain
export Linear
export Dense, FullyConnected
export Conv
export Deconv, ConvTransposed
export BatchNorm
export Embedding
export Dropout
export Relu
export Pool
export Activation
export Relu, ReLU
export Tanh
export Sigm
export Flatten
export FlattenRNNHidden

include("rnn.jl"); export init!, ReluRNN, TanhRNN, SigmRNN, LSTM, BiLSTM
include("optimizers.jl"); export init_optimizers!
include("util.jl"); export get_atype, get_etype, set_atype!, init_opt!
include("vgg.jl")
include("rnnutil.jl")
include("beautify.jl"); export show

end # module
