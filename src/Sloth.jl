module Sloth

using Knet
using MAT
using Images, FileIO

_etype = gpu() >= 0 ? Float32 : Float64
_atype = gpu() >= 0 ? KnetArray{_etype} : Array{_etype}

include("init.jl")
include("data.jl")

include("layers.jl")
export Chain
export Linear
export FullyConnected
export Dense
export Conv
export Deconv
export BatchNorm
export Embedding
export Dropout
export Relu
export Pool
export Activation
export Relu, ReLU
export Tanh
export Sigm

include("optimizers.jl")
export init_optimizers!

include("util.jl")
export get_atype
export init_opt!

include("vgg.jl")
include("rnnutil.jl")


end # module
