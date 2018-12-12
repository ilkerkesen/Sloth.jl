module Sloth

using Knet
_etype = gpu() >= 0 ? Float32 : Float64
_atype = gpu() >= 0 ? KnetArray{_etype} : Array{_etype}

include("init.jl")

include("layers.jl")
export Linear
export FullyConnected
export Conv
export Deconv
export BatchNorm
export Embedding

include("optimizers.jl")
export init_optimizers!

include("rnnutil.jl")

end # module
