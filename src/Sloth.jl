module Sloth

include("init.jl")

include("layers.jl")
export Projection
export Linear
export FullyConnected
export Conv
export Embedding
export Deconv

include("rnnutil.jl")

end # module
