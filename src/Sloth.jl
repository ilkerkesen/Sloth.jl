module Sloth

using Knet
using MAT
using Images, FileIO
import Knet.Ops20: BNMoments
import Knet: atype


const SlothArray = Union{Array{T}, KnetArray{T}} where T <: AbstractFloat
const SlothParam = Param{T} where T <: SlothArray
const SlothWeight = Union{SlothArray, SlothParam}
const SlothBias = Union{SlothArray, SlothParam, AbstractFloat}
const IntHyperparam = Union{Int, Tuple{Vararg{Int}}}
F(x::T) where T <: AbstractFloat = eltype(atype())(x)

include("layers.jl")
include("rnn.jl")
include("beautify.jl")

end # module