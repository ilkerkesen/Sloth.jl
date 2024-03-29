module Sloth

using Knet
using MAT
using Images, FileIO
import Knet.Ops20: BNMoments
import Knet: atype


const SlothArray = Union{Array{T}, KnetArray{T}, CuArray{T}} where T <: AbstractFloat
const SlothParam = Param{T} where T <: SlothArray
const SlothWeight = Union{SlothArray, SlothParam}
const SlothBias = Union{SlothArray, SlothParam, AbstractFloat}
const IntHyperparam = Union{Int, Tuple{Vararg{Int}}}
F(x::T) where T <: AbstractFloat = eltype(atype())(x)
dir() = abspath(@__DIR__)
dir(args...) = abspath(joinpath(dir(), args...))

include("layers.jl")
include("rnn.jl")
include("beautify.jl")
include("data.jl")
include("vgg.jl")

end # module