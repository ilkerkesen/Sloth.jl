function get_atype(array)
    atype = typeof(value(array)) <: KnetArray ? KnetArray : Array
    etype = eltype(value(array))
    return atype{etype}
end


function get_atype(layer::T) where T <: SlothLayer
    get_atype(layer.w)
end


get_atype() = _atype
get_etype() = _etype


function set_atype!(atype)
    _atype, _etype = atype, eltype(atype)
end


function init_opt!(model, optimizer="SGD()")
    for par in params(model)
        par.opt = eval(Meta.parse(optimizer))
    end
end
