function get_atype(array)
    atype = typeof(value(array)) <: KnetArray ? KnetArray : Array
    etype = eltype(value(array))
    return atype{etype}
end


function init_opt!(model, optimizer="SGD()")
    for par in params(model)
        par.opt = eval(Meta.parse(optimizer))
    end
end
