function init_optimizers!(model, optimizer="Sgd()")
    for par in params(model)
        par.opt = eval(Meta.parse(optimizer))
    end
end
