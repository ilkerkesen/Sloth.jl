function load_pytorch_lstm(wih, bih, whh, bhh)
    hidden = div(size(wih, 2), 4)

    _helper0(w, i) = w[:, (i-1)*hidden+1:i*hidden]
    _helper1(w, b, i) = (_helper0(w, i), _helper0(b, i))

    wxi, bxi = _helper1(wih, bih, 1)
    wxf, bxf = _helper1(wih, bih, 2)
    wxu, bxu = _helper1(wih, bih, 3)
    wxo, bxo = _helper1(wih, bih, 4)

    whi, bhi = _helper1(whh, bhh, 1)
    whf, bhf = _helper1(whh, bhh, 2)
    whu, bhu = _helper1(whh, bhh, 3)
    who, bho = _helper1(whh, bhh, 4)

    _helper3(w) = reshape(w, 1, length(w))
    knet_wb = mapreduce(
        _helper3,
        hcat,
        [wxi, wxf, wxu, wxo,
         whi, whf, whu, who,
         bxi, bxf, bxu, bxo,
         bhi, bhf, bhu, bho])
    return reshape(knet_wb, 1, 1, length(knet_wb))
end



function load_tensorflow_lstm(tf_w, tf_b)
    hidden = div(size(tf_w, 1), 4)
    tf_b = reshape(tf_bias, 4hidden, 1)

    _helper0(w, i) = w[(i-1)*hidden+1:i*hidden, :]
    _helper1(i) = (_helper0(tf_w, i), _helper0(tf_b, i))

    input_w, input_b = _helper1(1)
    update_w, update_b = _helper1(2)
    forget_w, forget_b = _helper1(3)
    output_w, output_b = _helper1(4)

    _helper2(w) = (w[1:hidden, :], w[hidden+1:end, :])
    wxi, whi = _helper2(input_w)
    wxu, whu = _helper2(update_w)
    wxf, whf = _helper2(forget_w)
    wxo, who = _helper2(output_w)

    _helper3(w) = ()
    bi = input_b/2
    bu = update_b/2
    bf = forget_b/2
    bo = output_b/2

    _helper4(w) = reshape(w, 1, length(w))
    knet_wb = mapreduce(
        _helper4,
        hcat,
        [wxi, wxf, wxu, wxo,
         whi, whf, whu, who,
         bi, bf, bu, bo,
         bi, bf, bu, bo])
    return reshape(knet_wb, 1, 1, length(knet_wb))
end
