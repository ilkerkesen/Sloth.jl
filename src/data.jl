function resize_center_crop(image, len=224)
    a0 = image
    new_size = ntuple(i->div(size(a0,i)*len,minimum(size(a0))),2)
    a1 = Images.imresize(a0, new_size)
    i1 = div(size(a1,1)-len,2)
    j1 = div(size(a1,2)-len,2)
    b1 = a1[i1+1:i1+len,j1+1:j1+len]
    RGB{Float64}.(b1)
end 


function vgg_image(cropped, average_image=F(0.0))
    b1 = convert(Array{FixedPointNumbers.Normed{UInt8,8},3}, channelview(cropped))
    c1 = permutedims(b1, (3,2,1))
    d1 = convert(Array{Float32}, c1)
    e1 = reshape(d1[:,:,1:3], (224,224,3,1))
    f1 = (255 * e1 .- average_image)
    g1 = permutedims(f1, [2,1,3,4])
end


function preprocess_image(img, average_image=F(0.0))
    vgg_image(resize_center_crop(img), average_image)
end