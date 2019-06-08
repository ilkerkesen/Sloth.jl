import Knet: relu

@testset "layers" begin
    batchsize, input, output = 2, 3, 4
    x2d = atype(randn(input, batchsize))
    @testset "linear" begin
        layer = y = nothing
        @test (layer=Linear(input, output); true)
        @test (y = layer(x2d); true)
        @test layer.w * x2d .+ layer.b ≈ y
        @test size(layer.w) == (output, input)
        @test layer.inputsize == input
        @test layer.outputsize == output
        @test size(layer.b) == (output, 1)
        @test (layer=Linear(input=input, output=output, bias=false); true)
        @test size(layer.w) == (output, input)
        @test size(layer.b) == ()
        @test layer.b == 0.0
    end

    @testset "dense" begin
        layer = nothing
        @test (layer=Dense(input, output); true)
        @test relu.(layer.w * x2d .+ layer.b) ≈ layer(x2d)
    end

    width, height, kernel = 8, 8, 4
    x4d = atype(rand(width, height, input, batchsize))
    y4d = nothing
    @testset "conv" begin
        layer = nothing
        @test (layer = Conv(input=input, output=output, kernel=kernel); true)
        @test (y4d = layer(x4d); true)
        @test conv4(layer.w, x4d) .+ layer.b ≈ y4d
        @test (l=Conv(input,output,kernel); conv4(l.w,x4d) .+ l.b ≈ l(x4d))
    end

    @testset "deconv" begin
        layer = nothing
        @test (layer=Deconv(input=output, output=input, kernel=kernel); true)
        @test (size(layer(y4d)) == size(x4d))
        @test (layer(y4d) ≈ deconv4(layer.w,y4d) .+ layer.b)
        @test (l=Deconv(output,input,kernel); size(l(y4d)) == size(x4d))
    end

    @testset "pool" begin
        layer = nothing
        @test (layer = Pool(); true)
        @test (layer(x4d) ≈ pool(x4d))
        @test (layer(x4d; mode=1) ≈ pool(x4d; mode=1))
    end

    @testset "batchnorm" begin
        layer = y = nothing
        @test (layer = BatchNorm(input); true)
        @test (y = layer(x2d); y ≈ batchnorm(x2d, layer.m, layer.w))
        @test (l = BatchNorm(input); l(x4d) ≈ batchnorm(x4d, l.m, l.w))
    end

    x1d = rand(1:input, batchsize)
    @testset "embedding" begin
        layer = y1d = nothing
        @test (layer = Embedding(vocabsize=input, embedsize=output); true)
        @test (y1d = layer(x1d); y1d ≈ layer.w[:, x1d])
        @test size(y1d) == (output, batchsize)
    end

    @testset "dropout" begin
        layer = y = nothing
        @test (layer = Dropout(); true)
        @test layer(x2d) == x2d
        @test (Knet.seed!(1); y = layer(x2d; drop=true); true)
        @test (Knet.seed!(1); dropout(x2d, layer.p; drop=true) ≈ y)
    end

    @testset "activation" begin
        layer = y = nothing
        @test (layer = Activation(); true)
        @test layer(x2d) ≈ relu.(x2d)
        @test (l = ReLU(); l(x2d) ≈ relu.(x2d))
        @test (l = Tanh(); l(x2d) ≈ tanh.(x2d))
        @test (l = Sigm(); l(x2d) ≈ sigm.(x2d))
    end
end
