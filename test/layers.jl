import Knet: relu, atype
import Sloth: F


@testset "layers" begin
    batch_size, in_features, out_features = 2, 3, 4
    x2d = atype(randn(in_features, batch_size))
    @testset "linear" begin
        import Sloth: Linear
        layer = Linear(in_features=in_features, out_features=out_features)
        @test layer.weight * x2d .+ layer.bias ≈ layer(x2d)
        @test size(layer.weight) == (out_features, in_features)
        @test layer.in_features == in_features
        @test layer.out_features == out_features
        @test size(layer.bias) == (out_features, 1)

        layer = Linear(
            in_features=in_features,
            out_features=out_features,
            bias=false)
        @test size(layer.bias) == ()
        @test layer.bias == F(0.0)
    end

    width, height, in_channels, out_channels, kernel_size = 8, 8, 2, 3, 5
    x4d = atype(rand(width, height, in_channels, batch_size))
    @testset "conv" begin
        import Sloth: Conv, ConvTranspose

        layer = Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size)
        @test conv4(layer.weight, x4d) .+ layer.bias ≈ layer(x4d)

        y4d = layer(x4d)
        layer = ConvTranspose(
            in_channels=out_channels,
            out_channels=in_channels,
            kernel_size=kernel_size)
        @test (size(layer(y4d)) == size(x4d))
        @test (layer(y4d) ≈ deconv4(layer.weight ,y4d) .+ layer.bias)
    end

    @testset "pool" begin
        import Sloth: MaxPool, AvgPool, PaddedAvgPool
        layer = MaxPool()
        @test (layer(x4d) ≈ pool(x4d))
        layer = AvgPool()
        @test (layer(x4d) ≈ pool(x4d, mode=2))
        layer = PaddedAvgPool()
        @test (layer(x4d) ≈ pool(x4d, mode=1))
    end

    @testset "batchnorm" begin
        import Sloth: BatchNorm
        layer = BatchNorm(num_features=in_features)
        @test layer(x2d) ≈ batchnorm(x2d, layer.moments, layer.weight)
        layer = BatchNorm(num_features=in_channels)
        @test layer(x4d) ≈ batchnorm(x4d, layer.moments, layer.weight)
    end

    x1d = rand(1:in_features, batch_size)
    @testset "embedding" begin
        import Sloth: Embedding
        layer = Embedding(
            num_embeddings=in_features,
            embedding_dim=out_features)
        @test (layer(x1d) ≈ layer.weight[:, x1d])
    end

    @testset "dropout" begin
        import Sloth: Dropout
        layer = Dropout()
        @test layer(x2d) == x2d
        @test begin
            Knet.seed!(1); y1 = layer(x2d; drop=true)
            Knet.seed!(1); y2 = dropout(x2d, layer.p; drop=true)
            y1 ≈ y2
        end
    end

    @testset "activation" begin
        import Sloth: Relu, Tanh, Sigm, LeakyRelu
        layer = Relu(); @test layer(x2d) ≈ relu.(x2d)
        layer = Tanh(); @test layer(x2d) ≈ tanh.(x2d)
        layer = Sigm(); @test layer(x2d) ≈ sigm.(x2d)
        layer = LeakyRelu(); F = eltype(atype())
        @test layer.α > F(0)
        @test layer(x2d) ≈ max.(F(0), x2d) + layer.α .* min.(F(0), x2d)
    end

    input_size, hidden_size = 3, 5
    @testset "recurrent" begin
        import Sloth: ReluRNN, TanhRNN, LSTM, GRU, BiLSTM
        relurnn = ReluRNN(input_size=input_size, hidden_size=hidden_size)
        tanhrnn = TanhRNN(input_size=input_size, hidden_size=hidden_size)
        lstm = LSTM(input_size=input_size, hidden_size=hidden_size)
        gru = GRU(input_size=input_size, hidden_size=hidden_size)
        bilstm = BiLSTM(input_size=input_size, hidden_size=hidden_size)

        @test relurnn.mode == 0
        @test tanhrnn.mode == 1
        @test lstm.mode == 2
        @test gru.mode == 3
        @test lstm.inputSize == input_size
        @test lstm.hiddenSize == hidden_size
        @test lstm.direction == 0
        @test bilstm.mode == 2
        @test bilstm.direction == 1
    end

    @testset "sequential" begin
        import Sloth: Sequential, Linear
        net = Sequential(
            Linear(in_features=in_features, out_features=out_features),
            Linear(in_features=out_features, out_features=out_features))
        @test net(x2d) ≈ net.layers[2](net.layers[1](x2d))
    end
end
