module VAETools
    export get_data, Encoder, Decoder, reconstruct, model_loss

    using Flux
    using Flux.Data.MNIST
    using Flux.Data: DataLoader
    using Flux: onehotbatch
    using Flux.Losses: logitbinarycrossentropy


    function get_data(batch_size)
        imgs = MNIST.images()
        labels = MNIST.labels()
        
        xtrain = hcat(float.(reshape.(imgs, :))...)
        ytrain = onehotbatch(labels, 0:9)

        DataLoader((xtrain, ytrain), batchsize=batch_size, shuffle=true)
    end

    struct Encoder
        linear
        μ
        logσ
    end

    @functor Encoder

    Encoder(input_dim::Int, latent_dim::Int, hidden_dim::Int) =
    Encoder(
        Dense(input_dim, hidden_dim, tanh),     # linear
        Dense(hidden_dim, latent_dim),          # μ
        Dense(hidden_dim, latent_dim),          # logσ
    )

    function (encoder::Encoder)(x)
        h = encoder.linear(x)
        encoder.μ(h), encoder.logσ(h)
    end

    Decoder(input_dim::Int, latent_dim::Int, hidden_dim::Int) = 
    Chain(
        Dense(latent_dim, hidden_dim, tanh),
        Dense(hidden_dim, input_dim)
    )

    function reconstruct(encoder, decoder, x, device)
        μ, logσ = encoder(x)
        z = μ + device(randn(Float32, size(logσ))) .* exp.(logσ)
        μ, logσ, decoder(z)
    end

    function model_loss(encoder, decoder, λ, x, device)
        μ, logσ, decoder_z = reconstuct(encoder, decoder, x, device)
        len = size(x)[end]
        # KL-divergence
        kl_q_p = 0.5f0 * sum(@. (exp(2f0 * logσ) + μ^2 -1f0 - 2f0 * logσ)) / len
    
        logp_x_z = -logitbinarycrossentropy(decoder_z, x, agg=sum) / len
        # regularization
        reg = λ * sum(x->sum(x.^2), Flux.params(decoder))
        
        -logp_x_z + kl_q_p + reg
    end


end