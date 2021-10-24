### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 62f910de-3462-11ec-0510-b983317ce54a
begin
	using Pkg
	Pkg.activate("../Project.toml")
end

# ╔═╡ a2e307c3-c6c9-4689-9b16-0bd269a62c9e
begin
	using Flux, Flux.Data.MNIST, Statistics 
	using Flux: throttle, params
	using Flux: @epochs, mse, onehotbatch, argmax, crossentropy
	using Base.Iterators: repeated, partition
	using Plots
	using Colors
	using Images
end

# ╔═╡ d473bd52-bc87-4c27-89ce-e54a5e63ab8a
using PlutoUI

# ╔═╡ e6a6e6e5-9b31-480e-badb-a8a65be9ad08
md"""
## References:

1) [Chapter 11, Variational Autoencoders by Hugh Murrell](https://nextjournal.com/DeepLearningNotes/Ch11Autoencoders) - trying to reproduce the above in Pluto.

2) [ACDME - Variational Autoencoder](https://docs.juliahub.com/ADCME/b8Ld2/0.7.3/vae/)

3) EECS 505 material

4) Flux Model Zoo (look for `vae_mnist` under `vision`). [Github](https://github.com/FluxML/model-zoo)
"""

# ╔═╡ 7d77c20a-6a38-4aac-aa41-443eeaf34b9d
md"""
## Display and helper functions
"""

# ╔═╡ 9d5cec86-b665-4b24-9660-57f47f0c4f13
my_norm(v) = sqrt(sum(v.^2))

# ╔═╡ a845a832-1088-4de9-992e-31d7c402e15c
function accuracy(lm, range, centroids)
    sum([ argmin(mapslices(my_norm,centroids.-lm(X[:,i]).data,dims=1))[2]-1==Y[i] 
        for i in range ]) / length(range)
end

# ╔═╡ b9dfee1a-a5ac-4cb8-a7fb-6931cec23dff
function computeClasses(Y)
    sort(unique(Y))
end

# ╔═╡ 2ec27fff-e3c0-4093-8c67-2ec581a94f16
function computeCentroids(lm, data, classes) 
    centroids = zeros((2,length(classes)))
    for k in 1:length(data)
        x = first(data[k])
        y = last(data[k])
        for i in 1:length(classes)
            points = lm(x[:,y.==classes[i]])
            centroids[:,i] += Flux.data.(mapslices(mean,points,dims=2))
        end
    end;
    centroids ./= length(data)
    return centroids
end

# ╔═╡ 52a6b4c8-989c-4828-9f6d-5078c5029f2d
function plotLatentSpace(lm, data, classes, width=0)
    centroids = computeCentroids(lm, data, classes)
    colors = distinguishable_colors(12, [RGB(1,1,1)])[3:12]
    if ( width > 0 )
        p = scatter(xlim=(-width,width),ylim=(-width,width))
    else
        p = scatter()
    end
    for lab = 1:length(classes)
        scatter!([centroids[1,lab]],[centroids[2,lab]],
                label=classes[lab],markersize=[9],markercolor=[colors[lab]])
    end
    for k in 1:length(data)
        x = first(data[k])
        y = last(data[k])
        for i in 1:length(classes)
            points = Flux.data(lm(x[:,y.==classes[i]]))
            scatter!(points[1,:],points[2,:],label="",
               markercolor=[colors[i]])
        end
    end
    for lab = 1:length(classes)
        scatter!([centroids[1,lab]],[centroids[2,lab]],
                label="",markersize=[9],markercolor=[colors[lab]])
    end
    scatter!(centroids[1,:],centroids[2,:],label="centroids")
    return p
end

# ╔═╡ c25f4c25-959d-4077-9ce5-fa0b26641b17
md"""
## Simple Autoencoder
"""

# ╔═╡ 3db0aa21-7a71-4d21-bc7f-f236b6dde600
# begin
# 	# Latent dimensionality, # hidden units.
# 	Dz, Dh = 2, 500
	
# 	# encoder
# 	g = Chain(Dense(28^2, Dh, tanh), Dense(Dh, Dz))
# 	# decoder
# 	f = Chain(Dense(Dz, Dh, tanh), Dense(Dh, 28^2, σ))
# 	# model
# 	sae = Chain(g,f) |> gpu
# 	# loss
# 	loss(X,Y) = mse(sae(X), X)
# 	# callback
# 	evalcb = throttle(() -> ( p=rand(1:N, M); @show(loss(Float32.(X[:, p]),Y[p]))), 20) 
# 	# optimization
# 	opt = ADAM()
# 	# parameters
# 	ps = params(sae);
# end

# ╔═╡ 9b6d280f-0994-4caf-8a07-bdf7f03cfda9
# p=rand(1:N, M)

# ╔═╡ 122fd414-1bbc-45cc-988a-bf4dda9ee5ae
# loss(Float32.(X[:,p]), Y[p])

# ╔═╡ 04ce4d2a-8b37-4e33-af12-15e2f05d9fb5
# trainDataGPU = gpu.(trainData);

# ╔═╡ 72b3993c-55fb-4a34-bf6b-87e538ab8b4c
# trainDataGPU

# ╔═╡ b78786c3-df99-46e6-a574-3c9a101f2bb6
# @epochs 2 Flux.train!(loss, params(sae), trainDataGPU, opt, cb = evalcb)

# ╔═╡ Cell order:
# ╟─e6a6e6e5-9b31-480e-badb-a8a65be9ad08
# ╠═62f910de-3462-11ec-0510-b983317ce54a
# ╠═a2e307c3-c6c9-4689-9b16-0bd269a62c9e
# ╠═d473bd52-bc87-4c27-89ce-e54a5e63ab8a
# ╠═7d77c20a-6a38-4aac-aa41-443eeaf34b9d
# ╠═9d5cec86-b665-4b24-9660-57f47f0c4f13
# ╠═a845a832-1088-4de9-992e-31d7c402e15c
# ╠═b9dfee1a-a5ac-4cb8-a7fb-6931cec23dff
# ╠═2ec27fff-e3c0-4093-8c67-2ec581a94f16
# ╠═52a6b4c8-989c-4828-9f6d-5078c5029f2d
# ╠═c25f4c25-959d-4077-9ce5-fa0b26641b17
# ╠═3db0aa21-7a71-4d21-bc7f-f236b6dde600
# ╠═9b6d280f-0994-4caf-8a07-bdf7f03cfda9
# ╠═122fd414-1bbc-45cc-988a-bf4dda9ee5ae
# ╠═04ce4d2a-8b37-4e33-af12-15e2f05d9fb5
# ╠═72b3993c-55fb-4a34-bf6b-87e538ab8b4c
# ╠═b78786c3-df99-46e6-a574-3c9a101f2bb6
