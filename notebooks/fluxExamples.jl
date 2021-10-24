### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 5295694c-344f-11ec-2842-d7207c32a925
begin
	using Pkg
	Pkg.activate("../Project.toml")
end

# ╔═╡ 57884ba0-1209-4ebe-aaaf-b03edd5814d8
begin
	using Flux
	using Flux.Data: MNIST 
	using Statistics
	using Flux: throttle, params
	using Flux: @epochs, @functor
	using Flux: onehotbatch, argmax
	using Flux.Losses: logitbinarycrossentropy, crossentropy, mse
	using Flux: chunk
	using ProgressMeter: Progress, next!
	using Base.Iterators: repeated, partition
	using Plots
	using Colors
	using Images
end

# ╔═╡ e5ae0713-8bb0-4568-8eda-3cf8fdc5cef0
using PlutoUI

# ╔═╡ f47b82e5-1110-4bfa-9ac1-637076c6faf2
md"""
## Some Examples of using Flux (reference: EECS505, other code from this forked repo)
"""

# ╔═╡ 09fea98d-aaf0-4971-a81f-90b50ef04b47
md"""
## Classifier for data belonging to two circles
"""

# ╔═╡ 3e79f9cc-f01c-4622-ade9-92d6aaa954e1
md"""
Inspect a few images.
"""

# ╔═╡ 6ebac32e-ad7d-4321-be2f-b61127c47771
# [img(X[:, i]) for i in 6:10]

# ╔═╡ f74b47d4-7dff-40a1-b086-26bf8b5c88d7
function generatedata_circle(r1, r2, N, σ=0.1) 
	φ1 = LinRange(0, 2 * π, N)
	φ2 = LinRange(0, 2 * π, N)
	rx1 = r1 .+ σ * randn(N)
	rx2 = r2 .+ σ * randn(N)
	X1 = [rx1 .* cos.(φ1) rx1 .* sin.(φ1)] 
	X2 = [rx2 .* cos.(φ2) rx2 .* sin.(φ2)] 		
	return X1', X2'
end

# ╔═╡ b0986b17-bd42-4fb4-a3f1-54a58e52593c
X1c, X2c = generatedata_circle(2, 0.5, 100)

# ╔═╡ 5b13fe75-6636-4ee2-9396-98ba53cedadd
begin	
	p1 = scatter(
				X1c[1, :], 
				X1c[2, :]; 
				color="red", 
				label="Class 1", 
				aspectratio=:equal,
				)
	scatter!(
			X2c[1, :], 
			X2c[2, :]; 
			color="blue", 
			label="Class 2", 
			legend=:bottomright,
			)
end

# ╔═╡ c1b9423c-c79b-46b5-8f19-c252aaa7473b
X = [X1c X2c]

# ╔═╡ 100900ac-2b1f-492f-9a50-14f4fc5506cb
Y = [zeros(1, size(X1c, 2)) ones(1, size(X2c, 2))]

# ╔═╡ d8ebaeef-27a9-4530-a19f-7987919178df
@bind nHidden Slider(2:10, default=3, show_value=true)

# ╔═╡ b2b1c6a3-acbe-4a39-8f1c-378de73a96ee
nHidden

# ╔═╡ 24d87786-904c-4c2e-9c97-e3574521f49e
activeFunction = relu

# ╔═╡ 123965a9-a4e3-4194-b079-d32bd941aa17
m = Chain(
		Dense(size(X,1), nHidden, activeFunction), 
		Dense(nHidden, nHidden, activeFunction),
		Dense(nHidden, 1, activeFunction)
		)

# ╔═╡ bdde6e5e-72ad-4071-ab98-15724f72ba86
md"Output has size $(size(m(X), 2))"

# ╔═╡ 8f5068ce-6301-4234-b284-9ec03b32a8e0
loss(x, y) = mse(m(x), y)

# ╔═╡ d3b9eaa8-be71-4b83-91fd-2662ce842259
iters = 10000

# ╔═╡ f373f536-9929-40a4-ad0d-b2adb606376a
md"""
We package the dataset into `(X, Y)` tuples.
"""

# ╔═╡ ea9abbbe-36a8-46d6-8987-0f4840e4b804
dataset = repeated((X, Y), iters)

# ╔═╡ 83b8153e-8b06-4f25-ad4d-b92c5ed21884
begin
	opt = ADAM()
	evalcb = () -> @show([loss(X, Y)])
end

# ╔═╡ 2c8025b6-cedc-463a-9625-551c4e402052
with_terminal() do
	Flux.train!(loss, params(m), dataset, opt; cb = throttle(evalcb, 0.5))
end

# ╔═╡ b36530a1-ad38-4178-a914-51cc59c1d294
function display_decision_boundaries(X1c, X2c, m, x1range, x2range, τ=0.0)
    D = [m([x1; x2])[1] for x2 in x2range, x1 in x1range] 
    heatmap(
			x1range, 
			x2range, 
			sign.(D .- τ); 
			color=:grays, 
			xlim = [minimum(x1range),maximum(x1range)], 
			ylim = [minimum(x2range),maximum(x2range)],
			) 
    scatter!(
			X1c[1, :], 
			X1c[2, :], 
			color="red", 
			label="Class 1", 
			aspectratio=:1.0,
			)
    scatter!(
			X2c[1, :], 
			X2c[2, :], 
			color="blue", 
			label="Class 2",
			)
end

# ╔═╡ 09081010-4a09-42fc-8658-f02d7c35a4e9
x1range = range(-3; stop=3, length=200)

# ╔═╡ a184440d-621b-42ff-b4a9-a596b248acbc
x2range = range(-3; stop=3, length=200)

# ╔═╡ e7f6169f-aa66-4a43-9831-227507047156
begin
	display_decision_boundaries(X1c, X2c, m, x1range, x2range)
	plot!(; title="Loss = $(round(loss(X,Y), digits=4))")
end

# ╔═╡ 6eccef88-ac15-4538-93f7-7e807e4891fd
with_terminal() do
	@show m.layers[1].W
	@show m.layers[1].b
	@show m.layers[1].σ;
end

# ╔═╡ b22cf1c6-aed7-43b4-9687-ad04dcd983d4
md"""
## Classifier for MNIST Data
"""

# ╔═╡ 6159bf70-b4d8-4bcb-ba24-4d3502d98e7c
imgs = MNIST.images()

# ╔═╡ 1df18a22-46e5-4c8a-a54e-88fe342ac39c
typeof(imgs)

# ╔═╡ 5a752624-23ad-41ab-8c53-2caf0ded9aa9


# ╔═╡ 91b6edde-c595-4bd4-99c8-dd07752c7a7f


# ╔═╡ 08bb5a70-39e0-46d3-81f7-71e27a87a4e0
function convert_to_image(x, y_size)
    Gray.(permutedims(vcat(reshape.(chunk(x |> cpu, y_size), 28, :)...), (2, 1)))
end

# ╔═╡ 54bc078c-8069-4d2c-a578-7b9d1d2fc367


# ╔═╡ 2cd4d120-f02d-4667-8977-a40e800e9a3a


# ╔═╡ e1f0af51-2519-4846-82d5-76e02f1bd938


# ╔═╡ a8c4ae86-2b50-4f7d-bfd3-763e2d5d2321


# ╔═╡ 10de7adb-52ec-4e24-bfa5-e5652cb3f226


# ╔═╡ Cell order:
# ╟─f47b82e5-1110-4bfa-9ac1-637076c6faf2
# ╠═5295694c-344f-11ec-2842-d7207c32a925
# ╠═57884ba0-1209-4ebe-aaaf-b03edd5814d8
# ╠═e5ae0713-8bb0-4568-8eda-3cf8fdc5cef0
# ╠═09fea98d-aaf0-4971-a81f-90b50ef04b47
# ╟─3e79f9cc-f01c-4622-ade9-92d6aaa954e1
# ╠═6ebac32e-ad7d-4321-be2f-b61127c47771
# ╠═f74b47d4-7dff-40a1-b086-26bf8b5c88d7
# ╠═b0986b17-bd42-4fb4-a3f1-54a58e52593c
# ╟─5b13fe75-6636-4ee2-9396-98ba53cedadd
# ╠═c1b9423c-c79b-46b5-8f19-c252aaa7473b
# ╠═100900ac-2b1f-492f-9a50-14f4fc5506cb
# ╠═d8ebaeef-27a9-4530-a19f-7987919178df
# ╠═b2b1c6a3-acbe-4a39-8f1c-378de73a96ee
# ╠═24d87786-904c-4c2e-9c97-e3574521f49e
# ╠═123965a9-a4e3-4194-b079-d32bd941aa17
# ╠═bdde6e5e-72ad-4071-ab98-15724f72ba86
# ╠═8f5068ce-6301-4234-b284-9ec03b32a8e0
# ╠═d3b9eaa8-be71-4b83-91fd-2662ce842259
# ╟─f373f536-9929-40a4-ad0d-b2adb606376a
# ╠═ea9abbbe-36a8-46d6-8987-0f4840e4b804
# ╠═83b8153e-8b06-4f25-ad4d-b92c5ed21884
# ╠═2c8025b6-cedc-463a-9625-551c4e402052
# ╠═b36530a1-ad38-4178-a914-51cc59c1d294
# ╠═09081010-4a09-42fc-8658-f02d7c35a4e9
# ╠═a184440d-621b-42ff-b4a9-a596b248acbc
# ╠═e7f6169f-aa66-4a43-9831-227507047156
# ╠═6eccef88-ac15-4538-93f7-7e807e4891fd
# ╟─b22cf1c6-aed7-43b4-9687-ad04dcd983d4
# ╠═6159bf70-b4d8-4bcb-ba24-4d3502d98e7c
# ╠═1df18a22-46e5-4c8a-a54e-88fe342ac39c
# ╠═5a752624-23ad-41ab-8c53-2caf0ded9aa9
# ╠═91b6edde-c595-4bd4-99c8-dd07752c7a7f
# ╠═08bb5a70-39e0-46d3-81f7-71e27a87a4e0
# ╠═54bc078c-8069-4d2c-a578-7b9d1d2fc367
# ╠═2cd4d120-f02d-4667-8977-a40e800e9a3a
# ╠═e1f0af51-2519-4846-82d5-76e02f1bd938
# ╠═a8c4ae86-2b50-4f7d-bfd3-763e2d5d2321
# ╠═10de7adb-52ec-4e24-bfa5-e5652cb3f226
