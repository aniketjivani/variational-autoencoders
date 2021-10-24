using Flux, Flux.Data.MNIST, Statistics 
using Flux: throttle, params
using Flux: @epochs, mse, onehotbatch, argmax, crossentropy
using Base.Iterators: repeated, partition
using Plots
using Colors
using Images

function generatedata_circle(r1, r2, N, σ=0.1) 
	φ1 = LinRange(0, 2 * π, N)
	φ2 = LinRange(0, 2 * π, N)
	rx1 = r1 .+ σ * randn(N)
	rx2 = r2 .+ σ * randn(N)
	X1 = [rx1 .* cos.(φ1) rx1 .* sin.(φ1)] 
	X2 = [rx2 .* cos.(φ2) rx2 .* sin.(φ2)] 		
	return X1', X2'
end

X1c, X2c = generatedata_circle(2, 0.5, 100)

X = [X1c X2c]
Y = [zeros(1, size(X1c, 2)) ones(1, size(X2c, 2))]

nHidden = 3

activeFunction = relu

m = Chain(
		Dense(size(X,1), nHidden, activeFunction), 
		Dense(nHidden, nHidden, activeFunction),
		Dense(nHidden, 1, activeFunction)
		)

loss(x, y) = mse(m(x), y)
iters = 10000

dataset = repeated((X, Y), iters)

opt = ADAM()
evalcb = () -> @show([loss(X, Y)])

Flux.train!(loss, params(m), dataset, opt; cb = throttle(evalcb, 0.5))

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

x1range = range(-3; stop=3, length=200)
x2range = range(-3; stop=3, length=200)


display_decision_boundaries(X1c, X2c, m, x1range, x2range)
plot!(; title="Loss = $(round(loss(X,Y), digits=4))")

@show m.layers[1].W
@show m.layers[1].b
@show m.layers[1].σ;
