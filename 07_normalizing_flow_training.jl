using InvertibleNetworks
using LinearAlgebra
using PyPlot
using Flux
using Random
using Distributions

import Flux.Optimise: ADAM, update!
Random.seed!(1234)

PyPlot.rc("font", family="serif");

n_train = 60000;
X_train = sample_banana(n_train);
# size(X_train) #(nx, ny, n_channels, n_samples) Note: we put 2 dimensions as channels

fig = figure(); title(L"x \sim p_x(x)")
scatter(X_train[1,1,1,1:400], X_train[1,1,2,1:400]; alpha=0.4, label = L"x \sim p_{X}(x)");
xlabel(L"x_1"); ylabel(L"x_2");
# xlim(-4,4); ylim(0,30);
legend();

function loss(G, X)
    batch_size = size(X)[end]

    Z, lgdet = G.forward(X)

    l2_loss = 0.5*norm(Z)^2 / batch_size  #likelihood under Normal Gaussian training
    dZ = Z / batch_size                   #gradient under Normal Gaussian training

    G.backward(dZ, Z)  #sets gradients of G wrt output and also logdet terms

    return (l2_loss, lgdet)
end

nx          = 1
ny          = 1

#network architecture
n_in        = 2 #put 2d variables into 2 channels
n_hidden    = 16
levels_L    = 1
flowsteps_K = 10
eps = 1

G = NetworkGlow(n_in, n_hidden, levels_L, flowsteps_K;)
#G = G |> gpu

#training parameters
batch_size = 150
maxiter    = cld(n_train, batch_size)

lr = 9f-4
opt = ADAM(lr)

loss_l2_list    = zeros(maxiter)
loss_lgdet_list = zeros(maxiter)

for ep in 1:eps
    for j = 1:maxiter
        Base.flush(Base.stdout)
        # idx = ((j-1)*batch_size+1) : (j*batch_size)
        idx = rand(1:n_train, batch_size)

        X = X_train[:,:,:,idx]
        #x = x |> gpu

        losses = loss(G, X) #sets gradients of G

        loss_l2_list[j]    = losses[1]
        loss_lgdet_list[j] = losses[2]

        (j%50==0) && println("Iteration=", j, "/", maxiter,
                "; f l2 = ",   loss_l2_list[j],
                "; f lgdet = ",loss_lgdet_list[j],
                "; f nll objective = ",loss_l2_list[j] - loss_lgdet_list[j])
                # "; ", idx)

        for p in get_params(G)
            update!(opt, p.data, p.grad)
        end
    end
end
