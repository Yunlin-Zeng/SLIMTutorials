using InvertibleNetworks
using LinearAlgebra
using Distributions
using PyPlot
using Flux
using Random

PyPlot.rc("font", family="serif"); 

# Turn off interactive plotting
PyPlot.ioff()

seed = parse(Int, ARGS[1])
println("seed = ", seed)
s = ARGS[2] 
s_clean = replace(s, r"[\(\)]" => "")  # Remove parentheses
numbers_str = split(s_clean, ", ")  # Split the string by comma and space
n_train, n_noise, repeat_train, repeat_noise = parse.(Int, numbers_str)
println(numbers_str)

### Define multivarate Gaussian distribtion

# Model and data dimension
dim_model = 2
dim_data  = 2

# Prior distribution
μ_x = 3.14f0*ones(Float32, dim_model)
σ_x = 1.0f0
Σ_x = σ_x^2*I
Λ_x = inv(Σ_x)

π_x = MvNormal(μ_x, Σ_x)

# Distribution of noise
μ_ϵ = 0.0f0*ones(Float32, dim_data)
σ_ϵ = 0.2f0
Σ_ϵ = σ_ϵ^2*I
Λ_ϵ = inv(Σ_ϵ)

π_ϵ = MvNormal(μ_ϵ, Σ_ϵ)

# Forward operator
Random.seed!(seed)
A = randn(Float32, dim_data, dim_model) #/ sqrt(dim_model*1.0f0)

# Analytic posterior distribution
Σ_post = inv(Λ_x + A'*Λ_ϵ*A)
R = cholesky(Σ_post, check=false).L
standard_normal = MvNormal(zeros(Float32, dim_model), 1.0f0*I)

function post_dist_sample(y)
  return R*rand(standard_normal) + post_dist_mean(reshape(y_test,:,size(y_test)[end]))[:]
end

function post_dist_mean(y)
    Σ_post*(A'*Λ_ϵ*(y) + Λ_x*μ_x)
end


### Prepare training data

nx = 1  # Express n-Dim variables as n channels
ny = 1  # Express n-Dim variables as n channels
n_in = dim_model  # Express n-Dim variables as n channels
# n_train = 35
# n_noise = n_train
# 
# if n_noise/n_train > 1
#     repeat_train = Integer(n_noise/n_train)
#     repeat_noise = 1
# else
#     repeat_train = repeat_noise = 2
# end

X_train1 = rand(π_x, n_train)
X_train = reshape(X_train1, nx, ny, dim_model, n_train)
e_train = rand(π_ϵ, n_noise)

X_train1 = cat([X_train1 for _ in 1:repeat_train]..., dims=2)
X_train = cat([X_train for _ in 1:repeat_train]..., dims=4)
e_train = cat([e_train for _ in 1:repeat_noise]..., dims=2)

Y_train = reshape(A * X_train1 + e_train, nx, ny, dim_model, n_noise*repeat_noise);

plot_num = min(100, n_train)


### Training an NF

function loss(H, X, Y)
    batch_size = size(X)[end]
    
    Zx, Zy, lgdet = H.forward(X, Y)
    l2_loss = 0.5*norm(tensor_cat(Zx, Zy))^2 / batch_size  #likelihood under Normal Gaussian training 
    
    #gradients under Normal Gaussian training
    dZx = Zx / batch_size 
    dZy = Zy / batch_size 
    
    H.backward(dZx, dZy, Zx, Zy) #sets gradients of G wrt output and also logdet terms
    
    return (l2_loss, lgdet)
end

function get_minibatches(X, Y, batch_size)
    n_samples = size(X, 4)  # Assuming the number of samples is the 4th dimension
    indices = randperm(n_samples)
    minibatches = []

    for i in 1:batch_size:n_samples
        batch_indices = indices[i:min(i+batch_size-1, n_samples)]
        X_batch = X[:, :, :, batch_indices]
        Y_batch = Y[:, :, :, batch_indices]
        push!(minibatches, (X_batch, Y_batch))
    end

    return minibatches
end

batch_size = 64
minibatches = get_minibatches(X_train, Y_train, batch_size);

# Define network
n_hidden = 64
batchsize = 64
depth = 10

# Construct HINT network
H = NetworkConditionalHINT(n_in, n_hidden, depth; k1=1, k2=1, p1=0, p2=0)

# Training
nepochs = 30
lr      = 5f-4
lr_decay_step = 90

# compose adam optimizer with exponential learning rate decay
opt = Flux.Optimiser(ExpDecay(lr, .9f0, lr_decay_step, 1f-6), Flux.ADAM(lr))

loss_l2_list    = []
loss_lgdet_list = []

for ep = 1:nepochs
    for (X, Y) in minibatches

        losses = loss(H, X, Y)
        # loss_l2_list[j]    = losses[1]
        # loss_lgdet_list[j] = losses[2]
        push!(loss_l2_list, losses[1])
        push!(loss_lgdet_list, losses[2])

        # print("Iter : iteration=", j, "/", maxiter, ", batch=",
        #         "; f l2 = ",   loss_l2_list[j],
        #         "; f lgdet = ",loss_lgdet_list[j],
        #         "; f nll objective = ",loss_l2_list[j] - loss_lgdet_list[j], "\n")

        # Update params
        for p in get_params(H)
            Flux.update!(opt, p.data, p.grad)
        end
    end
end


### Check training objective log 

# gt_l2 = 0.5*nx*ny*n_in*2 #l2 norm of noise. Note: extra 2 factor since learning a 2 rv. joint distribution
# 
# fig, axs = subplots(3, 1, sharex=true, figsize=(10,7))
# fig.subplots_adjust(hspace=0)
# 
# axs[1].plot(loss_l2_list, color="black", linewidth=0.6);
# axs[1].axhline(y=gt_l2,color="red",linestyle="--",label="Normal Noise Likelihood")
# axs[1].set_ylabel("L2 Norm")
# axs[1].yaxis.set_label_coords(-0.05, 0.5)
# axs[1].legend()
# 
# axs[2].plot(loss_lgdet_list, color="black", linewidth=0.6);
# axs[2].set_ylabel("Log DET")
# axs[2].yaxis.set_label_coords(-0.05, 0.5)
# 
# axs[3].plot(loss_l2_list - loss_lgdet_list, color="black", linewidth=0.6);
# axs[3].set_ylabel("Full Objective")
# axs[3].yaxis.set_label_coords(-0.05, 0.5)
# axs[3].set_xlabel("Parameter Update")
# 
# folder = "08_plots/seed$seed/training_log/repeat_train=$repeat_train/"
# if !isdir(folder)
#     mkpath(folder)
# end
# savefig(joinpath(folder, "mv_training_log_seed=$seed" * "_nepochs=$nepochs" * "_n_train=$n_train" * "_n_noise=$n_noise" * "_repeat_train=$repeat_train" * "_repeat_noise=$repeat_noise"))


### Testing a Conditional Normalizing Flow

num_test_samples = 500;
Zx_test = randn(Float32,nx,ny,n_in, num_test_samples);
Zy_test = randn(Float32,nx,ny,n_in, num_test_samples);

X_test, Y_test = H.inverse(Zx_test, Zy_test);


### generated samples

# fig = figure(figsize=(10,5));
# subplot(1,2,1); title(L"x \sim p(x)")
# scatter(X_train[1,1,1,1:plot_num], X_train[1,1,2,1:plot_num]; alpha=0.4, label = L"x \sim p(x)");
# scatter(X_test[1,1,1,1:plot_num], X_test[1,1,2,1:plot_num]; alpha=0.4, color="orange", label = L"x \sim p_{\theta}(x) = H_\theta^{-1}(Zx, Zy)");
# xlabel(L"x_1"); ylabel(L"x_2");
# # xlim(-4,4); ylim(0,30);
# legend();
# 
# subplot(1,2,2); title(L"Noisy data $y = Ax + \epsilon$")
# scatter(Y_train[1,1,1,1:plot_num], Y_train[1,1,2,1:plot_num]; alpha=0.4, label = L"y \sim p(y)");
# scatter(Y_test[1,1,1,1:plot_num], Y_test[1,1,2,1:plot_num]; alpha=0.4, color="orange", label = L"y \sim p_{\theta}(y) = H_\theta^{-1}(Zy)");
# xlabel(L"x_1"); ylabel(L"x_2");
# legend();
# 
# folder = "08_plots/seed$seed/generated_samples/repeat_train=$repeat_train/"
# if !isdir(folder)
#     mkpath(folder)
# end
# savefig(joinpath(folder, "mv_generated_sample_seed=$seed" * "_nepochs=$nepochs" * "_n_train=$n_train" * "_n_noise=$n_noise" * "_repeat_train=$repeat_train" * "_repeat_noise=$repeat_noise"))


function sample(seed)
    ### Test inference of inverse problem given noisy data
    ### Form conditional distribution given observed data $p(x|y_{obs})$
    
    Random.seed!(seed)   # reset seed to ensure we are looking at the same `x_star` in each experiment
    x_star = rand(π_x, 1)
    y_obs = reshape(x_star, nx, ny, dim_model, 1)
    e_train = rand(π_ϵ, 1)
    y_obs = reshape(A * x_star + e_train, nx, ny, dim_model, 1)
    x_star = reshape(x_star, nx, ny, dim_model, 1);
    
    zy_fixed = H.forward_Y(y_obs);
    
    cond_sampling_size = 50
    Zx = randn(Float32, nx, ny, n_in, cond_sampling_size)
    X_post = H.inverse(Zx, zy_fixed.*ones(Float32, nx, ny, n_in, cond_sampling_size))[1];
    
    y_test = y_obs
    X_post_true = zeros(Float32,1,1,dim_model,cond_sampling_size)
    for i in 1:cond_sampling_size
        X_post_true[:,:,:,i] = post_dist_sample(y_test)
    end
    
    # Compute the Euclidean distances
    distances = [norm(X_post[1, 1, :, i] - x_star[1, 1, :, 1]) for i in 1:size(X_post)[end]]
    
    # Calculate the average distance
    ave_dist = mean(distances)

    println(ave_dist)
end


function plot_samples()
    fig = figure(figsize=(15,5));
    subplot(1,3,1); title(L"Ground truth model $x^{*}$")
    scatter(X_train[1,1,1,1:plot_num], X_train[1,1,2,1:plot_num]; alpha=0.4, label = L"x \sim p(x,y)");
    scatter(x_star[1,1,1,1], x_star[1,1,2,1]; marker="*", color="black", label = L"x^{*}");
    xlabel(L"x_1"); ylabel(L"x_2");
    # xlim(-4,4); ylim(0,30);
    legend();
    
    subplot(1,3,2); title(L"Observed Noisy data $y_{obs} = Ax^{*} + \epsilon$")
    scatter(Y_train[1,1,1,1:plot_num], Y_train[1,1,2,1:plot_num]; alpha=0.4, label = L"y \sim p(x,y)");
    scatter(y_obs[1,1,1,1], y_obs[1,1,2,1]; marker="*", color="red", label = L"y_{obs}");
    xlabel(L"x_1"); ylabel(L"x_2");
    legend();
    
    subplot(1,3,3); title(L"Conditional Distribution given observed data $y_{obs}$")
    scatter(X_train[1,1,1,1:plot_num], X_train[1,1,2,1:plot_num]; alpha=0.4, label = L"x \sim p(x,y)");
    scatter(X_post[1,1,1,:], X_post[1,1,2,:]; alpha=0.4, color="red", label = L"x \sim p_{\theta}(x | y_{obs})");
    scatter(x_star[1,1,1,1], x_star[1,1,2,1]; marker="*",color="black",  label = L"x^{*}");
    scatter(X_post_true[1,1,1,:], X_post_true[1,1,2,:]; alpha=0.4, color="green", label = "true "*L"p(x | y_{obs})");
    xlabel(L"x_1"); ylabel(L"x_2");
    # xlim(-4,4); ylim(0,30);
    legend()
    
    folder = "08_plots/seed$seed/posterior/repeat_train=$repeat_train/"
    if !isdir(folder)
        mkpath(folder)
    end
    savefig(joinpath(folder, "mv_posterior_seed=$seed" * "_nepochs=$nepochs" * "_n_train=$n_train" * "_n_noise=$n_noise" * "_repeat_train=$repeat_train" * "_repeat_noise=$repeat_noise"))
end



