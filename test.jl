
a, b, c = parse.(Int, ARGS)

function loss(G, X)
    batch_size = size(X)[end]
    
    Z, lgdet = G.forward(X)
    
    function l2_loss(Z)
        # likelihood = -0.5*norm(Z)^2   # for sanity check. Recover normalizing flow result.
    
        cons = loggamma(0.5*(ν+d)) - loggamma(0.5*ν) - 0.5*d*log(pi*ν)
        likelihood = cons - 0.5*(ν + d) * log(1 + (1/ν) * norm(Z)^2)
    
        return -likelihood / batch_size
    end
    
    # use AD
    θ = Flux.params([Z])
    back = Flux.Zygote.pullback(() -> l2_loss(Z), θ)[2]
    grad = back(1f0)
    logprob_grad = grad[θ[1]]
    
    G.backward(logprob_grad, Z)  # sets gradients of G wrt output and also logdet terms
    
    return (l2_loss(Z), lgdet, logprob_grad, Z)
end
