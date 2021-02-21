using LinearAlgebra

function  ConvHullProj(x, V,alpha)
    x = x'
    a2 = maximum(svdvals(V))
    L = a2.^2
    t = 1/L
    m, d = size(V)
    if alpha == -1
        alpha = zeros(d,1)
        alpha[1] = 1
    end
    v = alpha
    theta = 1
    alpha_old = 1000*ones(d,1)
    counter = 1
    while (norm(alpha - alpha_old) > 1e-3 || counter < 5)
        alpha_old = alpha
        y = alpha + theta*(v-alpha)
        temp = y - t*(transpose(V)*V*y - transpose(V)*x)
        temp = reshape(temp,(length(temp)))
        alpha = projsplx(temp)
        v = alpha + (alpha - alpha_old)/theta
        theta = (sqrt(theta^4+ 4*theta^2) - theta^2)/2
        counter = counter + 1
    end
    y = V*alpha;
    return y, alpha
end

function projsplx(b)
    τ = 1
    n = length(b)
    bget = false
    idx = sortperm(b, rev=true)
    tsum = 0

    @inbounds for i = 1:n-1
        tsum += b[idx[i]]
        tmax = (tsum - τ)/i
        if tmax ≥ b[idx[i+1]]
            bget = true
            break
        end
    end

    if !bget
        tmax = (tsum + b[idx[n]] - τ) / n
    end

    @inbounds for i = 1:n
        b[i] = max(b[i] - tmax, 0)
    end
    return b
end
