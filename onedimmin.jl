using LinearAlgebra

include("ConvHullProj.jl")


function onedimmin(X, H, A, B, i, j, lambda, m, n ,k)
    max_iter = 50
    f = zeros(max_iter,1)
    f[1] = R(X, H, A, B,lambda)
    counter = 1
    diff = 1e10
    t = 0
    while (diff >= f[1]/1e8 && counter < max_iter)
        U = X - A*H
        V = H - B*X
        global t = max(-H[i,j],(transpose(A[:,i])*U[:,j] - lambda*V[i,j])/(lambda + norm(A[:,i])^2))
        H[i,j] = t + H[i,j]
        temp, a = ConvHullProj(transpose(H[i,:]), transpose(X),(B[i,:]))
        B[i,:] = transpose(a)
        for r=1:m
            tt, a = ConvHullProj(transpose(X[r,:]), transpose(H),(A[r,:]))
            A[r,:] = transpose(a)
        end
        counter = counter + 1
        f[counter] = R(X,H,A,B, lambda)
        diff = f[counter-1] - f[counter]
    end
    f = f[1:counter]
    return t+H[i,j], A, B, f[counter]
end

function R(X, H, A, B, lambda)
	return norm(X - A*H)^2 + lambda*norm(H - B*X)^2
end
