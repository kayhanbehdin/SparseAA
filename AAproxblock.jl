using LinearAlgebra, TSVD


function AAproxblock(X,H,A, B, lambda,  ell, max_iter, tol)
    _,a2,_ = tsvd(X)
    a2 = a2[1]
    LAMBDA = 2*lambda*a2.^2
    f = zeros(max_iter,1)
    f[1] = R(X, H, A, B,lambda )
    counter = 1
    diff = f[1]
    k,n = size(H)
    m, k = size(A)
    while (diff >= f[1]*tol && counter < max_iter)
        _,a2,_ = tsvd(A')
        a2 = a2[1]
        step = 1 / (2*a2^2 + 2*lambda)

        pi = H - step*(-2*transpose(A)*((X-A*H)) + 2*lambda*(H-B*X))
        I = pi.<= 0
        pi[I] = zeros(sum(I),1)
        H = pi
        if ell < k*n
            hh = H[:]
            I = sortperm( hh[:])
            hh[I[1:n*k - ell]] = zeros(size(I[1:n*k - ell]))
            H = reshape(hh,(k,n))
        end
        _,a2,_ = tsvd(H)
        a2 = a2[1]
        step = 1 / (2*a2^2);
        A = A - 2*step*(-((X - A*H))*transpose(H))
        for jj=1:m
            A[jj,:] = transpose(projsplx(A[jj,:]))
        end
        if (lambda != 0)
            step = 1/LAMBDA
            B = B - 2*step*(-lambda*((H - B*X))*X')
            for jj = 1:k
                B[jj,:] = transpose(projsplx(B[jj,:]))
            end
        end
        counter = counter + 1;
        f[counter] = R(X,H,A,B, lambda)
        diff = f[counter-1] - f[counter]
    end
    f = f[1:counter]
    return H, A, B, f
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

function R(X, H, A, B, lambda)
	return norm(X - A*H)^2 + lambda*norm(H - B*X)^2
end
