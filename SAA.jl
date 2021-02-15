using LinearAlgebra, TSVD

#################################################################################################
# Input arguments. X: The data matrix, H: The initial archetypes matrix                         #
# W,Wtilde: The initial simplex matrices, lambda: The target value of lambda                    #
# ell: The sparsity level, tol: Tolerance of convergence                                        #
# len: The number of points of continuation, lambda_max: The starting value for continuation    #
# path: if ~= 0, the complete regularization path is returned.                                  #
#################################################################################################

include("AAproxblock.jl")

function SAA(X, H, W, Wtilde, lambda,  ell, maxIter, tol, len, lambda_max, path)
    k, n = size(H)
    m, k = size(W)
    f = zeros(maxIter)
    if path
        H_tot = zeros(k, n, len)
        W_tot = zeros(m, k, len)
        Wtilde_tot = zeros(k, m, len)
        f_tot = zeros(maxIter, len)
    end
    lambda_final = lambda
    L = 10 .^ (range(log(10,lambda_final),stop=log(10,lambda_max),length=len))
    for ilen = 1:len
        lambda = L[len - ilen + 1]
        H, W, Wtilde, f =  AAproxblock(X, H, W, Wtilde, lambda,  ell, maxIter, tol)
        if path
            H_tot[:,:,ilen] = H
            W_tot[:,:,ilen] = W
            Wtilde_tot[:,:,ilen] = Wtilde
            f_tot[1:length(f), ilen] = f
        end
    end
    if path
        return H_tot, W_tot, Wtilde_tot, f_tot
    end
    return H, W, Wtilde, f
end
