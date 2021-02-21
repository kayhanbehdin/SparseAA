include("onedimmin.jl")


#################################################################################################
# Input arguments. X: The data matrix, H: The initial archetypes matrix                         #
# W,Wtilde: The initial simplex matrices, lambda: The value of lambda                           #
# ell: The sparsity level,                                                                      #
# Size: The number of candidate coordinates etering/leaving considered in each iteration        #
# The number of pairs considered in each iteration is Size^2                                    #
# max_iter: Maximum number of iterations          											    #
#################################################################################################


function SAALS(X,H, W, Wt, lambda,  ell, Size, max_iter)
    Cost = zeros(max_iter)
    m, n = size(X)
    k, n = size(H)
    counter = 0
    supchange = 0
    while (counter < max_iter)
        counter = counter + 1
		println(string("Local Search Iteration: ",counter))
        G = zeros(k,n)
        Z = H.>1e-10
        Z = Z + zeros(k,n)
        G = 2*lambda*(H - Wt*X)
        for i = 1:m
            G = G + 2*W[i,:]*(W[i,:]'*H-X[i,:]')
        end
        G = Z*1000 + (ones(k,n)-Z).*G
        temp = sort(G[:])
        thresh = temp[Size]
        set1 = findall(x -> x<=thresh, G)
        Htemp = H + (ones(k,n)-Z)*1e6
        temp = sort(Htemp[:])
        thresh = temp[Size]
        set2 = findall(x -> x<=thresh, Htemp)
        cost = R(X, H, W, Wt, lambda)
        Cost[counter] = cost
        counter2 = 0
        while (counter2 < Size)
            i = rand(1:k)
            j = rand(1:n)
            if(Z[i,j] == 0)
                continue
            end
            Htemp = H + zeros(k,n)
            t_best, W0, Wt0, u_best = onedimmin(X, Htemp, W, Wt, i, j, lambda, m, n ,k)
            if (u_best < cost)
                H[i,j] = t_best
                W = W0 + zeros(m,k)
                Wt = Wt0 + zeros(k,m)
            end
            counter2 = counter2 + 1

        end

        for i1 = 1:Size
            for j1 = 1:Size
                ind1 = set2[i1]
                Htemp = H + zeros(k,n)
                Htemp[ind1] = 0

                ind2 = set1[j1]
                i = ind2[1]
                j = ind2[2]

                t_best, W0, Wt0, u_best = onedimmin(X, Htemp, W, Wt, i, j, lambda, m, n ,k)
                   if (u_best < cost)
                       H[i,j] = t_best
                       H[ind1] = 0
                       W = W0 + zeros(m,k)
                       Wt = Wt0 + zeros(k,m)
                       supchange = supchange + 1
                   end
            end
        end

    end
    return H,W,Wt,Cost, supchange
end

function R(X, H, A, B, lambda)
	return norm(X - A*H)^2 + lambda*norm(H - B*X)^2
end
