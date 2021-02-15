using LinearAlgebra, StatsBase, Gurobi

include("InitializeMIP.jl")
include("SAA.jl")


##
env = Gurobi.Env()
##
m = 40
k = 2
n = 50
ell = 45
sigma = 0.1
##
H = rand(k*n)
idx = sample(1:k*n, k*n - ell, replace = false)
H[idx] = zeros(length(idx))
H = reshape(H,(k,n))
W = rand(m, k)
W = abs.(W)
for i = 1:m
    W[i,:] = W[i,:]/sum(W[i,:])
end
X = W*H
X0 = X
Z = randn(size(X))*sigma
X = Z + X
idx = findall(x -> x<0,X)
X[idx] = zeros(length(idx))
H0 = H + zeros(size(H))
W0 = W + zeros(size(W))
##
H_min, A, W_min, gap = InitializeMIP(X, H0 .>0, k, ell, 1, env, 8, 8, 0.05)
println(gap)
# SAA(X, H, W, zeros(k,m), 1,  ell-10, 150, 1e-4, 8, 30, false)
