using LinearAlgebra, StatsBase, Gurobi

include("InitializeMIP.jl")
include("SAA.jl")
include("SAALS.jl")


##
env = Gurobi.Env()
##
m = 40
k = 20
n = 500
ell = 20*200
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
X0 = X + zeros(m,n)
Z = randn(size(X))*sigma
X = Z + X
idx = findall(x -> x<0,X)
X[idx] = zeros(length(idx))
H0 = H + zeros(size(H))
W0 = W + zeros(size(W))
##
H_init, W_init, Wtilde_init, gap = InitializeMIP(X, zeros(k,n), k, ell-10, 1, env, 8, 10, 0.05)
H, W, Wtilde, f = SAA(X, H_init, W_init, Wtilde_init, 1,  ell-10, 150, 1e-4, 8, 30, false)
H,W,Wt,f_ls, supchange = SAALS(X,H, W, Wtilde, 1,  ell-10, 2, 5)

println(string("Objective achieved by continuation: ", f[end]))
println(string("Objective achieved by local search: ", f_ls[end]))
println(string("Number of support changes by local search: ", supchange))

# println(norm(X-W_init*H_init)^2 + norm(H_init - Wtilde_init*X)^2 )
# println(sum(H0 .> 0) - sum(H.> 0))
# println(sum(H.< 0))
# println(f)
