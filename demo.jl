using LinearAlgebra, StatsBase, Gurobi

include("InitializeMIP.jl")
include("SAA.jl")
include("SAALS.jl")
include("L.jl")


##
env = Gurobi.Env()
##
m = 40
k = 20
n = 500
ell = k*n
ell_desired = 10*n
sigma = 0.5
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
H_init, W_init, Wtilde_init, gap = InitializeMIP(X, zeros(k,n), k, ell_desired, 1, env, 8, 10, 0.05)
H, W, Wtilde, f = SAA(X, H_init, W_init, Wtilde_init, 1,  ell_desired, 150, 1e-4, 8, 30, false)
H, W, Wtilde, f_ls, supchange = SAALS(X,H, W, Wtilde, 1,  ell_desired, 2, 4)

println(string("Objective achieved by continuation: ", f[end]))
println(string("Objective achieved by local search: ", f_ls[end]))
println(string("Number of support changes by local search: ", supchange))
println(string("Weak robustness L(H_0,H): ", L(H0, H)))
println(string("Strong robustness L(H,H_0): ", L(H, H0)))
