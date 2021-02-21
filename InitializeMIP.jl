using Gurobi, JuMP, LinearAlgebra, TSVD, Dates

include("ConvHullProj.jl")


#################################################################################################
# Input arguments. X: The data matrix, H: The initial archetypes matrix, Z0: Initial support	#
# k: Rank, ell: Sparsity level, M: An upper bound on H_i,j, env: Gurobi license object		    #
# maxTime: Maximum runtime in minutes, maxIter: Maximum number of cuts                          #
# minGap: Minimum dual gap tolerance  													        #
#################################################################################################


function InitializeMIP(X, Z0, k, ell, M, env, maxTime, maxIter, minGap)
	t0 = now()
	_,a2,_ = tsvd(X)
	lambda_3 = 1/(2*a2[1].^2)
	m, n = size(X)
	Z = Z0
	model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(env), "OutputFlag" => 0))
	set_optimizer_attribute(model, "TimeLimit", maxTime*60)
	@variable(model, z[1:k,1:n],Bin)
	@variable(model, eta)
	@constraint(model, sum(sum(z[i,j] for i=1:k) for j = 1:n) <= ell)
	sense = MOI.MIN_SENSE
	@objective(model, sense,eta)
	counter = 1
	ETA = -1000
	c = -10
	gap = zeros(maxIter)
	diff = zeros(maxIter)
	cost = zeros(maxIter)
	c_min = 1e15
	z_min = zeros(k,n)
	H_min = zeros(k,n)
	W_min = zeros(k,m)
	while(counter <= maxIter)
		println(string("MIP cut: ",counter))
		c, H, W = F(Z, X, M, lambda_3)
		Lambda = 2*(W*X - H)
		indd = findall(x-> x<0,Lambda)
		Lambda[indd] = zeros(size(indd))
		grad = -M*Lambda
		if (c<c_min)
			Z_min = Z + zeros(k,n)
			c_min = c
			W_min = W + zeros(k,m)
			H_min = H + zeros(k,n)
		end
		@constraint(model, eta >= c + sum(sum( grad[i,j]*(z[i,j]-Z[i,j]) for i=1:k) for j = 1:n))
		cost[counter] = c
		diff[counter] = ETA - c
		status=optimize!(model)
		Z = value.(z)
		ETA = value.(eta)
		gap[counter] = -(ETA-c_min)/c_min
		if(gap[counter] < minGap)
			break
		end
		t2 = now()
		diff_t = Dates.value(convert(Dates.Millisecond, t2-t0))
		if (diff_t/1000 > maxTime*60)
			break
		end
		counter = counter + 1

	end
	e = zeros(k,1)
	e[1] = 1
	A = zeros(m,k)
	for i = 1:1:m
		t, a = ConvHullProj(transpose(X[i,:]), transpose(H_min), e)
		A[i,:] = transpose(a)
	end
	H = H_min + zeros(k,n)
	W = A + zeros(m,k)
	Wtilde = W_min + zeros(k,m)
	return H, W, Wtilde, gap
end

function F(Z, X, M, eta_3)
	m,n = size(X)
	k,n = size(Z)
	e = zeros(m,1)
	e[1] = 1
	Wt = ones(k,1)*e'
	H = zeros(k,n)
	counter = 1
	diff = 1
	while (counter < 200 && diff>1e-4)
		Hold = H
		H = Wt*X
		ind1 = findall(x -> x<0, H)
		H[ind1] = zeros(size(ind1))
		Htemp = H - M*Z
		ind2 = findall(x -> x>0, Htemp)
		H[ind2] = M*Z[ind2]
		Wt = Wt - 2*eta_3*(Wt*X-H)*X'
		for jj=1:k
			Wt[jj,:] = transpose(projsplx(Wt[jj,:]))
		end
		diff = norm(H-Hold)/norm(H)
		counter = counter + 1

	end

	return norm(H-Wt*X)^2, H, Wt
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
