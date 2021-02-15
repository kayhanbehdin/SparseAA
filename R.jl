function R(X, H, A, B, lambda)
	return norm(X - A*H)^2 + lambda*norm(H - B*X)^2
end
